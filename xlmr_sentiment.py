from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from sentiment_evaluation import (
    compute_metrics, print_metrics, write_results_csv, dump_history,
)


# ----------------------------------------------------------------------------- #
# Rating -> weak sentiment label (configurable)                                  #
# ----------------------------------------------------------------------------- #
def rating_to_sentiment(ratings: np.ndarray) -> np.ndarray:
    """Map numeric ratings to {0:negative, 1:neutral, 2:positive}. NaN -> -1."""
    r = np.asarray(ratings, dtype=np.float64)
    labels = np.full(r.shape, -1, dtype=np.int64)
    labels[r <= config.SENTIMENT_NEG_MAX] = 0
    labels[r == config.SENTIMENT_NEU_VALUE] = 1
    labels[r >= (config.SENTIMENT_NEU_VALUE + 1)] = 2
    return labels


def select_rows(split_cache: Dict, require_text: bool = True
                ) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (embeddings, labels) for a cached split, dropping rows without a
    valid rating (and, if require_text, rows with empty review text)."""
    emb = split_cache["embeddings"].float()
    ratings = split_cache.get("rating")
    if ratings is None:
        return emb[:0], np.zeros(0, dtype=np.int64)
    labels = rating_to_sentiment(ratings)
    keep = labels >= 0
    if require_text and split_cache.get("has_text") is not None:
        keep = keep & split_cache["has_text"]
    return emb[torch.from_numpy(keep)], labels[keep]


# ----------------------------------------------------------------------------- #
# Head                                                                           #
# ----------------------------------------------------------------------------- #
class SentimentHead(nn.Module):
    """Single linear layer: ŝ = softmax(W_s h + b_s). Returns logits."""

    def __init__(self, hidden: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    @torch.no_grad()
    def predict(self, h: torch.Tensor):
        logits = self.forward(h)
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs


# ----------------------------------------------------------------------------- #
# Training                                                                       #
# ----------------------------------------------------------------------------- #
def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    if inv.sum() == 0:
        return torch.ones(num_classes)
    w = inv / inv.sum() * num_classes           # normalise to mean 1
    return torch.tensor(w, dtype=torch.float32)


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_head(train_emb, train_labels, val_emb, val_labels,
               hidden: int, device: torch.device) -> Tuple[SentimentHead, list, Dict]:
    set_seed(config.RANDOM_SEED)
    head = SentimentHead(hidden, config.SENTIMENT_NUM_CLASSES).to(device)

    present = np.unique(train_labels)
    if len(present) < 2:
        print(f"  [warn] only {len(present)} sentiment class(es) present in TRAIN "
              f"({present.tolist()}). Training will proceed but is degenerate — "
              f"expected when ratings lack variance (e.g. all-5-star sources).")

    weight = (_class_weights(train_labels, config.SENTIMENT_NUM_CLASSES).to(device)
              if config.SENTIMENT_CLASS_WEIGHTING else None)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.SENTIMENT_LR,
                                 weight_decay=config.SENTIMENT_WEIGHT_DECAY)

    Xtr = train_emb.to(device)
    ytr = torch.from_numpy(train_labels).long().to(device)
    has_val = val_emb is not None and len(val_labels) > 0
    if has_val:
        Xva = val_emb.to(device)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = len(ytr)
    history = []
    best_metric = -1.0
    best_state = None
    best_epoch = -1
    patience = config.SENTIMENT_PATIENCE

    for epoch in range(1, config.SENTIMENT_EPOCHS + 1):
        head.train()
        order = rng.permutation(n)
        total, nb = 0.0, 0
        for s in range(0, n, config.SENTIMENT_BATCH_SIZE):
            idx = order[s:s + config.SENTIMENT_BATCH_SIZE]
            xb = Xtr[idx]
            yb = ytr[idx]
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item(); nb += 1
        row = {"epoch": epoch, "train_loss": round(total / max(nb, 1), 6)}

        if has_val:
            head.eval()
            with torch.no_grad():
                vpred = head(Xva).argmax(dim=1).cpu().numpy()
            vm = compute_metrics(val_labels, vpred, config.SENTIMENT_NUM_CLASSES,
                                config.SENTIMENT_LABELS)
            row["val_accuracy"] = round(vm["accuracy"], 6)
            row["val_macro_f1"] = round(vm["macro_f1"], 6)
            row["val_weighted_f1"] = round(vm["weighted_f1"], 6)
            cur = vm[config.SENTIMENT_SELECT_METRIC]
            if cur > best_metric + 1e-6:
                best_metric = cur; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}
                patience = config.SENTIMENT_PATIENCE
            else:
                patience -= 1
            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} "
                  f"val_acc={row['val_accuracy']:.4f} "
                  f"val_macroF1={row['val_macro_f1']:.4f} "
                  f"(best={best_metric:.4f}@{best_epoch}) patience={patience}")
            history.append(row)
            if patience <= 0:
                print(f"  early stopping at epoch {epoch}")
                break
        else:
            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} (no validation)")
            history.append(row)

    if best_state is not None:
        head.load_state_dict(best_state)
    meta = {"best_epoch": best_epoch, "best_val_metric": best_metric,
            "select_metric": config.SENTIMENT_SELECT_METRIC}
    return head, history, meta


def save_head(head: SentimentHead, hidden: int, meta: Dict):
    config.ensure_checkpoint_dir()
    torch.save({
        "state_dict": head.state_dict(),
        "hidden": hidden,
        "num_classes": config.SENTIMENT_NUM_CLASSES,
        "labels": config.SENTIMENT_LABELS,
        "rating_map": {"neg_max": config.SENTIMENT_NEG_MAX,
                       "neu_value": config.SENTIMENT_NEU_VALUE},
        "meta": meta,
    }, config.XLMR_SENTIMENT_BEST)


# ----------------------------------------------------------------------------- #
# Orchestration                                                                  #
# ----------------------------------------------------------------------------- #
def run():
    from review_embeddings import load_cache, get_device
    device = get_device()
    print(f"[sentiment] device={device}")
    cache = load_cache()
    hidden = cache["hidden"]

    tr = cache["splits"].get("train")
    if tr is None:
        raise RuntimeError("No train split in cache. Run review_embeddings.py first.")
    Xtr, ytr = select_rows(tr, require_text=config.SENTIMENT_REQUIRE_TEXT)
    va = cache["splits"].get("validation")
    Xva, yva = (select_rows(va, require_text=config.SENTIMENT_REQUIRE_TEXT)
                if va is not None else (None, np.zeros(0, dtype=np.int64)))
    print(f"[sentiment] train rows={len(ytr)} "
          f"class dist={np.bincount(ytr, minlength=3).tolist() if len(ytr) else []} "
          f"| val rows={len(yva)}")

    head, history, meta = train_head(Xtr, ytr, Xva, yva, hidden, device)
    save_head(head, hidden, meta)
    print(f"[sentiment] saved head -> {config.XLMR_SENTIMENT_BEST} "
          f"(best {meta['select_metric']}={meta['best_val_metric']:.4f}"
          f"@{meta['best_epoch']})")

    # final evaluation on all available splits
    split_metrics = {}
    head.eval()
    for name in ("train", "validation", "test"):
        s = cache["splits"].get(name)
        if s is None:
            continue
        X, y = select_rows(s, require_text=config.SENTIMENT_REQUIRE_TEXT)
        if len(y) == 0:
            continue
        with torch.no_grad():
            pred = head(X.float().to(device)).argmax(dim=1).cpu().numpy()
        m = compute_metrics(y, pred, config.SENTIMENT_NUM_CLASSES,
                            config.SENTIMENT_LABELS)
        split_metrics[name] = m
        print_metrics(name, m)

    write_results_csv(config.XLMR_SENTIMENT_RESULTS, split_metrics)
    dump_history(config.XLMR_SENTIMENT_HISTORY,
                 {"history": history, "meta": meta})
    print(f"[sentiment] wrote {config.XLMR_SENTIMENT_RESULTS} and "
          f"{config.XLMR_SENTIMENT_HISTORY}")
    return head, split_metrics


if __name__ == "__main__":
    run()
