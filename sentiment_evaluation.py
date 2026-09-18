
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------------- #
# Core metrics                                                                   #
# ----------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """cm[i, j] = # samples with true class i predicted as class j."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray):
    """Return per-class (precision, recall, f1) arrays from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)   # column sums
    true_pos = cm.sum(axis=1).astype(np.float64)   # row sums (support)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, true_pos


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int, labels: Optional[List[str]] = None) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1, support = per_class_prf(cm)

    n = len(y_true)
    accuracy = float((y_true == y_pred).sum()) / n if n else 0.0
    macro_f1 = float(f1.mean()) if num_classes else 0.0
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    w = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    weighted_f1 = float((f1 * w).sum())
    weighted_precision = float((precision * w).sum())
    weighted_recall = float((recall * w).sum())

    labels = labels or [str(i) for i in range(num_classes)]
    return {
        "n": int(n),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": {labels[c]: {"precision": float(precision[c]),
                                  "recall": float(recall[c]),
                                  "f1": float(f1[c]),
                                  "support": int(support[c])}
                      for c in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


# ----------------------------------------------------------------------------- #
# Pretty printing                                                                #
# ----------------------------------------------------------------------------- #
def print_metrics(name: str, m: Dict) -> None:
    print(f"\n=== {name} (n={m['n']}) ===")
    print(f"  accuracy          {m['accuracy']:.4f}")
    print(f"  macro    P/R/F1   {m['macro_precision']:.4f} / "
          f"{m['macro_recall']:.4f} / {m['macro_f1']:.4f}")
    print(f"  weighted P/R/F1   {m['weighted_precision']:.4f} / "
          f"{m['weighted_recall']:.4f} / {m['weighted_f1']:.4f}")
    print("  per-class:")
    for lab, d in m["per_class"].items():
        print(f"    {lab:<9} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"F1={d['f1']:.3f} (support={d['support']})")
    print("  confusion matrix (rows=true, cols=pred):")
    labels = m["labels"]
    header = "        " + " ".join(f"{l[:6]:>7}" for l in labels)
    print(header)
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"    {labels[i][:6]:>6} " + " ".join(f"{v:>7}" for v in row))


# ----------------------------------------------------------------------------- #
# Persistence                                                                    #
# ----------------------------------------------------------------------------- #
def write_results_csv(path: str, split_metrics: Dict[str, Dict]) -> None:
    """One row per split with the headline metrics."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["split", "n", "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for split, m in split_metrics.items():
            writer.writerow({"split": split, **{k: m[k] for k in cols[1:]}})


def dump_history(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------------- #
# CLI: evaluate best head on the test split                                      #
# ----------------------------------------------------------------------------- #
def main():
    import torch
    import config
    from xlmr_sentiment import SentimentHead, rating_to_sentiment, select_rows
    from review_embeddings import load_cache

    if not os.path.exists(config.XLMR_SENTIMENT_BEST):
        raise FileNotFoundError(
            f"No sentiment head at {config.XLMR_SENTIMENT_BEST}. "
            f"Train first (python xlmr_sentiment.py).")
    cache = load_cache()
    ckpt = torch.load(config.XLMR_SENTIMENT_BEST, map_location="cpu", weights_only=False)
    head = SentimentHead(ckpt["hidden"], ckpt["num_classes"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    split_metrics = {}
    for split in ("train", "validation", "test"):
        s = cache["splits"].get(split)
        if s is None:
            continue
        emb, y = select_rows(s, require_text=config.SENTIMENT_REQUIRE_TEXT)
        if len(y) == 0:
            continue
        with torch.no_grad():
            logits = head(emb.float())
            pred = logits.argmax(dim=1).numpy()
        m = compute_metrics(y, pred, config.SENTIMENT_NUM_CLASSES,
                            config.SENTIMENT_LABELS)
        split_metrics[split] = m
        print_metrics(split, m)

    write_results_csv(config.XLMR_SENTIMENT_RESULTS, split_metrics)
    print(f"\nwrote {config.XLMR_SENTIMENT_RESULTS}")


if __name__ == "__main__":
    main()
