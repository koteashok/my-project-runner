"""
lightgcn_train.py
=================
Train LightGCN with BPR on the Phase 1 interaction splits.

Pipeline:
    load train/val/test  ->  build Â from TRAIN  ->  BPR training with efficient
    negative sampling  ->  per-epoch validation (Recall/NDCG@K)  ->  early stopping
    ->  best-model checkpoint  ->  training-history CSV.

Artefacts:
    results/checkpoints/lightgcn_best.pt
    results/lightgcn_training_history.csv

Run:
    python lightgcn_train.py
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict

import numpy as np
import torch

import config
from lightgcn import LightGCN, build_norm_adjacency, set_seed, get_device
from lightgcn_evaluate import (
    load_splits, build_pos_dict, infer_sizes, evaluate_ranking, merge_exclusions,
)


# ----------------------------------------------------------------------------- #
# Efficient negative sampling                                                    #
# ----------------------------------------------------------------------------- #
class NegativeSampler:
    """Uniform negatives with rejection: a sampled item must not be a TRAIN
    positive of the corresponding user."""

    def __init__(self, user_pos: Dict[int, np.ndarray], num_items: int,
                 seed: int = 42):
        self.num_items = num_items
        # frozensets give O(1) membership checks
        self.user_pos = {u: set(items.tolist()) for u, items in user_pos.items()}
        self.rng = np.random.RandomState(seed)

    def sample(self, users: np.ndarray) -> np.ndarray:
        """One negative per user in `users` (vectorised with a small resample
        loop only for the few collisions)."""
        n = len(users)
        negs = self.rng.randint(0, self.num_items, size=n)
        # resample any negative that collides with a user's positive set
        for _ in range(20):  # bounded resampling passes
            collide = np.array(
                [negs[k] in self.user_pos.get(int(users[k]), ()) for k in range(n)],
                dtype=bool)
            if not collide.any():
                break
            negs[collide] = self.rng.randint(0, self.num_items, size=int(collide.sum()))
        return negs.astype(np.int64)


# ----------------------------------------------------------------------------- #
# Training                                                                       #
# ----------------------------------------------------------------------------- #
def train_one_epoch(model, optimizer, users_arr, pos_arr, sampler, device,
                    batch_size, rng) -> dict:
    model.train()
    order = rng.permutation(len(users_arr))
    users_arr = users_arr[order]
    pos_arr = pos_arr[order]

    total, total_bpr, total_reg, n_batches = 0.0, 0.0, 0.0, 0
    for start in range(0, len(users_arr), batch_size):
        bu = users_arr[start:start + batch_size]
        bp = pos_arr[start:start + batch_size]
        bn = sampler.sample(bu)

        u = torch.from_numpy(bu).to(device)
        p = torch.from_numpy(bp).to(device)
        ng = torch.from_numpy(bn).to(device)

        optimizer.zero_grad()
        loss, bpr, reg = model.bpr_loss(u, p, ng)
        loss.backward()
        optimizer.step()

        total += loss.item()
        total_bpr += float(bpr)
        total_reg += float(reg)
        n_batches += 1

    return {"loss": total / max(n_batches, 1),
            "bpr": total_bpr / max(n_batches, 1),
            "reg": total_reg / max(n_batches, 1)}


def save_checkpoint(model, path, extra: dict):
    config.ensure_checkpoint_dir()
    # Persist only the learned ego embeddings; the sparse graph is rebuilt from
    # `train` at load time, so it need not bloat the checkpoint on large graphs.
    emb_state = {k: v for k, v in model.state_dict().items()
                 if k.startswith("user_embedding") or k.startswith("item_embedding")}
    payload = {
        "model_state_dict": emb_state,
        "num_users": model.num_users,
        "num_items": model.num_items,
        "embedding_dim": model.embedding_dim,
        "num_layers": model.num_layers,
        "reg_lambda": model.reg_lambda,
    }
    payload.update(extra)
    torch.save(payload, path)


def train():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"[train] device={device}  seed={config.RANDOM_SEED}")

    # ---- data -------------------------------------------------------------
    train_df, val_df, test_df = load_splits()
    M, N = infer_sizes(train_df, val_df, test_df)
    print(f"[train] users(M)={M:,}  items(N)={N:,}  "
          f"train_interactions={len(train_df):,}")

    users_arr = train_df["user_idx"].to_numpy().astype(np.int64)
    pos_arr = train_df["item_idx"].to_numpy().astype(np.int64)

    train_pos = build_pos_dict(train_df, N)
    val_pos = build_pos_dict(val_df, N)

    # ---- graph + model ----------------------------------------------------
    norm_adj, norm_R, _, _ = build_norm_adjacency(users_arr, pos_arr, M, N, device)
    model = LightGCN(
        num_users=M, num_items=N, norm_adj=norm_adj,
        embedding_dim=config.EMBEDDING_DIM, num_layers=config.NUM_LAYERS,
        reg_lambda=config.BPR_REG_LAMBDA, emb_init_std=config.EMB_INIT_STD,
        norm_R=norm_R,
    ).to(device)
    print(f"[train] LightGCN d={config.EMBEDDING_DIM} K={config.NUM_LAYERS} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    sampler = NegativeSampler(train_pos, N, seed=config.RANDOM_SEED)
    rng = np.random.RandomState(config.RANDOM_SEED)

    can_validate = len(val_pos) > 0
    metric_key = f"{config.EVAL_METRIC}@{config.EVAL_K}"
    if not can_validate:
        print("[train] no validation split — training will run all epochs and "
              "checkpoint the final model (early stopping disabled).")

    # ---- loop -------------------------------------------------------------
    history = []
    best_metric = -1.0
    best_epoch = -1
    patience_left = config.EARLY_STOP_PATIENCE

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, optimizer, users_arr, pos_arr, sampler,
                             device, config.BATCH_SIZE, rng)
        row = {"epoch": epoch, "train_loss": round(tr["loss"], 6),
               "bpr": round(tr["bpr"], 6), "reg": round(tr["reg"], 6),
               "seconds": round(time.time() - t0, 3)}

        if can_validate and (epoch % config.EVAL_EVERY == 0):
            metrics = evaluate_ranking(model, val_pos, train_pos, N,
                                       config.TOP_K, device,
                                       config.EVAL_USER_BATCH)
            for k in config.TOP_K:
                row[f"val_recall@{k}"] = round(metrics[f"recall@{k}"], 6)
                row[f"val_ndcg@{k}"] = round(metrics[f"ndcg@{k}"], 6)
            cur = metrics[metric_key]

            improved = cur > best_metric + 1e-6
            if improved:
                best_metric = cur
                best_epoch = epoch
                patience_left = config.EARLY_STOP_PATIENCE
                save_checkpoint(model, config.LIGHTGCN_BEST_PATH,
                                {"epoch": epoch, "best_metric": best_metric,
                                 "metric_key": metric_key,
                                 "config": config.as_dict()})
            else:
                patience_left -= 1

            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} "
                  f"{metric_key}={cur:.4f} "
                  f"(best={best_metric:.4f}@{best_epoch}) "
                  f"patience={patience_left} {row['seconds']:.1f}s")

            if patience_left <= 0:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no {metric_key} improvement for "
                      f"{config.EARLY_STOP_PATIENCE} evals).")
                history.append(row)
                break
        else:
            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} "
                  f"{row['seconds']:.1f}s")

        history.append(row)

    # if we never validated, checkpoint the final model
    if not can_validate:
        save_checkpoint(model, config.LIGHTGCN_BEST_PATH,
                        {"epoch": config.EPOCHS, "best_metric": None,
                         "metric_key": None, "config": config.as_dict()})
        best_epoch = config.EPOCHS

    # ---- history CSV ------------------------------------------------------
    _write_history(history)
    print(f"[train] wrote {config.LIGHTGCN_HISTORY_PATH}")
    print(f"[train] best checkpoint: {config.LIGHTGCN_BEST_PATH} "
          f"(epoch {best_epoch}, {metric_key}="
          f"{best_metric:.4f})" if can_validate else
          f"[train] final checkpoint saved (epoch {best_epoch}).")
    return history


def _write_history(history):
    config.ensure_dirs()
    # union of all keys across rows (later epochs may add val_* columns)
    cols = []
    for row in history:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(config.LIGHTGCN_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


if __name__ == "__main__":
    train()
