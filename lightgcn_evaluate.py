"""
lightgcn_evaluate.py
====================
Full-ranking evaluation for LightGCN: Recall@K, NDCG@K, Precision@K with
seen-item masking, plus top-K recommendation generation.

Ranking protocol
----------------
For each evaluation user, score every item, mask out items the user has already
interacted with (its training items, and — when evaluating the test set — its
validation items too), rank the rest, and compare the top-K against the held-out
ground-truth items for that user. Metrics are averaged over evaluation users.

Run standalone to evaluate the best checkpoint on the test split:
    python lightgcn_evaluate.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import config
from lightgcn import LightGCN, build_norm_adjacency, get_device


# ----------------------------------------------------------------------------- #
# Data helpers                                                                   #
# ----------------------------------------------------------------------------- #
def _read_split(basename: str):
    """Load a Phase 1 processed split (parquet preferred, csv fallback)."""
    import pandas as pd
    p_parq = os.path.join(config.PROCESSED_DIR, basename + ".parquet")
    p_csv = os.path.join(config.PROCESSED_DIR, basename + ".csv")
    if os.path.exists(p_parq):
        return pd.read_parquet(p_parq)
    if os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    return None


def load_splits():
    """Return (train_df, val_df, test_df) with user_idx/item_idx columns."""
    train = _read_split("train")
    val = _read_split("validation")
    test = _read_split("test")
    if train is None:
        raise FileNotFoundError(
            f"No train split under {config.PROCESSED_DIR}. Run Phase 1 first "
            f"(python phase1_dataset.py).")
    return train, val, test


def build_pos_dict(df, num_items: int) -> Dict[int, np.ndarray]:
    """user_idx -> array of item_idx (positives) for the given split."""
    d: Dict[int, List[int]] = {}
    if df is None or len(df) == 0:
        return {}
    u = df["user_idx"].to_numpy()
    it = df["item_idx"].to_numpy()
    for a, b in zip(u, it):
        d.setdefault(int(a), []).append(int(b))
    return {k: np.asarray(v, dtype=np.int64) for k, v in d.items()}


def infer_sizes(train, val, test, mappings_dir: Optional[str] = None):
    """Determine (num_users, num_items). Prefer the Phase 1 id mappings; fall
    back to max index + 1 across all splits."""
    mappings_dir = mappings_dir or config.MAPPINGS_DIR
    u_map = os.path.join(mappings_dir, "user2idx.json")
    i_map = os.path.join(mappings_dir, "item2idx.json")
    if os.path.exists(u_map) and os.path.exists(i_map):
        with open(u_map) as f:
            M = len(json.load(f))
        with open(i_map) as f:
            N = len(json.load(f))
        return M, N
    max_u = max_i = -1
    for df in (train, val, test):
        if df is not None and len(df):
            max_u = max(max_u, int(df["user_idx"].max()))
            max_i = max(max_i, int(df["item_idx"].max()))
    return max_u + 1, max_i + 1


# ----------------------------------------------------------------------------- #
# Metrics                                                                        #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_ranking(model: LightGCN,
                     eval_pos: Dict[int, np.ndarray],
                     exclude: Dict[int, np.ndarray],
                     num_items: int,
                     ks: List[int],
                     device: torch.device,
                     user_batch: int = 1024) -> Dict[str, float]:
    """
    Compute mean Recall@K, NDCG@K, Precision@K over the users in `eval_pos`.
    `exclude[u]` lists items to mask (already-seen) for user u.
    """
    model.eval()
    users = np.asarray(sorted(eval_pos.keys()), dtype=np.int64)
    if len(users) == 0:
        return {f"recall@{k}": 0.0 for k in ks}

    all_u, all_i = model.get_all_embeddings()
    all_i = all_i.to(device)
    kmax = max(ks)

    # discount table for NDCG: 1/log2(rank+2), rank from 0
    discounts = 1.0 / torch.log2(torch.arange(2, kmax + 2, device=device).float())

    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"ndcg@{k}": 0.0 for k in ks})
    totals.update({f"precision@{k}": 0.0 for k in ks})
    n_users = 0

    for start in range(0, len(users), user_batch):
        batch = users[start:start + user_batch]
        bt = torch.from_numpy(batch).to(device)
        scores = torch.matmul(all_u[bt], all_i.t())      # (B, N)

        # mask seen items and ground-truth-relevance matrix
        gt_mask = torch.zeros_like(scores, dtype=torch.bool)
        for row, u in enumerate(batch):
            if u in exclude:
                ex = torch.from_numpy(exclude[u]).to(device)
                scores[row, ex] = float("-inf")
            gt = torch.from_numpy(eval_pos[u]).to(device)
            gt_mask[row, gt] = True

        topk = torch.topk(scores, k=kmax, dim=1).indices          # (B, kmax)
        rel = torch.gather(gt_mask.float(), 1, topk)              # (B, kmax)
        n_gt = gt_mask.sum(dim=1).clamp(min=1).float()           # |gt| per user

        for k in ks:
            rel_k = rel[:, :k]
            hits = rel_k.sum(dim=1)
            totals[f"recall@{k}"] += (hits / n_gt).sum().item()
            totals[f"precision@{k}"] += (hits / k).sum().item()
            dcg = (rel_k * discounts[:k]).sum(dim=1)
            # ideal DCG for min(|gt|, k) relevant items
            ideal_n = torch.minimum(n_gt, torch.full_like(n_gt, k)).long()
            idcg = torch.stack([discounts[:m].sum() for m in ideal_n])
            idcg = idcg.clamp(min=1e-12)
            totals[f"ndcg@{k}"] += (dcg / idcg).sum().item()

        n_users += len(batch)

    return {m: (v / n_users) for m, v in totals.items()}


@torch.no_grad()
def recommend_topk(model: LightGCN,
                   user_ids: np.ndarray,
                   k: int,
                   exclude: Optional[Dict[int, np.ndarray]],
                   device: torch.device) -> np.ndarray:
    """Return a (len(user_ids) × k) array of recommended item indices."""
    model.eval()
    all_u, all_i = model.get_all_embeddings()
    ut = torch.from_numpy(np.asarray(user_ids, dtype=np.int64)).to(device)
    scores = torch.matmul(all_u[ut], all_i.t())
    if exclude:
        for row, u in enumerate(user_ids):
            if u in exclude:
                ex = torch.from_numpy(exclude[u]).to(device)
                scores[row, ex] = float("-inf")
    topk = torch.topk(scores, k=k, dim=1).indices
    return topk.cpu().numpy()


def merge_exclusions(*dicts) -> Dict[int, np.ndarray]:
    """Union of several user->items dicts (e.g. train + val for test eval)."""
    out: Dict[int, list] = {}
    for d in dicts:
        if not d:
            continue
        for u, items in d.items():
            out.setdefault(u, []).extend(list(items))
    return {u: np.unique(np.asarray(v, dtype=np.int64)) for u, v in out.items()}


# ----------------------------------------------------------------------------- #
# CLI: evaluate best checkpoint on the test split                                #
# ----------------------------------------------------------------------------- #
def _rebuild_model_from_checkpoint(ckpt, norm_adj, norm_R, device):
    model = LightGCN(
        num_users=ckpt["num_users"], num_items=ckpt["num_items"],
        norm_adj=norm_adj, embedding_dim=ckpt["embedding_dim"],
        num_layers=ckpt["num_layers"], reg_lambda=ckpt.get("reg_lambda", 1e-4),
        norm_R=norm_R,
    ).to(device)
    # graph buffers are rebuilt from train; only embeddings are in the checkpoint
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


def main():
    device = get_device()
    print(f"[eval] device={device}")
    train, val, test = load_splits()
    M, N = infer_sizes(train, val, test)
    print(f"[eval] users={M:,} items={N:,}")

    norm_adj, norm_R, _, _ = build_norm_adjacency(
        train["user_idx"].to_numpy(), train["item_idx"].to_numpy(), M, N, device)

    if not os.path.exists(config.LIGHTGCN_BEST_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {config.LIGHTGCN_BEST_PATH}. Train first "
            f"(python lightgcn_train.py).")
    ckpt = torch.load(config.LIGHTGCN_BEST_PATH, map_location=device, weights_only=False)
    model = _rebuild_model_from_checkpoint(ckpt, norm_adj, norm_R, device)

    train_pos = build_pos_dict(train, N)
    val_pos = build_pos_dict(val, N)
    test_pos = build_pos_dict(test, N)

    if not test_pos:
        print("[eval] test split is empty — nothing to evaluate. "
              "(Amazon-C4-style single-interaction users produce no test set.)")
        return

    exclude = merge_exclusions(train_pos, val_pos)
    metrics = evaluate_ranking(model, test_pos, exclude, N, config.TOP_K,
                               device, config.EVAL_USER_BATCH)
    print("[eval] TEST metrics:")
    for k in config.TOP_K:
        print(f"   Recall@{k}={metrics[f'recall@{k}']:.4f}  "
              f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
              f"Precision@{k}={metrics[f'precision@{k}']:.4f}")


if __name__ == "__main__":
    main()
