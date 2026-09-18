"""
hybrid_evaluate.py
==================
Evaluate the trained hybrid recommender on the test split (full-ranking
Recall/NDCG/Precision@K, reusing the Phase 2 metric code). The graph and the
TRAIN-only review reps are rebuilt from data; only the learned parameters come
from the checkpoint — so evaluation representations never include test reviews.

Run:
    python hybrid_evaluate.py
"""

from __future__ import annotations

import os

import torch

import config
from lightgcn import build_norm_adjacency, get_device
from lightgcn_evaluate import (
    load_splits, build_pos_dict, infer_sizes, evaluate_ranking, merge_exclusions,
)
from hybrid_model import HybridRecommender
from hybrid_train import build_text_reprs


def load_hybrid(device):
    if not os.path.exists(config.HYBRID_BEST_PATH):
        raise FileNotFoundError(
            f"No hybrid checkpoint at {config.HYBRID_BEST_PATH}. Train first "
            f"(python hybrid_train.py).")
    train_df, val_df, test_df = load_splits()
    M, N = infer_sizes(train_df, val_df, test_df)

    z_user, z_item, _ = build_text_reprs(M, N, device)
    norm_adj, norm_R, _, _ = build_norm_adjacency(
        train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy(),
        M, N, device)

    ckpt = torch.load(config.HYBRID_BEST_PATH, map_location=device, weights_only=False)
    model = HybridRecommender(
        M, N, norm_adj, z_user, z_item,
        lgcn_dim=ckpt["lgcn_dim"], common_dim=ckpt["common_dim"],
        num_layers=ckpt["num_layers"], fusion=ckpt["fusion"],
        alpha=config.ALPHA, reg_lambda=ckpt.get("reg_lambda", 1e-4),
        proj_bias=config.FUSION_PROJECTION_BIAS, norm_R=norm_R,
    ).to(device)
    # graph buffers + z rebuilt above; load the rest of the parameters
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model, (train_df, val_df, test_df), (M, N)


def main():
    device = get_device()
    print(f"[hybrid-eval] device={device}")
    model, (train_df, val_df, test_df), (M, N) = load_hybrid(device)

    train_pos = build_pos_dict(train_df, N)
    val_pos = build_pos_dict(val_df, N)
    test_pos = build_pos_dict(test_df, N)
    if not test_pos:
        print("[hybrid-eval] empty test split — nothing to evaluate.")
        return

    exclude = merge_exclusions(train_pos, val_pos)
    metrics = evaluate_ranking(model, test_pos, exclude, N, config.HYBRID_TOPK,
                               device, config.EVAL_USER_BATCH)
    a = model.current_alpha()
    print(f"[hybrid-eval] fusion={model.fusion}"
          + (f" alpha={a:.3f}" if a is not None else "") + "  TEST:")
    for k in config.HYBRID_TOPK:
        print(f"   Recall@{k}={metrics[f'recall@{k}']:.4f}  "
              f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
              f"Precision@{k}={metrics[f'precision@{k}']:.4f}")
    return metrics


if __name__ == "__main__":
    main()
