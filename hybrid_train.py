"""
hybrid_train.py
===============
Train the hybrid recommender (LightGCN + XLM-R review reps) with BPR.

Leakage safety
--------------
The review user/item representations z_u / z_i are aggregated from TRAIN reviews
ONLY (``REVIEW_REPR_SOURCE = "train"``) and are used unchanged for both training
and evaluation, so test (and val) reviews never enter the representations used to
score held-out interactions.

Artefacts:
    results/checkpoints/hybrid_best.pt
    results/hybrid_training_history.csv

Run:
    python hybrid_train.py
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

import config
from lightgcn import build_norm_adjacency, set_seed, get_device
from lightgcn_train import NegativeSampler
from lightgcn_evaluate import (
    load_splits, build_pos_dict, infer_sizes, evaluate_ranking,
)
from hybrid_model import HybridRecommender


# ----------------------------------------------------------------------------- #
# Build TRAIN-only review representations z_u, z_i = agg([h || e || s])          #
# ----------------------------------------------------------------------------- #
def build_text_reprs(num_users: int, num_items: int,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Concatenate per-review [h || e || s] over TRAIN reviews and mean-aggregate to
    users/items. Returns (z_user (M,D), z_item (N,D), info). Falls back to
    h-only (or zeros) if the emotion / sentiment artefacts are absent.
    """
    from review_embeddings import load_cache as load_h_cache, aggregate_user_item
    info = {"components": [], "text_dim": 0}

    h_cache = load_h_cache()
    tr = h_cache["splits"].get("train")
    if tr is None:
        raise RuntimeError("No train split in review-embedding cache. Run Phase 3.")
    h = tr["embeddings"].float()
    user_idx = tr["user_idx"]
    item_idx = tr["item_idx"]
    parts = [h]
    info["components"].append(f"h({h.shape[1]})")

    # emotion probabilities e (Phase 4) — aligned rows
    try:
        from emotion_features import load_emotion_cache
        e_cache = load_emotion_cache()
        etr = e_cache["splits"].get("train")
        if etr is not None and np.array_equal(etr["user_idx"], user_idx) \
                and np.array_equal(etr["item_idx"], item_idx):
            e = etr["probs"].float()
            parts.append(e)
            info["components"].append(f"e({e.shape[1]})")
        else:
            print("  [warn] emotion cache missing/misaligned — omitting e from z.")
    except FileNotFoundError:
        print("  [warn] no emotion cache — omitting e from z (run Phase 4 to include it).")

    # sentiment representation s (Phase 3 head applied to h)
    try:
        from emotion_representation import _sentiment_rep
        s = _sentiment_rep(h)
        parts.append(s)
        info["components"].append(f"s({s.shape[1]})")
    except Exception as exc:
        print(f"  [warn] sentiment rep unavailable ({exc}); omitting s from z.")

    z_rev = torch.cat(parts, dim=1)                     # (n_train, D)
    info["text_dim"] = int(z_rev.shape[1])

    agg = aggregate_user_item(z_rev, user_idx, item_idx, num_users, num_items)
    info["users_covered"] = int((agg["user_count"] > 0).sum())
    info["items_covered"] = int((agg["item_count"] > 0).sum())
    return agg["z_user"].to(device), agg["z_item"].to(device), info


# ----------------------------------------------------------------------------- #
# Optional: initialise collaborative branch from the Phase 2 checkpoint          #
# ----------------------------------------------------------------------------- #
def maybe_init_lightgcn(model: HybridRecommender):
    if not config.HYBRID_INIT_LIGHTGCN_FROM_CKPT:
        return
    path = config.LIGHTGCN_BEST_PATH
    if not os.path.exists(path):
        print("  [init] no Phase 2 LightGCN checkpoint — training embeddings from scratch.")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", {})
    emb = {k: v for k, v in sd.items()
           if k.startswith("user_embedding") or k.startswith("item_embedding")}
    if emb and emb.get("user_embedding.weight", torch.empty(0)).shape[0] == model.num_users:
        model.lightgcn.load_state_dict(emb, strict=False)
        print(f"  [init] warm-started LightGCN embeddings from {path}")
    else:
        print("  [init] Phase 2 checkpoint shape mismatch — training from scratch.")


# ----------------------------------------------------------------------------- #
# Forward-pass sanity test (before training)                                     #
# ----------------------------------------------------------------------------- #
def forward_pass_test(model: HybridRecommender, device: torch.device):
    model.eval()
    B = 4
    users = torch.randint(0, model.num_users, (B,), device=device)
    pos = torch.randint(0, model.num_items, (B,), device=device)
    neg = torch.randint(0, model.num_items, (B,), device=device)
    with torch.no_grad():
        h_u, h_i = model.compute_repr()
        loss, bpr, reg = model.bpr_loss(users, pos, neg)
        scores = model.score_users(users[:2], h_u, h_i)
    d = model.common_dim
    assert h_u.shape == (model.num_users, d), h_u.shape
    assert h_i.shape == (model.num_items, d), h_i.shape
    assert scores.shape == (2, model.num_items), scores.shape
    assert torch.isfinite(loss), "non-finite loss in forward-pass test"
    print(f"  [forward-test] h_u{tuple(h_u.shape)} h_i{tuple(h_i.shape)} "
          f"scores{tuple(scores.shape)} loss={loss.item():.4f} — OK")


def print_report(model: HybridRecommender, text_info: dict):
    d = model.dims()
    print("\n" + "=" * 60)
    print("HYBRID MODEL DIMENSIONS")
    print("=" * 60)
    print(f"  LightGCN representation dimension : {d['lightgcn_dim']}")
    print(f"  XLM-R representation dimension    : {d['xlmr_text_dim']}  "
          f"(= {' + '.join(text_info['components'])})")
    print(f"  Projected dimension               : {d['projected_dim']}")
    print(f"  Fused representation dimension     : {d['fused_dim']}")
    print(f"  Fusion strategy                   : {d['fusion']}"
          + (f" (alpha={d['alpha']:.3f})" if d['alpha'] is not None else ""))
    print(f"  Number of trainable parameters    : {d['trainable_params']:,}")
    print("=" * 60)


# ----------------------------------------------------------------------------- #
# Training                                                                       #
# ----------------------------------------------------------------------------- #
def train_one_epoch(model, optimizer, users_arr, pos_arr, sampler, device,
                    batch_size, rng):
    model.train()
    order = rng.permutation(len(users_arr))
    users_arr, pos_arr = users_arr[order], pos_arr[order]
    tot, tb, tr, nb = 0.0, 0.0, 0.0, 0
    for s in range(0, len(users_arr), batch_size):
        bu = users_arr[s:s + batch_size]
        bp = pos_arr[s:s + batch_size]
        bn = sampler.sample(bu)
        u = torch.from_numpy(bu).to(device)
        p = torch.from_numpy(bp).to(device)
        ng = torch.from_numpy(bn).to(device)
        optimizer.zero_grad()
        loss, bpr, reg = model.bpr_loss(u, p, ng)
        loss.backward()
        optimizer.step()
        tot += loss.item(); tb += float(bpr); tr += float(reg); nb += 1
    return {"loss": tot / max(nb, 1), "bpr": tb / max(nb, 1), "reg": tr / max(nb, 1)}


def save_checkpoint(model, path, extra):
    config.ensure_checkpoint_dir()
    # store trainable + fusion params; rebuild graph & z from data at load time
    skip = ("z_user", "z_item")
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("lightgcn._norm_adj") and k not in skip}
    payload = {"model_state_dict": state,
               "num_users": model.num_users, "num_items": model.num_items,
               "lgcn_dim": model.lgcn_dim, "text_dim": model.text_dim,
               "common_dim": model.common_dim, "fusion": model.fusion,
               "num_layers": model.lightgcn.num_layers,
               "reg_lambda": model.reg_lambda}
    payload.update(extra)
    torch.save(payload, path)


def train():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"[hybrid] device={device} seed={config.RANDOM_SEED} "
          f"fusion={config.FUSION_STRATEGY}")

    train_df, val_df, test_df = load_splits()
    M, N = infer_sizes(train_df, val_df, test_df)
    users_arr = train_df["user_idx"].to_numpy().astype(np.int64)
    pos_arr = train_df["item_idx"].to_numpy().astype(np.int64)
    train_pos = build_pos_dict(train_df, N)
    val_pos = build_pos_dict(val_df, N)

    print(f"[hybrid] users={M:,} items={N:,} train_interactions={len(train_df):,}")
    print("[hybrid] building TRAIN-only review reps z_u/z_i (leakage-safe)...")
    z_user, z_item, text_info = build_text_reprs(M, N, device)
    print(f"  z_user{tuple(z_user.shape)} z_item{tuple(z_item.shape)} "
          f"(users covered {text_info['users_covered']}/{M}, "
          f"items covered {text_info['items_covered']}/{N})")

    norm_adj, norm_R, _, _ = build_norm_adjacency(users_arr, pos_arr, M, N, device)
    model = HybridRecommender(
        M, N, norm_adj, z_user, z_item,
        lgcn_dim=config.EMBEDDING_DIM, common_dim=config.FUSION_COMMON_DIM,
        num_layers=config.NUM_LAYERS, fusion=config.FUSION_STRATEGY,
        alpha=config.ALPHA, reg_lambda=config.HYBRID_L2,
        proj_bias=config.FUSION_PROJECTION_BIAS, norm_R=norm_R,
    ).to(device)
    maybe_init_lightgcn(model)

    print_report(model, text_info)
    forward_pass_test(model, device)          # BEFORE training (required)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.HYBRID_LR,
                                 weight_decay=config.HYBRID_WEIGHT_DECAY)
    sampler = NegativeSampler(train_pos, N, seed=config.RANDOM_SEED)
    rng = np.random.RandomState(config.RANDOM_SEED)
    can_val = len(val_pos) > 0
    metric_key = f"{config.HYBRID_EVAL_METRIC}@{config.HYBRID_EVAL_K}"

    history, best_metric, best_epoch = [], -1.0, -1
    patience = config.HYBRID_PATIENCE

    print("\n[hybrid] training...")
    for epoch in range(1, config.HYBRID_EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, optimizer, users_arr, pos_arr, sampler,
                             device, config.HYBRID_BATCH_SIZE, rng)
        row = {"epoch": epoch, "train_loss": round(tr["loss"], 6),
               "bpr": round(tr["bpr"], 6), "reg": round(tr["reg"], 6),
               "alpha": model.current_alpha(), "seconds": round(time.time() - t0, 3)}
        if can_val:
            metrics = evaluate_ranking(model, val_pos, train_pos, N,
                                       config.HYBRID_TOPK, device,
                                       config.EVAL_USER_BATCH)
            for k in config.HYBRID_TOPK:
                row[f"val_recall@{k}"] = round(metrics[f"recall@{k}"], 6)
                row[f"val_ndcg@{k}"] = round(metrics[f"ndcg@{k}"], 6)
            cur = metrics[metric_key]
            if cur > best_metric + 1e-6:
                best_metric, best_epoch = cur, epoch
                patience = config.HYBRID_PATIENCE
                save_checkpoint(model, config.HYBRID_BEST_PATH,
                                {"epoch": epoch, "best_metric": best_metric,
                                 "metric_key": metric_key, "config": config.as_dict()})
            else:
                patience -= 1
            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} "
                  f"{metric_key}={cur:.4f} (best={best_metric:.4f}@{best_epoch}) "
                  f"patience={patience} {row['seconds']:.1f}s")
            history.append(row)
            if patience <= 0:
                print(f"[hybrid] early stopping at epoch {epoch}")
                break
        else:
            print(f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} {row['seconds']:.1f}s")
            history.append(row)

    if not can_val:
        save_checkpoint(model, config.HYBRID_BEST_PATH,
                        {"epoch": config.HYBRID_EPOCHS, "best_metric": None,
                         "metric_key": None, "config": config.as_dict()})
        best_epoch = config.HYBRID_EPOCHS

    _write_history(history)
    print(f"[hybrid] wrote {config.HYBRID_HISTORY_PATH}")
    print(f"[hybrid] best checkpoint {config.HYBRID_BEST_PATH} "
          f"(epoch {best_epoch}" + (f", {metric_key}={best_metric:.4f})" if can_val else ")"))
    return history


def _write_history(history):
    config.ensure_dirs()
    cols = []
    for r in history:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(config.HYBRID_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in history:
            w.writerow(r)


if __name__ == "__main__":
    train()
