"""
hybrid_test.py
==============
Self-contained test of the hybrid recommender (Phase 5). Runs fully offline with
real PyTorch (the review side enters as precomputed feature vectors, so no
HuggingFace download is needed).

Confirms:
  - forward pass: both branches + fusion produce compatible dims (all fusion modes)
  - required dimension report (LightGCN / XLM-R / projected / fused / #params)
  - BPR loss decreases while training
  - recommendation scores can be generated
  - full train -> checkpoint -> evaluate integration via the real drivers,
    with TRAIN-only review reps (leakage-safe)
"""

from __future__ import annotations

import os
import numpy as np
import torch

import config
from lightgcn import build_norm_adjacency, set_seed, get_device
from hybrid_model import HybridRecommender


def community_graph(M=80, N=50, groups=5, seed=42):
    rng = np.random.RandomState(seed)
    ipg = N // groups
    u, i = [], []
    for uu in range(M):
        g = uu % groups
        lo, hi = g * ipg, (g + 1) * ipg
        for it in rng.choice(np.arange(lo, hi), size=rng.randint(4, ipg), replace=False):
            u.append(uu); i.append(int(it))
    return np.array(u, np.int64), np.array(i, np.int64), M, N


def main():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"device={device}")
    u, i, M, N = community_graph()
    norm_adj, norm_R, _, _ = build_norm_adjacency(u, i, M, N, device)
    text_dim = 782                                   # e.g. 768 + 11 + 3
    z_user = torch.randn(M, text_dim)
    z_item = torch.randn(N, text_dim)

    # ---------------------------------------- 1. all fusion modes forward pass
    for fusion in ("weighted", "weighted_learnable", "gate", "concat"):
        m = HybridRecommender(M, N, norm_adj, z_user, z_item,
                              lgcn_dim=config.EMBEDDING_DIM,
                              common_dim=config.FUSION_COMMON_DIM,
                              num_layers=config.NUM_LAYERS, fusion=fusion,
                              alpha=config.ALPHA, norm_R=norm_R).to(device)
        h_u, h_i = m.compute_repr()
        assert h_u.shape == (M, config.FUSION_COMMON_DIM)
        assert h_i.shape == (N, config.FUSION_COMMON_DIM)
        loss, _, _ = m.bpr_loss(torch.arange(4, device=device),
                                torch.arange(4, device=device),
                                torch.arange(4, 8, device=device))
        assert torch.isfinite(loss)
        d = m.dims()
        print(f"[1] fusion={fusion:<18} h_u{tuple(h_u.shape)} h_i{tuple(h_i.shape)} "
              f"params={d['trainable_params']:,} "
              f"alpha={d['alpha'] if d['alpha'] is not None else '-'}")

    # ---------------------------------------- 2. required dimension report
    model = HybridRecommender(M, N, norm_adj, z_user, z_item,
                              lgcn_dim=config.EMBEDDING_DIM,
                              common_dim=config.FUSION_COMMON_DIM,
                              num_layers=config.NUM_LAYERS,
                              fusion=config.FUSION_STRATEGY,
                              alpha=config.ALPHA, norm_R=norm_R).to(device)
    d = model.dims()
    print("\n[2] dimension report (default fusion):")
    print(f"    LightGCN representation dimension : {d['lightgcn_dim']}")
    print(f"    XLM-R representation dimension    : {d['xlmr_text_dim']}")
    print(f"    Projected dimension               : {d['projected_dim']}")
    print(f"    Fused representation dimension     : {d['fused_dim']}")
    print(f"    Number of trainable parameters    : {d['trainable_params']:,}")

    # ---------------------------------------- 3. forward-pass test (pre-train)
    B = 4
    uu = torch.randint(0, M, (B,), device=device)
    pp = torch.randint(0, N, (B,), device=device)
    nn_ = torch.randint(0, N, (B,), device=device)
    with torch.no_grad():
        hu, hi = model.compute_repr()
        sc = model.score_users(uu[:2], hu, hi)
    assert sc.shape == (2, N)
    print(f"[3] forward-pass test OK: scores{tuple(sc.shape)} (compatible dims)")

    # ---------------------------------------- 4. BPR loss decreases
    opt = torch.optim.Adam(model.parameters(), lr=config.HYBRID_LR,
                           weight_decay=config.HYBRID_WEIGHT_DECAY)
    rng = np.random.RandomState(0)
    E = len(u)
    losses = []
    for ep in range(40):
        order = rng.permutation(E)
        us, ps = u[order], i[order]
        tot, nb = 0.0, 0
        for s in range(0, E, config.HYBRID_BATCH_SIZE):
            bu, bp = us[s:s + config.HYBRID_BATCH_SIZE], ps[s:s + config.HYBRID_BATCH_SIZE]
            bn = rng.randint(0, N, size=len(bu))
            loss, _, _ = model.bpr_loss(torch.from_numpy(bu).to(device),
                                        torch.from_numpy(bp).to(device),
                                        torch.from_numpy(bn.astype(np.int64)).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        losses.append(tot / nb)
    assert np.isfinite(losses).all() and losses[-1] < losses[0], losses[:1] + losses[-1:]
    print(f"[4] training OK: BPR loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({100*(losses[0]-losses[-1])/losses[0]:.1f}% down over 40 epochs)")

    # ---------------------------------------- 5. recommendation scores
    with torch.no_grad():
        hu, hi = model.get_all_embeddings()
        scores = model.score_users(torch.arange(5, device=device), hu, hi)
    topk = torch.topk(scores, k=10, dim=1).indices
    assert topk.shape == (5, 10) and topk.min() >= 0 and topk.max() < N
    print(f"[5] recommendation OK: scores{tuple(scores.shape)}; top-10 valid")

    # ---------------------------------------- 6. full train/eval integration
    _integration(device)
    print("\nALL HYBRID SELF-TESTS PASSED.")


def _integration(device):
    """Exercise the real hybrid_train / hybrid_evaluate drivers on synthetic
    Phase 1 splits + Phase 3/4 caches (random features)."""
    import json, pandas as pd
    tmp = "results_t5"
    for k, v in {"RESULTS_DIR": tmp, "PROCESSED_DIR": f"{tmp}/processed",
                 "MAPPINGS_DIR": f"{tmp}/mappings", "CACHE_DIR": f"{tmp}/cache",
                 "CHECKPOINT_DIR": f"{tmp}/checkpoints",
                 "REVIEW_EMB_CACHE": f"{tmp}/cache/review_embeddings.pt",
                 "EMOTION_FEATURES_CACHE": f"{tmp}/cache/emotion_features.pt",
                 "XLMR_SENTIMENT_BEST": f"{tmp}/checkpoints/xlmr_sentiment_best.pt",
                 "LIGHTGCN_BEST_PATH": f"{tmp}/checkpoints/lightgcn_best.pt",
                 "HYBRID_BEST_PATH": f"{tmp}/checkpoints/hybrid_best.pt",
                 "HYBRID_HISTORY_PATH": f"{tmp}/hybrid_training_history.csv",
                 "HYBRID_EPOCHS": 6}.items():
        setattr(config, k, v)
    for dsub in ("processed", "mappings", "cache", "checkpoints"):
        os.makedirs(f"{tmp}/{dsub}", exist_ok=True)

    rng = np.random.RandomState(1)
    M, N, G = 80, 50, 5
    ipg = N // G
    rows = []
    for uu in range(M):
        g = uu % G
        for _ in range(rng.randint(4, 9)):
            it = int(rng.choice(np.arange(g * ipg, (g + 1) * ipg)))
            rows.append((uu, it, int(rng.randint(1, 6)), "review text",
                         int(rng.randint(1_500_000_000, 1_700_000_000))))
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "review", "timestamp"])
    tr, va, te = [], [], []
    for uu, g in df.groupby("user_idx"):
        g = g.sort_values("timestamp")
        if len(g) >= 3: tr.append(g.iloc[:-2]); va.append(g.iloc[-2:-1]); te.append(g.iloc[-1:])
        elif len(g) == 2: tr.append(g.iloc[:1]); te.append(g.iloc[1:2])
        else: tr.append(g)
    splits = {"train": pd.concat(tr), "validation": pd.concat(va), "test": pd.concat(te)}
    for name, d in splits.items():
        d = d.copy(); d["user"] = d["user_idx"]; d["item"] = d["item_idx"]
        d.to_parquet(f"{config.PROCESSED_DIR}/{name}.parquet", index=False)
    json.dump({str(x): x for x in range(M)}, open(f"{config.MAPPINGS_DIR}/user2idx.json", "w", encoding="utf-8"))
    json.dump({str(x): x for x in range(N)}, open(f"{config.MAPPINGS_DIR}/item2idx.json", "w", encoding="utf-8"))

    # fake Phase 3 h-cache + Phase 4 emotion cache (aligned), + sentiment head
    hcache = {"hidden": 768, "splits": {}}
    ecache = {"labels": [f"emo{c}" for c in range(11)], "num_labels": 11, "splits": {}}
    for name, d in splits.items():
        ui = d["user_idx"].to_numpy().astype(np.int64)
        ii = d["item_idx"].to_numpy().astype(np.int64)
        n = len(d)
        hcache["splits"][name] = {"embeddings": torch.randn(n, 768).half(),
                                  "user_idx": ui, "item_idx": ii,
                                  "rating": d["rating"].to_numpy().astype(np.float32),
                                  "has_text": np.ones(n, bool)}
        probs = torch.softmax(torch.randn(n, 11), dim=1)
        ecache["splits"][name] = {"probs": probs, "user_idx": ui, "item_idx": ii,
                                  "rating": d["rating"].to_numpy().astype(np.float32),
                                  "language": None, "has_text": np.ones(n, bool)}
    torch.save(hcache, config.REVIEW_EMB_CACHE)
    torch.save(ecache, config.EMOTION_FEATURES_CACHE)
    from xlmr_sentiment import SentimentHead
    torch.save({"state_dict": SentimentHead(768, 3).state_dict(), "hidden": 768,
                "num_classes": 3, "labels": config.SENTIMENT_LABELS,
                "rating_map": {}, "meta": {}}, config.XLMR_SENTIMENT_BEST)

    import importlib, hybrid_train, hybrid_evaluate
    importlib.reload(hybrid_train); importlib.reload(hybrid_evaluate)
    print("\n[6] full integration: hybrid_train.train() ...")
    hybrid_train.train()
    assert os.path.exists(config.HYBRID_BEST_PATH) and os.path.exists(config.HYBRID_HISTORY_PATH)
    print("[6] hybrid_evaluate.main() ...")
    hybrid_evaluate.main()
    print("[6] integration OK: checkpoint + history written; evaluation ran")


if __name__ == "__main__":
    main()
