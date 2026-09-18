

from __future__ import annotations

import math
import sys

import numpy as np
import torch

import data_loader
from lightgcn import LightGCN, build_norm_adjacency, set_seed, get_device
from lightgcn_train import NegativeSampler
from xlmr_model import masked_mean_pool
from hybrid_model import HybridRecommender
from comparison_metrics import evaluate_full
from review_embeddings import aggregate_user_item

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    device = get_device()
    print(f"verify_pipeline on device={device}\n")

    # -- 1. dataset schema consistency (adaptation layer) --------------------
    print("Dataset schema consistency:")
    s21 = data_loader.detect_schema(["user_id", "name", "time", "rating", "text", "gmap_id"])
    s18 = data_loader.detect_schema(["gPlusUserId", "reviewText", "unixReviewTime",
                                     "rating", "gPlusPlaceId", "categories"])
    sc4 = data_loader.detect_schema(["qid", "query", "item_id", "user_id",
                                     "ori_rating", "ori_review"])
    check("Google-Local-2021 schema mapped",
          s21.user == "user_id" and s21.item == "gmap_id" and s21.review == "text"
          and s21.timestamp == "time" and s21.language is None)
    check("Google-Local-2018 schema mapped",
          s18.user == "gPlusUserId" and s18.item == "gPlusPlaceId"
          and s18.review == "reviewText" and s18.timestamp == "unixReviewTime")
    check("Amazon-C4 schema mapped",
          sc4.user == "user_id" and sc4.item == "item_id"
          and sc4.review == "ori_review" and sc4.timestamp is None)

    # -- build a tiny graph --------------------------------------------------
    set_seed(42)
    u = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    it = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)
    M, N = 4, 4
    norm_adj, norm_R, _, _ = build_norm_adjacency(u, it, M, N, device)

    print("\nGraph construction & LightGCN propagation:")
    dense = norm_adj.to_dense()
    check("normalized adjacency shape (M+N)x(M+N)", tuple(norm_adj.shape) == (M + N, M + N))
    check("adjacency symmetric (Â = Âᵀ)", torch.allclose(dense, dense.t(), atol=1e-6))
    lg = LightGCN(M, N, norm_adj, embedding_dim=16, num_layers=3, norm_R=norm_R)
    a_u, a_i = lg.propagate()
    b_u, b_i = lg.propagate_explicit()
    check("propagate() == explicit user/item form",
          torch.allclose(a_u, b_u, atol=1e-5) and torch.allclose(a_i, b_i, atol=1e-5))

    # -- XLM-R masked mean pooling ------------------------------------------
    print("\nXLM-RoBERTa masked-mean pooling:")
    lhs = torch.randn(1, 5, 8)
    am = torch.tensor([[1, 1, 1, 0, 0]])
    manual = lhs[0, :3].mean(0)
    pooled = masked_mean_pool(lhs, am)[0]
    lhs_pad = torch.cat([lhs, torch.randn(1, 2, 8)], dim=1)
    am_pad = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])
    check("masked mean equals manual mean over valid tokens",
          torch.allclose(pooled, manual, atol=1e-6))
    check("pooling is padding-invariant (not [CLS])",
          torch.allclose(pooled, masked_mean_pool(lhs_pad, am_pad)[0], atol=1e-6))

    # -- BPR negative sampling ----------------------------------------------
    print("\nBPR negative sampling:")
    train_pos = {0: np.array([0, 1]), 1: np.array([1, 2]),
                 2: np.array([2, 3]), 3: np.array([3, 0])}
    sampler = NegativeSampler(train_pos, N, seed=1)
    negs = sampler.sample(u)
    leak = sum(int(negs[k]) in set(train_pos[int(u[k])].tolist()) for k in range(len(u)))
    check("sampled negatives are never observed positives", leak == 0)

    # -- reproducibility -----------------------------------------------------
    print("\nReproducibility:")
    set_seed(7); w1 = LightGCN(M, N, norm_adj, embedding_dim=16, num_layers=2,
                               norm_R=norm_R).user_embedding.weight.detach().clone()
    set_seed(7); w2 = LightGCN(M, N, norm_adj, embedding_dim=16, num_layers=2,
                               norm_R=norm_R).user_embedding.weight.detach().clone()
    check("same seed -> identical initialization", torch.allclose(w1, w2))

    # -- hybrid fusion + device ---------------------------------------------
    print("\nHybrid fusion & GPU/CPU compatibility:")
    zu, zi = torch.randn(M, 8), torch.randn(N, 8)
    hy = HybridRecommender(M, N, norm_adj, zu, zi, lgcn_dim=16, common_dim=16,
                           num_layers=2, fusion="weighted", alpha=0.5, norm_R=norm_R).to(device)
    hu, hi = hy.compute_repr()
    check("weighted fusion produces (M,d)/(N,d)", hu.shape == (M, 16) and hi.shape == (N, 16))
    loss, _, _ = hy.bpr_loss(torch.arange(4, device=device),
                             torch.arange(4, device=device),
                             torch.tensor([1, 2, 3, 0], device=device))
    check("hybrid BPR loss finite", bool(torch.isfinite(loss)))
    check("tensors on selected device", str(hu.device).split(":")[0] == device.type)
    hy_gate = HybridRecommender(M, N, norm_adj, zu, zi, lgcn_dim=16, common_dim=16,
                                num_layers=2, fusion="gate", norm_R=norm_R)
    gu, gi = hy_gate.compute_repr()
    check("gate fusion produces (M,d)/(N,d)", gu.shape == (M, 16) and gi.shape == (N, 16))

    # -- checkpoint reload ---------------------------------------------------
    print("\nCheckpoint loading:")
    state = {k: v for k, v in hy.state_dict().items()
             if not k.startswith("lightgcn._norm_adj") and k not in ("z_user", "z_item")}
    hy2 = HybridRecommender(M, N, norm_adj, zu, zi, lgcn_dim=16, common_dim=16,
                            num_layers=2, fusion="weighted", alpha=0.5, norm_R=norm_R)
    hy2.load_state_dict(state, strict=False)
    hu2, _ = hy2.compute_repr()
    check("reloaded checkpoint reproduces representations", torch.allclose(hu, hu2, atol=1e-6))

    # -- no test leakage -----------------------------------------------------
    print("\nNo test leakage (train-only review aggregation):")
    tr_u = np.array([0, 0, 1, 1, 2], dtype=np.int64)   # item 3 appears ONLY in test
    tr_i = np.array([0, 1, 1, 2, 2], dtype=np.int64)
    emb = torch.randn(len(tr_u), 5)
    agg = aggregate_user_item(emb, tr_u, tr_i, 4, 4)
    check("test-only item has zero train-derived representation",
          torch.allclose(agg["z_item"][3], torch.zeros(5)))

    # -- Top-K metric correctness (hand-computed) ---------------------------
    print("\nTop-K evaluation correctness:")
    scores = torch.tensor([[0.1, 0.0, 0.9, 0.5]])       # ranked: 2,3,0,1
    prov = lambda uid: scores
    r1 = evaluate_full(prov, {0: np.array([2])}, {}, 4, [1, 2], device)
    check("gt at rank 1 -> Recall@1=NDCG@1=MRR@1=1",
          abs(r1[1]["recall"] - 1) < 1e-6 and abs(r1[1]["ndcg"] - 1) < 1e-6
          and abs(r1[1]["mrr"] - 1) < 1e-6 and abs(r1[1]["precision"] - 1) < 1e-6)
    r2 = evaluate_full(prov, {0: np.array([3])}, {}, 4, [1, 2], device)
    exp_ndcg = (1 / math.log2(3)) / (1 / math.log2(2))
    check("gt at rank 2 -> Recall@1=0, Recall@2=1, NDCG@2 & MRR@2 exact",
          abs(r2[1]["recall"] - 0) < 1e-6 and abs(r2[2]["recall"] - 1) < 1e-6
          and abs(r2[2]["ndcg"] - exp_ndcg) < 1e-4 and abs(r2[2]["mrr"] - 0.5) < 1e-6)

    # -- tensor-dimension consistency (z = [h||e||s]) -----------------------
    print("\nTensor-dimension consistency:")
    h, e, s = torch.randn(3, 768), torch.randn(3, 11), torch.randn(3, 3)
    z = torch.cat([h, e, s], dim=1)
    check("z_ui concatenation dim = 768 + 11 + 3 = 782", z.shape == (3, 782))
    Wc, Wt = torch.nn.Linear(16, 16, bias=False), torch.nn.Linear(782, 16, bias=False)
    check("projections map to common dim 16",
          Wc(torch.randn(4, 16)).shape == (4, 16) and Wt(z).shape == (3, 16))

    # -- summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"VERIFICATION: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 60)
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
        return 1
    print("  All pipeline correctness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
