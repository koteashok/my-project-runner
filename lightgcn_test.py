"""
lightgcn_test.py
================
Small self-contained test of the LightGCN component. Uses a synthetic
interaction graph with community structure (so BPR has a learnable signal) and
confirms, with REAL numbers (nothing fabricated):

  1. graph dimensions are correct
  2. embedding dimensions are correct
  3. forward propagation works (and the matrix form == the explicit user/item form)
  4. training proceeds and the loss decreases
  5. recommendation scores / top-K can be generated

Run:
    python lightgcn_test.py
"""

from __future__ import annotations

import numpy as np
import torch

import config
from lightgcn import LightGCN, build_norm_adjacency, set_seed, get_device
from lightgcn_train import NegativeSampler
from lightgcn_evaluate import build_pos_dict, evaluate_ranking, recommend_topk


def make_synonym_graph(num_users=60, num_items=40, groups=4, seed=42):
    """Each user belongs to a group and interacts with items of that group,
    yielding repeat structure and a signal LightGCN can learn."""
    rng = np.random.RandomState(seed)
    u_list, i_list = [], []
    items_per_group = num_items // groups
    for u in range(num_users):
        g = u % groups
        lo, hi = g * items_per_group, (g + 1) * items_per_group
        k = rng.randint(4, items_per_group)
        picks = rng.choice(np.arange(lo, hi), size=k, replace=False)
        for it in picks:
            u_list.append(u)
            i_list.append(int(it))
    return (np.asarray(u_list, dtype=np.int64),
            np.asarray(i_list, dtype=np.int64),
            num_users, num_items)


def main():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"device = {device}")

    u, i, M, N = make_synonym_graph()
    E = len(u)
    d, K = config.EMBEDDING_DIM, config.NUM_LAYERS

    # ---------------------------------------------------------------- 1. graph
    norm_adj, norm_R, deg_u, deg_i = build_norm_adjacency(u, i, M, N, device)
    assert tuple(norm_adj.shape) == (M + N, M + N), "adjacency shape wrong"
    assert tuple(norm_R.shape) == (M, N), "norm_R shape wrong"
    assert norm_adj._nnz() == 2 * norm_R._nnz(), "A should hold R and R^T"
    # symmetry check: Â should equal Âᵀ
    dense = norm_adj.to_dense()
    assert torch.allclose(dense, dense.t(), atol=1e-6), "Â not symmetric"
    print(f"[1] graph OK: A={tuple(norm_adj.shape)} nnz={norm_adj._nnz()} "
          f"(edges E={E}, 2E={2*E}); Â symmetric; "
          f"deg_user sum={int(deg_u.sum())} deg_item sum={int(deg_i.sum())}")

    # ------------------------------------------------------------ 2. embeddings
    model = LightGCN(M, N, norm_adj, embedding_dim=d, num_layers=K,
                     reg_lambda=config.BPR_REG_LAMBDA, norm_R=norm_R).to(device)
    assert tuple(model.user_embedding.weight.shape) == (M, d)
    assert tuple(model.item_embedding.weight.shape) == (N, d)
    assert tuple(model.ego_embeddings().shape) == (M + N, d)
    print(f"[2] embeddings OK: E^(0)=({M+N},{d}) "
          f"users=({M},{d}) items=({N},{d})")

    # -------------------------------------------------------- 3. forward + equiv
    uemb, iemb = model.propagate()
    assert tuple(uemb.shape) == (M, d) and tuple(iemb.shape) == (N, d)
    assert torch.isfinite(uemb).all() and torch.isfinite(iemb).all()
    uemb2, iemb2 = model.propagate_explicit()
    same = (torch.allclose(uemb, uemb2, atol=1e-5) and
            torch.allclose(iemb, iemb2, atol=1e-5))
    max_diff = max((uemb - uemb2).abs().max().item(),
                   (iemb - iemb2).abs().max().item())
    assert same, f"matrix vs explicit propagation differ (max {max_diff:.2e})"
    print(f"[3] forward OK: propagate()==propagate_explicit() "
          f"(max abs diff {max_diff:.2e}); outputs finite")

    # ------------------------------------------------------- neg-sampler safety
    train_pos = build_pos_dict(
        _as_df(u, i), N)  # user->positives
    sampler = NegativeSampler(train_pos, N, seed=1)
    negs = sampler.sample(u)
    leak = sum(int(negs[k]) in set(train_pos[int(u[k])].tolist()) for k in range(E))
    assert leak == 0, f"negative sampler leaked {leak} positives"
    print(f"[neg] sampler OK: 0/{E} sampled negatives were observed positives")

    # ---------------------------------------------------------- 4. loss decreases
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    rng = np.random.RandomState(0)
    losses = []
    n_epochs = 30
    for ep in range(n_epochs):
        order = rng.permutation(E)
        uu, pp = u[order], i[order]
        ep_loss, nb = 0.0, 0
        for s in range(0, E, config.BATCH_SIZE):
            bu, bp = uu[s:s + config.BATCH_SIZE], pp[s:s + config.BATCH_SIZE]
            bn = sampler.sample(bu)
            tu = torch.from_numpy(bu).to(device)
            tp = torch.from_numpy(bp).to(device)
            tn = torch.from_numpy(bn).to(device)
            optimizer.zero_grad()
            loss, _, _ = model.bpr_loss(tu, tp, tn)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / nb)
    first, last = losses[0], losses[-1]
    assert np.isfinite(losses).all(), "non-finite loss encountered"
    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
    print(f"[4] training OK: BPR loss {first:.4f} -> {last:.4f} "
          f"over {n_epochs} epochs (decrease {100*(first-last)/first:.1f}%)")

    # ----------------------------------------------------- 5. recommendation
    some_users = np.arange(min(5, M), dtype=np.int64)
    all_u_emb, all_i_emb = model.get_all_embeddings()
    scores = model.score_users(torch.from_numpy(some_users).to(device),
                               all_i_emb, all_u_emb)
    assert tuple(scores.shape) == (len(some_users), N)
    assert torch.isfinite(scores).all()
    topk = recommend_topk(model, some_users, k=min(10, N), exclude=train_pos,
                          device=device)
    assert topk.shape == (len(some_users), min(10, N))
    assert topk.min() >= 0 and topk.max() < N
    # sanity: a held-out ranking metric runs end-to-end (not a performance claim)
    eval_pos = {int(uu): np.array([int(train_pos[int(uu)][0])]) for uu in some_users}
    m = evaluate_ranking(model, eval_pos, {}, N, [10], device)
    print(f"[5] recommendation OK: scores={tuple(scores.shape)} finite; "
          f"top-10 indices valid; sample Recall@10={m['recall@10']:.3f}")

    print("\nALL LIGHTGCN SELF-TESTS PASSED.")


def _as_df(u, i):
    import pandas as pd
    return pd.DataFrame({"user_idx": u, "item_idx": i})


if __name__ == "__main__":
    main()
