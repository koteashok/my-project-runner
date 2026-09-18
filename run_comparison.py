

from __future__ import annotations

import csv
import os
from typing import Dict

import numpy as np
import torch

import config
from lightgcn import LightGCN, build_norm_adjacency, set_seed, get_device
from lightgcn_train import NegativeSampler
from lightgcn_evaluate import load_splits, build_pos_dict, infer_sizes, merge_exclusions
from baselines import PopularityRecommender, BPRMF, TextRecommender
from hybrid_model import HybridRecommender
from comparison_metrics import (
    evaluate_full, embedding_score_provider, popularity_score_provider,
)


# ----------------------------------------------------------------------------- #
# Shared BPR trainer (used by MF / LightGCN / Text / Hybrid)                     #
# ----------------------------------------------------------------------------- #
def train_bpr(model, users_arr, pos_arr, train_pos, val_pos, N, device,
              tag: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.COMPARE_LR,
                                 weight_decay=config.COMPARE_WEIGHT_DECAY)
    sampler = NegativeSampler(train_pos, N, seed=config.RANDOM_SEED)
    rng = np.random.RandomState(config.RANDOM_SEED)
    can_val = len(val_pos) > 0
    metric_key = f"{config.COMPARE_EVAL_METRIC}@{config.COMPARE_EVAL_K}"
    best, best_state, patience = -1.0, None, config.COMPARE_PATIENCE

    for epoch in range(1, config.COMPARE_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(users_arr))
        ua, pa = users_arr[order], pos_arr[order]
        tot = nb = 0
        for s in range(0, len(ua), config.COMPARE_BATCH_SIZE):
            bu = ua[s:s + config.COMPARE_BATCH_SIZE]
            bp = pa[s:s + config.COMPARE_BATCH_SIZE]
            bn = sampler.sample(bu)
            loss, _, _ = model.bpr_loss(torch.from_numpy(bu).to(device),
                                        torch.from_numpy(bp).to(device),
                                        torch.from_numpy(bn).to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item(); nb += 1
        if can_val:
            uemb, iemb = model.get_all_embeddings()
            prov = embedding_score_provider(uemb, iemb)
            m = evaluate_full(prov, val_pos, train_pos, N, [config.COMPARE_EVAL_K],
                              device, config.EVAL_USER_BATCH)
            cur = m[config.COMPARE_EVAL_K][config.COMPARE_EVAL_METRIC]
            if cur > best + 1e-6:
                best = cur
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience = config.COMPARE_PATIENCE
            else:
                patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{tag}] trained ({metric_key} best={best:.4f})" if can_val
          else f"  [{tag}] trained ({config.COMPARE_EPOCHS} epochs, no val)")
    return model


# ----------------------------------------------------------------------------- #
# Build model score providers                                                    #
# ----------------------------------------------------------------------------- #
def provider_for(model, device):
    if isinstance(model, PopularityRecommender):
        return popularity_score_provider(model.popularity_tensor(device))
    uemb, iemb = model.get_all_embeddings()
    return embedding_score_provider(uemb, iemb)


# ----------------------------------------------------------------------------- #
# Orchestration                                                                  #
# ----------------------------------------------------------------------------- #
def run(splits=None, z_user=None, z_item=None):
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"[compare] device={device} seed={config.RANDOM_SEED} K={config.K_VALUES}")

    if splits is None:
        train_df, val_df, test_df = load_splits()
    else:
        train_df, val_df, test_df = splits
    M, N = infer_sizes(train_df, val_df, test_df)
    users_arr = train_df["user_idx"].to_numpy().astype(np.int64)
    pos_arr = train_df["item_idx"].to_numpy().astype(np.int64)
    train_pos = build_pos_dict(train_df, N)
    val_pos = build_pos_dict(val_df, N)
    test_pos = build_pos_dict(test_df, N)
    exclude = merge_exclusions(train_pos, val_pos)
    print(f"[compare] users={M:,} items={N:,} "
          f"train={len(train_df):,} val={len(val_df) if val_df is not None else 0:,} "
          f"test={len(test_df) if test_df is not None else 0:,}")
    if not test_pos:
        raise RuntimeError("Empty test split — cannot run comparison.")

    # graph + review reps (shared)
    norm_adj, norm_R, _, _ = build_norm_adjacency(users_arr, pos_arr, M, N, device)
    if z_user is None or z_item is None:
        from hybrid_train import build_text_reprs
        z_user, z_item, tinfo = build_text_reprs(M, N, device)
        print(f"[compare] review reps z: dim={z_user.shape[1]} "
              f"({' + '.join(tinfo['components'])}), TRAIN-only")
    else:
        z_user, z_item = z_user.to(device), z_item.to(device)

    # ---- instantiate models --------------------------------------------------
    pop = PopularityRecommender(M, N); pop.fit(pos_arr)
    mf = BPRMF(M, N, dim=config.COMPARE_DIM, reg_lambda=config.HYBRID_L2).to(device)
    lgcn = LightGCN(M, N, norm_adj, embedding_dim=config.EMBEDDING_DIM,
                    num_layers=config.NUM_LAYERS, reg_lambda=config.HYBRID_L2,
                    norm_R=norm_R).to(device)
    text = TextRecommender(z_user, z_item, common_dim=config.COMPARE_DIM,
                           reg_lambda=config.HYBRID_L2).to(device)
    hybrid = HybridRecommender(M, N, norm_adj, z_user, z_item,
                               lgcn_dim=config.EMBEDDING_DIM,
                               common_dim=config.FUSION_COMMON_DIM,
                               num_layers=config.NUM_LAYERS,
                               fusion=config.FUSION_STRATEGY, alpha=config.ALPHA,
                               reg_lambda=config.HYBRID_L2, norm_R=norm_R).to(device)

    # ---- train ---------------------------------------------------------------
    print("[compare] training models (shared BPR budget)...")
    train_bpr(mf, users_arr, pos_arr, train_pos, val_pos, N, device, "BPR-MF")
    train_bpr(lgcn, users_arr, pos_arr, train_pos, val_pos, N, device, "LightGCN")
    train_bpr(text, users_arr, pos_arr, train_pos, val_pos, N, device, "XLM-R-Text")
    train_bpr(hybrid, users_arr, pos_arr, train_pos, val_pos, N, device, "Hybrid")

    models = {
        "Popularity": pop,
        "BPR-MF": mf,
        "LightGCN": lgcn,
        "XLM-R-Text": text,
        "Hybrid": hybrid,
    }

    # ---- evaluate (identical protocol) --------------------------------------
    print("[compare] evaluating on TEST (train+val masked)...")
    results: Dict[str, Dict] = {}
    for name, model in models.items():
        prov = provider_for(model, device)
        results[name] = evaluate_full(prov, test_pos, exclude, N, config.K_VALUES,
                                      device, config.EVAL_USER_BATCH)

    _write_csv(results)
    _print_table(results)
    _print_improvements(results)
    return results


# ----------------------------------------------------------------------------- #
# Reporting                                                                      #
# ----------------------------------------------------------------------------- #
CSV_COLS = ["Model", "K", "Precision", "Recall", "F1", "HitRate", "NDCG",
            "MAP", "MRR", "Coverage", "RecCoverage", "AvgScore"]


def _write_csv(results):
    config.ensure_dirs()
    with open(config.MODEL_COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for name, per_k in results.items():
            for k in config.K_VALUES:
                r = per_k[k]
                w.writerow([name, k,
                            f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                            f"{r['f1']:.6f}", f"{r['hitrate']:.6f}",
                            f"{r['ndcg']:.6f}", f"{r['map']:.6f}",
                            f"{r['mrr']:.6f}", f"{r['catalog_coverage']:.6f}",
                            f"{r['rec_coverage']:.6f}", f"{r['avg_score']:.6f}"])
    print(f"[compare] wrote {config.MODEL_COMPARISON_CSV}")


def _print_table(results):
    print("\n" + "=" * 108)
    print("MODEL COMPARISON  (test set; higher is better except AvgScore which is "
          "model-internal)")
    print("=" * 108)
    header = (f"{'Model':<12}{'K':>3} | {'Prec':>7}{'Recall':>8}{'F1':>7}"
              f"{'HR':>7}{'NDCG':>7}{'MAP':>7}{'MRR':>7} | {'CatCov':>7}{'RecCov':>7}")
    for k in config.K_VALUES:
        print("-" * 108)
        print(header)
        print("-" * 108)
        for name, per_k in results.items():
            r = per_k[k]
            print(f"{name:<12}{k:>3} | {r['precision']:>7.4f}{r['recall']:>8.4f}"
                  f"{r['f1']:>7.4f}{r['hitrate']:>7.4f}{r['ndcg']:>7.4f}"
                  f"{r['map']:>7.4f}{r['mrr']:>7.4f} | "
                  f"{r['catalog_coverage']:>7.4f}{r['rec_coverage']:>7.4f}")
    print("=" * 108)


def _print_improvements(results):
    if "Hybrid" not in results or "LightGCN" not in results:
        return
    print("\nHYBRID improvement over LightGCN  (Metric_H - Metric_L)/Metric_L * 100")
    print("-" * 60)
    pairs = [("HR@10", 10, "hitrate"), ("Recall@10", 10, "recall"),
             ("NDCG@10", 10, "ndcg"), ("F1@10", 10, "f1")]
    for label, k, key in pairs:
        h = results["Hybrid"][k][key]
        l = results["LightGCN"][k][key]
        if l > 0:
            imp = (h - l) / l * 100.0
            print(f"  {label:<12} LightGCN={l:.4f}  Hybrid={h:.4f}  "
                  f"improvement={imp:+.2f}%")
        else:
            print(f"  {label:<12} LightGCN={l:.4f}  Hybrid={h:.4f}  "
                  f"improvement=n/a (LightGCN metric is 0)")
    print("-" * 60)


if __name__ == "__main__":
    run()
