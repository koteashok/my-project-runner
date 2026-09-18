
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

import config
from lightgcn import LightGCN, build_norm_adjacency, set_seed, get_device
from lightgcn_evaluate import load_splits, build_pos_dict, infer_sizes, merge_exclusions
from baselines import TextRecommender
from hybrid_model import HybridRecommender
from comparison_metrics import evaluate_full, embedding_score_provider
from run_comparison import train_bpr



def build_components(M: int, N: int, device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:

    from review_embeddings import load_cache, aggregate_user_item
    from emotion_features import load_emotion_cache
    from emotion_representation import _sentiment_rep

    h_cache = load_cache()
    tr = h_cache["splits"]["train"]
    h = tr["embeddings"].float()
    ui, ii = tr["user_idx"], tr["item_idx"]

    e_cache = load_emotion_cache()
    etr = e_cache["splits"]["train"]
    if not (np.array_equal(etr["user_idx"], ui) and np.array_equal(etr["item_idx"], ii)):
        raise RuntimeError("emotion cache misaligned with review cache.")
    e = etr["probs"].float()
    s = _sentiment_rep(h)

    comps = {}
    for name, mat in (("h", h), ("e", e), ("s", s)):
        agg = aggregate_user_item(mat, ui, ii, M, N)
        comps[name] = (agg["z_user"].to(device), agg["z_item"].to(device))
    return comps


def concat_components(comps, names: List[str]):
    zu = torch.cat([comps[n][0] for n in names], dim=1)
    zi = torch.cat([comps[n][1] for n in names], dim=1)
    return zu, zi


# ----------------------------------------------------------------------------- #
# Train + evaluate one model                                                     #
# ----------------------------------------------------------------------------- #
class _Ctx:
    """Shared training/eval context."""
    def __init__(self):
        self.device = get_device()
        self.train_df, self.val_df, self.test_df = load_splits()
        self.M, self.N = infer_sizes(self.train_df, self.val_df, self.test_df)
        self.users_arr = self.train_df["user_idx"].to_numpy().astype(np.int64)
        self.pos_arr = self.train_df["item_idx"].to_numpy().astype(np.int64)
        self.train_pos = build_pos_dict(self.train_df, self.N)
        self.val_pos = build_pos_dict(self.val_df, self.N)
        self.test_pos = build_pos_dict(self.test_df, self.N)
        self.exclude = merge_exclusions(self.train_pos, self.val_pos)
        self.norm_adj, self.norm_R, _, _ = build_norm_adjacency(
            self.users_arr, self.pos_arr, self.M, self.N, self.device)
        self.comps = build_components(self.M, self.N, self.device)


def _eval(model, ctx) -> Tuple[dict, dict]:
    uemb, iemb = model.get_all_embeddings()
    prov = embedding_score_provider(uemb, iemb)
    val = evaluate_full(prov, ctx.val_pos, ctx.train_pos, ctx.N, config.K_VALUES,
                        ctx.device, config.EVAL_USER_BATCH)
    test = evaluate_full(prov, ctx.test_pos, ctx.exclude, ctx.N, config.K_VALUES,
                         ctx.device, config.EVAL_USER_BATCH)
    return val, test


def _train_eval(model, ctx, tag) -> Tuple[dict, dict]:
    set_seed(config.RANDOM_SEED)
    train_bpr(model, ctx.users_arr, ctx.pos_arr, ctx.train_pos, ctx.val_pos,
              ctx.N, ctx.device, tag)
    return _eval(model, ctx)


# -- model builders ----------------------------------------------------------- #
def _lightgcn(ctx, layers=None, dim=None):
    return LightGCN(ctx.M, ctx.N, ctx.norm_adj,
                    embedding_dim=dim or config.EMBEDDING_DIM,
                    num_layers=layers or config.NUM_LAYERS,
                    reg_lambda=config.HYBRID_L2, norm_R=ctx.norm_R).to(ctx.device)


def _text_only(ctx):
    zu, zi = ctx.comps["h"]
    return TextRecommender(zu, zi, common_dim=config.FUSION_COMMON_DIM,
                           reg_lambda=config.HYBRID_L2).to(ctx.device)


def _hybrid(ctx, names, fusion="weighted", alpha=0.5, layers=None, dim=None):
    zu, zi = concat_components(ctx.comps, names)
    d = dim or config.EMBEDDING_DIM
    return HybridRecommender(ctx.M, ctx.N, ctx.norm_adj, zu, zi,
                             lgcn_dim=d, common_dim=d,
                             num_layers=layers or config.NUM_LAYERS,
                             fusion=fusion, alpha=alpha,
                             reg_lambda=config.HYBRID_L2, norm_R=ctx.norm_R).to(ctx.device)


# ----------------------------------------------------------------------------- #
# Ablation                                                                       #
# ----------------------------------------------------------------------------- #
def run_ablation(ctx) -> Dict[str, Tuple[dict, dict]]:
    print("\n### ABLATION (default hyperparameters) ###")
    specs = [
        ("A1_LightGCN", lambda: _lightgcn(ctx)),
        ("A2_Semantic_only", lambda: _text_only(ctx)),
        ("A3_LGCN+semantic", lambda: _hybrid(ctx, ["h"])),
        ("A4_LGCN+sentiment", lambda: _hybrid(ctx, ["s"])),
        ("A5_LGCN+emotion", lambda: _hybrid(ctx, ["e"])),
        ("A6_LGCN+sem+sent+emo", lambda: _hybrid(ctx, ["h", "e", "s"])),
    ]
    out = {}
    for name, build in specs:
        val, test = _train_eval(build(), ctx, name)
        out[name] = (val, test)
        print(f"  {name:<22} val_NDCG@10={val[10]['ndcg']:.4f} "
              f"test_NDCG@10={test[10]['ndcg']:.4f}")
    return out


# ----------------------------------------------------------------------------- #
# Sensitivity sweeps (full hybrid h||e||s)                                        #
# ----------------------------------------------------------------------------- #
def run_alpha_sweep(ctx):
    print("\n### FUSION-WEIGHT SWEEP (alpha) ###")
    rows = []
    for a in config.ALPHA_VALUES:
        val, test = _train_eval(_hybrid(ctx, ["h", "e", "s"], fusion="weighted",
                                        alpha=a), ctx, f"alpha={a}")
        rows.append((a, val, test))
        print(f"  alpha={a}: val_NDCG@10={val[10]['ndcg']:.4f} "
              f"test_NDCG@10={test[10]['ndcg']:.4f}")
    return rows


def run_layer_sweep(ctx):
    print("\n### LIGHTGCN-DEPTH SWEEP (num_layers) ###")
    rows = []
    for L in config.LAYER_VALUES:
        val, test = _train_eval(_hybrid(ctx, ["h", "e", "s"], layers=L),
                                ctx, f"layers={L}")
        rows.append((L, val, test))
        print(f"  layers={L}: val_NDCG@10={val[10]['ndcg']:.4f} "
              f"test_NDCG@10={test[10]['ndcg']:.4f}")
    return rows


def run_dim_sweep(ctx):
    print("\n### EMBEDDING-DIMENSION SWEEP ###")
    rows = []
    for d in config.EMBEDDING_DIMS:
        val, test = _train_eval(_hybrid(ctx, ["h", "e", "s"], dim=d),
                                ctx, f"dim={d}")
        rows.append((d, val, test))
        print(f"  dim={d}: val_NDCG@10={val[10]['ndcg']:.4f} "
              f"test_NDCG@10={test[10]['ndcg']:.4f}")
    return rows


# ----------------------------------------------------------------------------- #
# CSV writers                                                                     #
# ----------------------------------------------------------------------------- #
def _write_ablation_csv(ablation, proposed):
    config.ensure_dirs()
    cols = ["Model", "K", "Precision", "Recall", "F1", "HitRate", "NDCG",
            "MAP", "MRR", "Coverage", "ValNDCG@10"]
    with open(config.ABLATION_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        items = list(ablation.items()) + [("Proposed", proposed)]
        for name, (val, test) in items:
            v10 = val[config.SELECT_K][config.SELECT_METRIC]
            for k in config.K_VALUES:
                r = test[k]
                w.writerow([name, k, f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                            f"{r['f1']:.6f}", f"{r['hitrate']:.6f}", f"{r['ndcg']:.6f}",
                            f"{r['map']:.6f}", f"{r['mrr']:.6f}",
                            f"{r['catalog_coverage']:.6f}", f"{v10:.6f}"])
    print(f"[ablation] wrote {config.ABLATION_RESULTS_CSV}")


def _write_sweep_csv(path, xname, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([xname, "val_NDCG@10", "test_NDCG@10", "test_Recall@10",
                    "test_F1@10", "test_HitRate@10", "test_MAP@10", "test_MRR@10"])
        for x, val, test in rows:
            t = test[10]
            w.writerow([x, f"{val[10]['ndcg']:.6f}", f"{t['ndcg']:.6f}",
                        f"{t['recall']:.6f}", f"{t['f1']:.6f}", f"{t['hitrate']:.6f}",
                        f"{t['map']:.6f}", f"{t['mrr']:.6f}"])
    print(f"[ablation] wrote {path}")


# ----------------------------------------------------------------------------- #
# Plots                                                                          #
# ----------------------------------------------------------------------------- #
def _line_plot(xs, val_ys, test_ys, xlabel, title, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, val_ys, "o-", label="validation", color="#4C72B0")
    ax.plot(xs, test_ys, "s--", label="test", color="#DD8452")
    ax.set_xlabel(xlabel); ax.set_ylabel("NDCG@10"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[ablation] wrote {path}")


def _bar_plot(names, vals, title, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 4.2))
    x = np.arange(len(names))
    ax.bar(x, vals, color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("NDCG@10 (test)"); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[ablation] wrote {path}")


# ----------------------------------------------------------------------------- #
# Orchestration                                                                  #
# ----------------------------------------------------------------------------- #
def run():
    set_seed(config.RANDOM_SEED)
    ctx = _Ctx()
    print(f"[ablation] device={ctx.device} users={ctx.M} items={ctx.N} "
          f"(select on VALIDATION {config.SELECT_METRIC}@{config.SELECT_K})")
    config.ensure_figures_dir()

    ablation = run_ablation(ctx)
    alpha_rows = run_alpha_sweep(ctx)
    layer_rows = run_layer_sweep(ctx)
    dim_rows = run_dim_sweep(ctx)

    # --- select best config on VALIDATION NDCG@10 ----------------------------
    sk, sm = config.SELECT_K, config.SELECT_METRIC
    best_alpha = max(alpha_rows, key=lambda r: r[1][sk][sm])[0]
    best_layers = max(layer_rows, key=lambda r: r[1][sk][sm])[0]
    best_dim = max(dim_rows, key=lambda r: r[1][sk][sm])[0]
    print(f"\n[selection] best on VALIDATION {sm}@{sk}: "
          f"alpha={best_alpha}, layers={best_layers}, dim={best_dim}")

    # --- train Proposed with the selected config -----------------------------
    proposed = _train_eval(
        _hybrid(ctx, ["h", "e", "s"], fusion="weighted", alpha=best_alpha,
                layers=best_layers, dim=best_dim), ctx, "Proposed(best)")

    # --- write CSVs ----------------------------------------------------------
    _write_ablation_csv(ablation, proposed)
    _write_sweep_csv(config.FUSION_SENS_CSV, "alpha", alpha_rows)
    _write_sweep_csv(config.LAYER_SENS_CSV, "num_layers", layer_rows)
    _write_sweep_csv(config.EMBEDDING_SENS_CSV, "embedding_dim", dim_rows)

    # --- plots ---------------------------------------------------------------
    fig_dir = config.ABLATION_FIG_DIR
    _line_plot(config.ALPHA_VALUES,
               [r[1][10]["ndcg"] for r in alpha_rows],
               [r[2][10]["ndcg"] for r in alpha_rows],
               "fusion weight α", "NDCG@10 vs fusion weight",
               os.path.join(fig_dir, "fusion_sensitivity.png"))
    _line_plot(config.LAYER_VALUES,
               [r[1][10]["ndcg"] for r in layer_rows],
               [r[2][10]["ndcg"] for r in layer_rows],
               "LightGCN layers", "NDCG@10 vs LightGCN depth",
               os.path.join(fig_dir, "layer_sensitivity.png"))
    _line_plot(config.EMBEDDING_DIMS,
               [r[1][10]["ndcg"] for r in dim_rows],
               [r[2][10]["ndcg"] for r in dim_rows],
               "embedding dimension", "NDCG@10 vs embedding dimension",
               os.path.join(fig_dir, "embedding_sensitivity.png"))
    ab_names = list(ablation.keys()) + ["Proposed"]
    ab_vals = [ablation[n][1][10]["ndcg"] for n in ablation] + [proposed[1][10]["ndcg"]]
    _bar_plot(ab_names, ab_vals, "Ablation comparison (test NDCG@10)",
              os.path.join(fig_dir, "ablation_comparison.png"))

    # --- summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION (selected on validation NDCG@10)")
    print("=" * 60)
    print(f"  alpha={best_alpha}  layers={best_layers}  dim={best_dim}")
    print(f"  Proposed  val_NDCG@10={proposed[0][10]['ndcg']:.4f}  "
          f"test_NDCG@10={proposed[1][10]['ndcg']:.4f}")
    print("=" * 60)
    return {"ablation": ablation, "alpha": alpha_rows, "layers": layer_rows,
            "dims": dim_rows, "proposed": proposed,
            "best": {"alpha": best_alpha, "layers": best_layers, "dim": best_dim}}


if __name__ == "__main__":
    run()
