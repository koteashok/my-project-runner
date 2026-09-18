"""
final_analysis.py
=================
Final multilingual & emotion-aware analysis (Phase 8).

Produces, from ACTUAL execution only (no invented/estimated values):
  - language identification (used if present, else inferred + documented)
  - language statistics (reviews/users/items/avg rating; sentiment & emotion by language)
  - hybrid HR/Recall/NDCG/F1@10 per sufficiently-represented language
  - emotion behaviour (frequency, avg rating by dominant emotion, sentiment↔emotion,
    hybrid rec performance by dominant emotion where statistically meaningful)
  - final comparison table (Popularity / BPR-MF / LightGCN / XLM-R / Proposed Hybrid)
  - publication-quality figures
  - an auto-generated experimental summary

Association only — NO causal claims are made about emotion and behaviour.

Run:
    python final_analysis.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

import config
from lightgcn import set_seed
from baselines import PopularityRecommender, BPRMF
from comparison_metrics import (
    evaluate_full, embedding_score_provider, popularity_score_provider,
)
from run_comparison import train_bpr
from ablation import _Ctx, _lightgcn, _text_only, _hybrid
from language_utils import LanguageDetector, assign_languages
from xlmr_sentiment import rating_to_sentiment

MET10 = ("hitrate", "recall", "ndcg", "f1", "map", "mrr")


# ----------------------------------------------------------------------------- #
# Train the five models (kept as objects)                                        #
# ----------------------------------------------------------------------------- #
def train_models(ctx):
    print("[final] training 5 models (shared BPR budget)...")
    pop = PopularityRecommender(ctx.M, ctx.N); pop.fit(ctx.pos_arr)
    mf = BPRMF(ctx.M, ctx.N, dim=config.COMPARE_DIM, reg_lambda=config.HYBRID_L2).to(ctx.device)
    lgcn = _lightgcn(ctx)
    text = _text_only(ctx)
    hybrid = _hybrid(ctx, ["h", "e", "s"], fusion=config.FUSION_STRATEGY, alpha=config.ALPHA)
    for name, m in (("BPR-MF", mf), ("LightGCN", lgcn), ("XLM-RoBERTa", text),
                    ("Proposed Hybrid", hybrid)):
        set_seed(config.RANDOM_SEED)
        train_bpr(m, ctx.users_arr, ctx.pos_arr, ctx.train_pos, ctx.val_pos,
                  ctx.N, ctx.device, name)
    return {"Popularity": pop, "BPR-MF": mf, "LightGCN": lgcn,
            "XLM-RoBERTa": text, "Proposed Hybrid": hybrid}


def provider_of(model, ctx):
    if isinstance(model, PopularityRecommender):
        return popularity_score_provider(model.popularity_tensor(ctx.device))
    u, i = model.get_all_embeddings()
    return embedding_score_provider(u, i)


def eval10(provider, test_pos, exclude, ctx) -> Dict[str, float]:
    r = evaluate_full(provider, test_pos, exclude, ctx.N, [10], ctx.device,
                      config.EVAL_USER_BATCH)[10]
    return r


# ----------------------------------------------------------------------------- #
# Combined per-row table with language + emotion + sentiment                     #
# ----------------------------------------------------------------------------- #
def build_row_table(ctx, detector):
    """Return a dict of arrays over ALL split rows plus per-split test mapping."""
    from emotion_features import load_emotion_cache
    ecache = load_emotion_cache()
    labels = ecache["labels"]

    frames = {"train": ctx.train_df, "validation": ctx.val_df, "test": ctx.test_df}
    all_user, all_item, all_rating, all_lang, all_emo, all_split = [], [], [], [], [], []
    inferred_any = False
    test_user_lang, test_user_emo = {}, {}

    for split, df in frames.items():
        if df is None or len(df) == 0:
            continue
        lang, inferred = assign_languages(df, detector)
        inferred_any = inferred_any or inferred
        es = ecache["splits"].get(split)
        if es is not None and np.array_equal(es["user_idx"],
                                             df["user_idx"].to_numpy()):
            dom = np.array([labels[i] for i in es["dominant_idx"]], dtype=object)
        else:
            dom = np.array(["unknown"] * len(df), dtype=object)
        rating = (df["rating"].to_numpy().astype(float) if "rating" in df.columns
                  else np.full(len(df), np.nan))
        u = df["user_idx"].to_numpy()
        it = df["item_idx"].to_numpy()
        all_user.append(u); all_item.append(it); all_rating.append(rating)
        all_lang.append(lang); all_emo.append(dom)
        all_split.append(np.array([split] * len(df), dtype=object))
        if split == "test":
            for uu, lg, em in zip(u, lang, dom):
                test_user_lang[int(uu)] = str(lg)
                test_user_emo[int(uu)] = str(em)

    T = {"user": np.concatenate(all_user), "item": np.concatenate(all_item),
         "rating": np.concatenate(all_rating), "language": np.concatenate(all_lang),
         "emotion": np.concatenate(all_emo), "split": np.concatenate(all_split)}
    T["sentiment_idx"] = rating_to_sentiment(T["rating"])
    return T, labels, inferred_any, test_user_lang, test_user_emo


# ----------------------------------------------------------------------------- #
# Language analysis                                                              #
# ----------------------------------------------------------------------------- #
def language_analysis(ctx, T, inferred, test_user_lang, hybrid_provider):
    langs = T["language"]
    uniq = sorted(set(langs.tolist()))
    print(f"[final] languages detected (inferred={inferred}): "
          f"{ {l: int((langs==l).sum()) for l in uniq} }")

    # per-language basic stats
    stats = {}
    for l in uniq:
        m = langs == l
        stats[l] = {
            "reviews": int(m.sum()),
            "users": int(len(np.unique(T["user"][m]))),
            "items": int(len(np.unique(T["item"][m]))),
            "avg_rating": float(np.nanmean(T["rating"][m])) if m.any() else float("nan"),
        }

    # per-language hybrid rec metrics (test users grouped by test-review language)
    groups = defaultdict(list)
    for u in ctx.test_pos:
        groups[test_user_lang.get(int(u), "unknown")].append(u)
    rec = {}
    for l, users in groups.items():
        if len(users) >= config.LANG_MIN_TEST_USERS:
            sub = {u: ctx.test_pos[u] for u in users}
            r = eval10(hybrid_provider, sub, ctx.exclude, ctx)
            rec[l] = {"test_users": len(users), **{k: r[k] for k in
                       ("hitrate", "recall", "ndcg", "f1")}}
        else:
            rec[l] = {"test_users": len(users)}

    # write language_results.csv
    config.ensure_result_tree()
    with open(config.LANG_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Language", "Reviews", "Users", "Items", "AvgRating",
                    "TestUsers", "HR@10", "Recall@10", "NDCG@10", "F1@10", "Inferred"])
        for l in uniq:
            s = stats[l]; rr = rec.get(l, {})
            def g(k): return f"{rr[k]:.6f}" if k in rr else ""
            w.writerow([l, s["reviews"], s["users"], s["items"],
                        f"{s['avg_rating']:.4f}" if s["avg_rating"] == s["avg_rating"] else "",
                        rr.get("test_users", 0), g("hitrate"), g("recall"),
                        g("ndcg"), g("f1"), inferred])
    print(f"[final] wrote {config.LANG_RESULTS_CSV}")

    # sentiment & emotion distribution by language -> tables
    _crosstab_csv(config.SENTIMENT_BY_LANG_CSV, langs, T["sentiment_idx"],
                  uniq, config.SENTIMENT_LABELS, "language")
    emo_labels = sorted(set(T["emotion"].tolist()))
    _crosstab_labels_csv(config.EMOTION_BY_LANG_CSV, langs, T["emotion"], uniq,
                         emo_labels, "language")
    return stats, rec, uniq


def _crosstab_csv(path, rows, col_idx, row_vals, col_labels, rowname):
    C = len(col_labels)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([rowname] + list(col_labels))
        for rv in row_vals:
            m = rows == rv
            counts = np.bincount(col_idx[m][col_idx[m] >= 0], minlength=C)
            w.writerow([rv] + [int(x) for x in counts])


def _crosstab_labels_csv(path, rows, col_labels_arr, row_vals, col_labels, rowname):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([rowname] + list(col_labels))
        for rv in row_vals:
            m = rows == rv
            counts = [int((col_labels_arr[m] == c).sum()) for c in col_labels]
            w.writerow([rv] + counts)


# ----------------------------------------------------------------------------- #
# Emotion behaviour analysis                                                     #
# ----------------------------------------------------------------------------- #
def emotion_analysis(ctx, T, test_user_emo, hybrid_provider):
    emo = T["emotion"]
    emo_labels = sorted(set(emo.tolist()))
    freq = {l: int((emo == l).sum()) for l in emo_labels}
    avg_rating = {l: (float(np.nanmean(T["rating"][emo == l]))
                      if (emo == l).any() else float("nan")) for l in emo_labels}

    with open(config.EMOTION_BEHAVIOR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["dominant_emotion", "frequency", "avg_rating"])
        for l in emo_labels:
            w.writerow([l, freq[l],
                        f"{avg_rating[l]:.4f}" if avg_rating[l] == avg_rating[l] else ""])
    print(f"[final] wrote {config.EMOTION_BEHAVIOR_CSV}")

    # sentiment vs emotion crosstab
    _crosstab_csv(os.path.join(config.TABLES_DIR, "sentiment_vs_emotion.csv"),
                  emo, T["sentiment_idx"], emo_labels, config.SENTIMENT_LABELS,
                  "dominant_emotion")

    # per-dominant-emotion hybrid rec metrics (where statistically meaningful)
    groups = defaultdict(list)
    for u in ctx.test_pos:
        groups[test_user_emo.get(int(u), "unknown")].append(u)
    with open(config.EMOTION_REC_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dominant_emotion", "TestUsers", "HR@10", "Recall@10",
                    "NDCG@10", "F1@10", "sufficient"])
        emo_rec = {}
        for l, users in sorted(groups.items()):
            suff = len(users) >= config.EMOTION_MIN_TEST_USERS
            if suff:
                sub = {u: ctx.test_pos[u] for u in users}
                r = eval10(hybrid_provider, sub, ctx.exclude, ctx)
                emo_rec[l] = r
                w.writerow([l, len(users), f"{r['hitrate']:.6f}",
                            f"{r['recall']:.6f}", f"{r['ndcg']:.6f}",
                            f"{r['f1']:.6f}", "yes"])
            else:
                w.writerow([l, len(users), "", "", "", "", "no"])
    print(f"[final] wrote {config.EMOTION_REC_CSV}")
    return freq, avg_rating, emo_labels, emo_rec


# ----------------------------------------------------------------------------- #
# Figures                                                                        #
# ----------------------------------------------------------------------------- #
def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_final_comparison(final_tbl):
    plt = _mpl()
    models = list(final_tbl.keys())
    metrics = ["hitrate", "recall", "ndcg", "f1"]
    labels = ["HR@10", "Recall@10", "NDCG@10", "F1@10"]
    x = np.arange(len(models)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, (mk, lb) in enumerate(zip(metrics, labels)):
        ax.bar(x + (j - 1.5) * w, [final_tbl[m][mk] for m in models], w, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.set_title("Final model comparison @10")
    ax.legend(ncol=4, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(config.FIGURES_DIR, "final_model_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"[final] wrote {p}")


def fig_bar(names, vals, ylabel, title, fname, rotate=30):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    ax.bar(np.arange(len(names)), vals, color="#4C72B0")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=rotate, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); p = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(p, dpi=150); plt.close(fig); print(f"[final] wrote {p}")


def fig_heatmap(matrix, row_labels, col_labels, title, fname):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels)), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title(title)
    fig.tight_layout(); p = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(p, dpi=150); plt.close(fig); print(f"[final] wrote {p}")


# ----------------------------------------------------------------------------- #
# Orchestration                                                                  #
# ----------------------------------------------------------------------------- #
def run():
    set_seed(config.RANDOM_SEED)
    config.ensure_result_tree()
    ctx = _Ctx()
    print(f"[final] device={ctx.device} users={ctx.M} items={ctx.N} "
          f"test_users={len(ctx.test_pos)}")

    models = train_models(ctx)

    # ---- final comparison table @10 -----------------------------------------
    final_tbl = {}
    for name, model in models.items():
        final_tbl[name] = eval10(provider_of(model, ctx), ctx.test_pos,
                                 ctx.exclude, ctx)
    _write_final_table(final_tbl)
    _print_final_table(final_tbl)
    fig_final_comparison(final_tbl)

    hybrid_provider = provider_of(models["Proposed Hybrid"], ctx)

    # ---- language analysis ---------------------------------------------------
    detector = LanguageDetector(seed=config.LANG_DETECT_SEED,
                                min_chars=config.LANG_DETECT_MIN_CHARS)
    T, emo_labels_cache, inferred, tu_lang, tu_emo = build_row_table(ctx, detector)
    lstats, lrec, uniq_langs = language_analysis(ctx, T, inferred, tu_lang,
                                                 hybrid_provider)
    fig_bar(uniq_langs, [lstats[l]["reviews"] for l in uniq_langs], "reviews",
            "Reviews per language" + (" (inferred)" if inferred else ""),
            "language_distribution.png")
    suff_langs = [l for l in uniq_langs if "ndcg" in lrec.get(l, {})]
    if suff_langs:
        fig_bar(suff_langs, [lrec[l]["ndcg"] for l in suff_langs], "NDCG@10",
                "Hybrid NDCG@10 by language (sufficiently represented)",
                "language_ndcg.png")

    # ---- emotion behaviour ---------------------------------------------------
    freq, avg_rating, emo_labels, emo_rec = emotion_analysis(ctx, T, tu_emo,
                                                             hybrid_provider)
    fig_bar(emo_labels, [freq[l] for l in emo_labels], "count",
            "Dominant-emotion frequency (pseudo-labels)", "emotion_frequency.png")
    fig_bar(emo_labels, [avg_rating[l] if avg_rating[l] == avg_rating[l] else 0
                         for l in emo_labels], "avg rating",
            "Average rating by dominant emotion (association, not causal)",
            "rating_by_emotion.png")
    # sentiment x emotion heatmap
    S = len(config.SENTIMENT_LABELS)
    mat = np.zeros((len(emo_labels), S))
    for ri, el in enumerate(emo_labels):
        m = T["emotion"] == el
        si = T["sentiment_idx"][m]
        mat[ri] = np.bincount(si[si >= 0], minlength=S)
    fig_heatmap(mat, emo_labels, config.SENTIMENT_LABELS,
                "Dominant emotion × rating-derived sentiment", "sentiment_emotion_heatmap.png")

    # ---- experimental summary (measured values only) ------------------------
    _write_summary(ctx, final_tbl, lstats, lrec, uniq_langs, inferred,
                   freq, avg_rating, emo_rec)
    print(f"\n[final] all artefacts under {config.RESULTS_DIR}/ "
          f"(figures/ tables/ final_results/ checkpoints/ cache/)")
    return {"final_table": final_tbl, "language": (lstats, lrec),
            "emotion": (freq, avg_rating, emo_rec), "inferred": inferred}


def _write_final_table(final_tbl):
    with open(config.FINAL_COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "HR@10", "Recall@10", "NDCG@10", "F1@10", "MAP@10", "MRR@10"])
        for name, r in final_tbl.items():
            w.writerow([name, f"{r['hitrate']:.6f}", f"{r['recall']:.6f}",
                        f"{r['ndcg']:.6f}", f"{r['f1']:.6f}", f"{r['map']:.6f}",
                        f"{r['mrr']:.6f}"])
    print(f"[final] wrote {config.FINAL_COMPARISON_CSV}")


def _print_final_table(final_tbl):
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (test @10)")
    print("=" * 78)
    print(f"{'Model':<18}{'HR@10':>9}{'Recall@10':>11}{'NDCG@10':>9}"
          f"{'F1@10':>9}{'MAP@10':>9}{'MRR@10':>9}")
    print("-" * 78)
    for name, r in final_tbl.items():
        print(f"{name:<18}{r['hitrate']:>9.4f}{r['recall']:>11.4f}{r['ndcg']:>9.4f}"
              f"{r['f1']:>9.4f}{r['map']:>9.4f}{r['mrr']:>9.4f}")
    print("=" * 78)


def _write_summary(ctx, final_tbl, lstats, lrec, uniq_langs, inferred,
                   freq, avg_rating, emo_rec):
    lines = []
    A = lines.append
    A("# Experimental Summary (measured results only)\n")
    A(f"- Users: {ctx.M}  Items: {ctx.N}  Test users: {len(ctx.test_pos)}")
    A(f"- Language labels: {'INFERRED by ' + config.LANG_DETECT_BACKEND if inferred else 'from dataset field'} "
      f"(the dataset has no human emotion labels; emotion features are model-derived pseudo-labels).\n")

    A("## Final comparison (test @10)")
    A("| Model | HR@10 | Recall@10 | NDCG@10 | F1@10 | MAP@10 | MRR@10 |")
    A("|---|---|---|---|---|---|---|")
    for name, r in final_tbl.items():
        A(f"| {name} | {r['hitrate']:.4f} | {r['recall']:.4f} | {r['ndcg']:.4f} "
          f"| {r['f1']:.4f} | {r['map']:.4f} | {r['mrr']:.4f} |")
    # improvement Hybrid vs LightGCN (measured)
    if "Proposed Hybrid" in final_tbl and "LightGCN" in final_tbl:
        h, l = final_tbl["Proposed Hybrid"], final_tbl["LightGCN"]
        A("\n## Hybrid vs LightGCN (measured, @10)")
        for lab, k in (("NDCG", "ndcg"), ("Recall", "recall"), ("HR", "hitrate"), ("F1", "f1")):
            if l[k] > 0:
                A(f"- {lab}@10: {l[k]:.4f} -> {h[k]:.4f} "
                  f"({(h[k]-l[k])/l[k]*100:+.2f}%)")
            else:
                A(f"- {lab}@10: {l[k]:.4f} -> {h[k]:.4f} (n/a)")

    A(f"\n## Languages (n={len(uniq_langs)}, "
      f"{'inferred' if inferred else 'from field'})")
    A("| Language | Reviews | Users | Items | AvgRating | Hybrid NDCG@10 (if sufficient) |")
    A("|---|---|---|---|---|---|")
    for lg in uniq_langs:
        s = lstats[lg]; rr = lrec.get(lg, {})
        nd = f"{rr['ndcg']:.4f}" if "ndcg" in rr else "—"
        ar = f"{s['avg_rating']:.3f}" if s["avg_rating"] == s["avg_rating"] else "—"
        A(f"| {lg} | {s['reviews']} | {s['users']} | {s['items']} | {ar} | {nd} |")
    A(f"\n(Only languages with ≥ {config.LANG_MIN_TEST_USERS} test users get "
      f"recommendation metrics.)")

    A("\n## Emotion (model-derived pseudo-labels; association, not causation)")
    A("| Dominant emotion | Frequency | Avg rating |")
    A("|---|---|---|")
    for e in sorted(freq, key=lambda x: -freq[x]):
        ar = f"{avg_rating[e]:.3f}" if avg_rating[e] == avg_rating[e] else "—"
        A(f"| {e} | {freq[e]} | {ar} |")
    if emo_rec:
        A(f"\nHybrid @10 by dominant emotion (≥ {config.EMOTION_MIN_TEST_USERS} test users):")
        for e, r in emo_rec.items():
            A(f"- {e}: HR={r['hitrate']:.4f} Recall={r['recall']:.4f} "
              f"NDCG={r['ndcg']:.4f} F1={r['f1']:.4f}")
    else:
        A(f"\nNo dominant-emotion group reached ≥ {config.EMOTION_MIN_TEST_USERS} "
          f"test users, so no per-emotion recommendation metrics are reported.")

    A("\n_All values above are produced by actual execution; none are estimated._")
    os.makedirs(os.path.dirname(config.EXPERIMENT_SUMMARY_MD) or ".", exist_ok=True)
    with open(config.EXPERIMENT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[final] wrote {config.EXPERIMENT_SUMMARY_MD}")


if __name__ == "__main__":
    run()
