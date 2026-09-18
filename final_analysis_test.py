"""
final_analysis_test.py
======================
Runs the full Phase 8 final analysis offline on synthetic MULTILINGUAL data +
synthetic Phase 3/4 caches, validating that language detection, per-language and
per-emotion metrics, the final table, figures, and the summary are all produced.

Numbers are from SYNTHETIC data (random review features / random emotion probs) —
they exercise the pipeline, not research performance.
"""

from __future__ import annotations

import json, os
import numpy as np
import torch

import config

MULTI = [
    "I absolutely love this product, it works perfectly and I recommend it",
    "Este producto es excelente y lo recomiendo totalmente a todos",
    "Dieses Produkt ist sehr schlecht und funktioniert überhaupt nicht gut",
    "Ce produit est vraiment fantastique je le recommande vivement à tous",
    "Questo prodotto e davvero eccellente lo consiglio a tutti quanti",
    "Este produto e muito bom e chegou rapido eu recomendo bastante",
]


def setup(tmp="results_t8", seed=3, M=150, N=60, G=6):
    for k, v in {"RESULTS_DIR": tmp, "PROCESSED_DIR": f"{tmp}/processed",
                 "MAPPINGS_DIR": f"{tmp}/mappings", "CACHE_DIR": f"{tmp}/cache",
                 "CHECKPOINT_DIR": f"{tmp}/checkpoints", "FIGURES_DIR": f"{tmp}/figures",
                 "TABLES_DIR": f"{tmp}/tables", "FINAL_RESULTS_DIR": f"{tmp}/final_results",
                 "REVIEW_EMB_CACHE": f"{tmp}/cache/review_embeddings.pt",
                 "EMOTION_FEATURES_CACHE": f"{tmp}/cache/emotion_features.pt",
                 "XLMR_SENTIMENT_BEST": f"{tmp}/checkpoints/xlmr_sentiment_best.pt",
                 "LANG_RESULTS_CSV": f"{tmp}/language_results.csv",
                 "FINAL_COMPARISON_CSV": f"{tmp}/tables/final_comparison.csv",
                 "SENTIMENT_BY_LANG_CSV": f"{tmp}/tables/sentiment_by_language.csv",
                 "EMOTION_BY_LANG_CSV": f"{tmp}/tables/emotion_by_language.csv",
                 "EMOTION_BEHAVIOR_CSV": f"{tmp}/tables/emotion_behavior.csv",
                 "EMOTION_REC_CSV": f"{tmp}/tables/emotion_recommendation.csv",
                 "EXPERIMENT_SUMMARY_MD": f"{tmp}/final_results/experiment_summary.md",
                 "COMPARE_EPOCHS": 8, "COMPARE_PATIENCE": 3,
                 "LANG_MIN_TEST_USERS": 12, "EMOTION_MIN_TEST_USERS": 8}.items():
        setattr(config, k, v)
    for d in ("processed", "mappings", "cache", "checkpoints", "figures",
              "tables", "final_results"):
        os.makedirs(f"{tmp}/{d}", exist_ok=True)

    import pandas as pd
    rng = np.random.RandomState(seed)
    ipg = N // G
    rows = []
    for u in range(M):
        g = u % G
        for _ in range(rng.randint(5, 12)):
            it = int(rng.choice(np.arange(g * ipg, (g + 1) * ipg)))
            rows.append((u, it, int(rng.randint(1, 6)), MULTI[rng.randint(0, len(MULTI))],
                         int(rng.randint(1_500_000_000, 1_700_000_000))))
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "review", "timestamp"])
    tr, va, te = [], [], []
    for u, g in df.groupby("user_idx"):
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

    hcache = {"hidden": 768, "splits": {}}
    ecache = {"labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness",
                         "surprise", "love", "gratitude", "frustration", "contempt"],
              "num_labels": 11, "splits": {}}
    for name, d in splits.items():
        ui = d["user_idx"].to_numpy().astype(np.int64); ii = d["item_idx"].to_numpy().astype(np.int64)
        n = len(d); probs = torch.softmax(torch.randn(n, 11), 1)
        dom = probs.argmax(1).numpy().astype(np.int64)
        hcache["splits"][name] = {"embeddings": torch.randn(n, 768).half(), "user_idx": ui,
                                  "item_idx": ii, "rating": d["rating"].to_numpy().astype(np.float32),
                                  "has_text": np.ones(n, bool)}
        ecache["splits"][name] = {"probs": probs, "dominant_idx": dom, "user_idx": ui,
                                  "item_idx": ii, "rating": d["rating"].to_numpy().astype(np.float32),
                                  "language": None, "has_text": np.ones(n, bool)}
    torch.save(hcache, config.REVIEW_EMB_CACHE)
    torch.save(ecache, config.EMOTION_FEATURES_CACHE)
    from xlmr_sentiment import SentimentHead
    torch.save({"state_dict": SentimentHead(768, 3).state_dict(), "hidden": 768,
                "num_classes": 3, "labels": config.SENTIMENT_LABELS, "rating_map": {}, "meta": {}},
               config.XLMR_SENTIMENT_BEST)


def main():
    setup()
    import importlib, final_analysis
    importlib.reload(final_analysis)
    res = final_analysis.run()

    for p in (config.LANG_RESULTS_CSV, config.FINAL_COMPARISON_CSV,
              config.SENTIMENT_BY_LANG_CSV, config.EMOTION_BY_LANG_CSV,
              config.EMOTION_BEHAVIOR_CSV, config.EMOTION_REC_CSV,
              config.EXPERIMENT_SUMMARY_MD):
        assert os.path.exists(p), f"missing {p}"
    for fig in ("final_model_comparison.png", "language_distribution.png",
                "emotion_frequency.png", "rating_by_emotion.png",
                "sentiment_emotion_heatmap.png"):
        assert os.path.exists(os.path.join(config.FIGURES_DIR, fig)), fig

    import csv
    with open(config.FINAL_COMPARISON_CSV) as f:
        rows = list(csv.DictReader(f))
    models = {r["Model"] for r in rows}
    assert models == {"Popularity", "BPR-MF", "LightGCN", "XLM-RoBERTa",
                      "Proposed Hybrid"}, models
    for r in rows:
        for c in ("HR@10", "Recall@10", "NDCG@10", "F1@10", "MAP@10", "MRR@10"):
            assert 0 <= float(r[c]) <= 1

    with open(config.LANG_RESULTS_CSV) as f:
        lrows = list(csv.DictReader(f))
    assert len(lrows) >= 2, "expected multiple detected languages"
    assert all(r["Inferred"] == "True" for r in lrows), "labels must be marked inferred"
    print(f"\n[check] detected {len(lrows)} languages (all marked inferred);")
    print("[check] final table has 5 models; all @10 metrics in [0,1];")
    print("[check] language/emotion tables, figures, and summary written.")
    print("ALL FINAL-ANALYSIS SELF-TESTS PASSED.")


if __name__ == "__main__":
    main()
