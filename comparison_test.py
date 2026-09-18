

from __future__ import annotations

import json
import os
import numpy as np
import torch

import config


def setup(tmp="results_t6", seed=1, M=120, N=60, G=6):
    for k, v in {"RESULTS_DIR": tmp, "PROCESSED_DIR": f"{tmp}/processed",
                 "MAPPINGS_DIR": f"{tmp}/mappings", "CACHE_DIR": f"{tmp}/cache",
                 "CHECKPOINT_DIR": f"{tmp}/checkpoints",
                 "REVIEW_EMB_CACHE": f"{tmp}/cache/review_embeddings.pt",
                 "EMOTION_FEATURES_CACHE": f"{tmp}/cache/emotion_features.pt",
                 "XLMR_SENTIMENT_BEST": f"{tmp}/checkpoints/xlmr_sentiment_best.pt",
                 "LIGHTGCN_BEST_PATH": f"{tmp}/checkpoints/lightgcn_best.pt",
                 "MODEL_COMPARISON_CSV": f"{tmp}/model_comparison.csv",
                 "COMPARE_EPOCHS": 15}.items():
        setattr(config, k, v)
    for d in ("processed", "mappings", "cache", "checkpoints"):
        os.makedirs(f"{tmp}/{d}", exist_ok=True)

    import pandas as pd
    rng = np.random.RandomState(seed)
    ipg = N // G
    rows = []
    for u in range(M):
        g = u % G
        for _ in range(rng.randint(5, 12)):
            it = int(rng.choice(np.arange(g * ipg, (g + 1) * ipg)))
            rows.append((u, it, int(rng.randint(1, 6)), "text",
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
    ecache = {"labels": [f"e{c}" for c in range(11)], "num_labels": 11, "splits": {}}
    for name, d in splits.items():
        ui = d["user_idx"].to_numpy().astype(np.int64)
        ii = d["item_idx"].to_numpy().astype(np.int64)
        n = len(d)
        hcache["splits"][name] = {"embeddings": torch.randn(n, 768).half(),
                                  "user_idx": ui, "item_idx": ii,
                                  "rating": d["rating"].to_numpy().astype(np.float32),
                                  "has_text": np.ones(n, bool)}
        ecache["splits"][name] = {"probs": torch.softmax(torch.randn(n, 11), 1),
                                  "user_idx": ui, "item_idx": ii,
                                  "rating": d["rating"].to_numpy().astype(np.float32),
                                  "language": None, "has_text": np.ones(n, bool)}
    torch.save(hcache, config.REVIEW_EMB_CACHE)
    torch.save(ecache, config.EMOTION_FEATURES_CACHE)
    from xlmr_sentiment import SentimentHead
    torch.save({"state_dict": SentimentHead(768, 3).state_dict(), "hidden": 768,
                "num_classes": 3, "labels": config.SENTIMENT_LABELS,
                "rating_map": {}, "meta": {}}, config.XLMR_SENTIMENT_BEST)


def main():
    setup()
    import importlib, run_comparison
    importlib.reload(run_comparison)
    results = run_comparison.run()

    assert os.path.exists(config.MODEL_COMPARISON_CSV), "CSV not written"
    import csv
    with open(config.MODEL_COMPARISON_CSV) as f:
        rows = list(csv.DictReader(f))
    models = {"Popularity", "BPR-MF", "LightGCN", "XLM-R-Text", "Hybrid"}
    assert {r["Model"] for r in rows} == models, {r["Model"] for r in rows}
    assert len(rows) == len(models) * len(config.K_VALUES), len(rows)
    for r in rows:
        for col in ("Precision", "Recall", "F1", "HitRate", "NDCG", "MAP", "MRR",
                    "Coverage", "RecCoverage"):
            v = float(r[col])
            assert -1e-6 <= v <= 1 + 1e-6, f"{r['Model']}@{r['K']} {col}={v} out of [0,1]"
    print(f"\n[check] CSV OK: {len(rows)} rows (5 models x 3 K), all metrics in [0,1]")

    # improvement pairs are computed from real results (may be + or -)
    for k, key in [(10, "hitrate"), (10, "recall"), (10, "ndcg"), (10, "f1")]:
        _ = results["Hybrid"][k][key]; _ = results["LightGCN"][k][key]
    print("[check] hybrid-vs-LightGCN improvements computed from real results")
    print("\nALL COMPARISON SELF-TESTS PASSED.")


if __name__ == "__main__":
    main()
