

from __future__ import annotations

import json, os
import numpy as np
import torch

import config


def setup(tmp="results_t7", seed=2, M=120, N=60, G=6):
    for k, v in {"RESULTS_DIR": tmp, "PROCESSED_DIR": f"{tmp}/processed",
                 "MAPPINGS_DIR": f"{tmp}/mappings", "CACHE_DIR": f"{tmp}/cache",
                 "CHECKPOINT_DIR": f"{tmp}/checkpoints", "FIGURES_DIR": f"{tmp}/figures",
                 "ABLATION_FIG_DIR": f"{tmp}/figures",
                 "REVIEW_EMB_CACHE": f"{tmp}/cache/review_embeddings.pt",
                 "EMOTION_FEATURES_CACHE": f"{tmp}/cache/emotion_features.pt",
                 "XLMR_SENTIMENT_BEST": f"{tmp}/checkpoints/xlmr_sentiment_best.pt",
                 "ABLATION_RESULTS_CSV": f"{tmp}/ablation_results.csv",
                 "FUSION_SENS_CSV": f"{tmp}/fusion_sensitivity.csv",
                 "LAYER_SENS_CSV": f"{tmp}/layer_sensitivity.csv",
                 "EMBEDDING_SENS_CSV": f"{tmp}/embedding_sensitivity.csv",
                 "COMPARE_EPOCHS": 8, "COMPARE_PATIENCE": 3}.items():
        setattr(config, k, v)
    for d in ("processed", "mappings", "cache", "checkpoints", "figures"):
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
        ui = d["user_idx"].to_numpy().astype(np.int64); ii = d["item_idx"].to_numpy().astype(np.int64)
        n = len(d)
        hcache["splits"][name] = {"embeddings": torch.randn(n, 768).half(), "user_idx": ui,
                                  "item_idx": ii, "rating": d["rating"].to_numpy().astype(np.float32),
                                  "has_text": np.ones(n, bool)}
        ecache["splits"][name] = {"probs": torch.softmax(torch.randn(n, 11), 1), "user_idx": ui,
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
    import importlib, ablation
    importlib.reload(ablation)
    res = ablation.run()

    for p in (config.ABLATION_RESULTS_CSV, config.FUSION_SENS_CSV,
              config.LAYER_SENS_CSV, config.EMBEDDING_SENS_CSV):
        assert os.path.exists(p), f"missing {p}"
    for fig in ("fusion_sensitivity.png", "layer_sensitivity.png",
                "embedding_sensitivity.png", "ablation_comparison.png"):
        assert os.path.exists(os.path.join(config.ABLATION_FIG_DIR, fig)), fig

    # ablation csv: 7 models x 3 K = 21 rows
    import csv
    with open(config.ABLATION_RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    models = {r["Model"] for r in rows}
    assert len(rows) == 7 * len(config.K_VALUES), len(rows)
    assert "Proposed" in models and "A1_LightGCN" in models
    for r in rows:
        for c in ("Precision", "Recall", "F1", "HitRate", "NDCG", "MAP", "MRR", "Coverage"):
            assert -1e-6 <= float(r[c]) <= 1 + 1e-6

    # selection came from the sweeps (validation-based)
    assert res["best"]["alpha"] in config.ALPHA_VALUES
    assert res["best"]["layers"] in config.LAYER_VALUES
    assert res["best"]["dim"] in config.EMBEDDING_DIMS
    # verify best alpha == argmax of VALIDATION ndcg@10 among alpha rows
    va = max(res["alpha"], key=lambda r: r[1][10]["ndcg"])[0]
    assert res["best"]["alpha"] == va, "best alpha must be val-selected"
    print("\n[check] 4 CSVs + 4 PNGs written; ablation has 7 models x 3 K;")
    print("[check] best config selected on VALIDATION NDCG@10 (not test).")
    print("ALL ABLATION SELF-TESTS PASSED.")


if __name__ == "__main__":
    main()
