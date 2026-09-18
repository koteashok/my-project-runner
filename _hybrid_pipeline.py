

from __future__ import annotations

import csv
import os

import pandas as pd

import config
import data_loader
import phase1_dataset


REQUIRED_ROLES = ("user", "item", "review")
PROPOSED_RESULTS_CSV = os.path.join(config.RESULTS_DIR, "proposed_hybrid_results.csv")


def _validate_schema(schema):
    missing = [r for r in REQUIRED_ROLES if not schema.has(r)]
    if missing:
        raise ValueError(
            "The proposed hybrid requires user, item and review-text columns, but "
            f"the dataset is missing: {missing}.\n"
            f"  detected columns: {schema.all_columns}\n"
            f"  resolved -> user={schema.user}, item={schema.item}, "
            f"review={schema.review}, rating={schema.rating}")


def preprocess(raw_df: pd.DataFrame):

    if "__split__" not in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["__split__"] = "full"
    schema = data_loader.detect_schema([c for c in raw_df.columns if c != "__split__"])
    _validate_schema(schema)
    if not schema.has("rating"):
        print("[pipeline] NOTE: no rating column detected — interactions will use "
              "implicit (review-exists) positives, and sentiment labels are skipped.")
    report = data_loader.inspect_and_report(raw_df, None, schema)
    phase1_dataset.run(df=raw_df, schema=schema, report=report)
    return schema


def extract_features():

    import review_embeddings
    review_embeddings.extract_and_cache()          # XLM-RoBERTa semantic h_ui
    import xlmr_sentiment
    xlmr_sentiment.run()                            # sentiment head -> s_ui
    import emotion_features
    emotion_features.extract_and_cache()            # emotion probabilities e_ui


def train():
    import hybrid_train
    hybrid_train.train()


def evaluate():

    import torch
    from hybrid_evaluate import load_hybrid
    from lightgcn_evaluate import build_pos_dict, merge_exclusions
    from comparison_metrics import evaluate_full, embedding_score_provider
    from lightgcn import get_device

    device = get_device()
    model, (train_df, val_df, test_df), (M, N) = load_hybrid(device)
    test_pos = build_pos_dict(test_df, N)
    if not test_pos:
        print("[pipeline] test split is empty — nothing to evaluate.")
        return None
    exclude = merge_exclusions(build_pos_dict(train_df, N), build_pos_dict(val_df, N))

    u, i = model.get_all_embeddings()
    prov = embedding_score_provider(u, i)
    ks = [5, 10, 20]
    res = evaluate_full(prov, test_pos, exclude, N, ks, device, config.EVAL_USER_BATCH)

    # print + save (proposed hybrid ONLY)
    print("\n" + "=" * 78)
    print("PROPOSED HYBRID — TEST RESULTS")
    print("=" * 78)
    print(f"{'K':>3}{'HR':>10}{'Recall':>10}{'NDCG':>10}{'F1':>10}"
          f"{'MAP':>10}{'MRR':>10}{'CatCov':>10}")
    print("-" * 78)
    for k in ks:
        r = res[k]
        print(f"{k:>3}{r['hitrate']:>10.4f}{r['recall']:>10.4f}{r['ndcg']:>10.4f}"
              f"{r['f1']:>10.4f}{r['map']:>10.4f}{r['mrr']:>10.4f}"
              f"{r['catalog_coverage']:>10.4f}")
    print("=" * 78)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(PROPOSED_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "K", "HR", "Recall", "NDCG", "F1", "MAP", "MRR",
                    "CatalogCoverage", "RecCoverage"])
        for k in ks:
            r = res[k]
            w.writerow(["Proposed Hybrid", k, f"{r['hitrate']:.6f}",
                        f"{r['recall']:.6f}", f"{r['ndcg']:.6f}", f"{r['f1']:.6f}",
                        f"{r['map']:.6f}", f"{r['mrr']:.6f}",
                        f"{r['catalog_coverage']:.6f}", f"{r['rec_coverage']:.6f}"])
    print(f"[pipeline] wrote {PROPOSED_RESULTS_CSV}")
    return res


def run_proposed_hybrid(raw_df: pd.DataFrame):
    print(f"[pipeline] proposed hybrid on {len(raw_df):,} raw records "
          f"(fusion={config.FUSION_STRATEGY}, alpha={config.ALPHA})")
    preprocess(raw_df)
    extract_features()
    train()
    return evaluate()
