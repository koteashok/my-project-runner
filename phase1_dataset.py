

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import config
import data_loader
import preprocessing as pp


def _save_frame(frame: pd.DataFrame, basename: str) -> list[str]:
    """Persist a frame as parquet and/or csv under results/processed/."""
    paths = []
    cols = [c for c in frame.columns]
    if config.WRITE_PARQUET:
        p = os.path.join(config.PROCESSED_DIR, basename + ".parquet")
        try:
            frame.to_parquet(p, index=False)
            paths.append(p)
        except Exception as e:  # pragma: no cover
            print(f"  [warn] could not write parquet {p}: {e}")
    if config.WRITE_CSV:
        p = os.path.join(config.PROCESSED_DIR, basename + ".csv")
        frame.to_csv(p, index=False)
        paths.append(p)
    return paths


def run(df=None, schema=None, report=None) -> dict:
    """
    Execute the full Phase 1 pipeline.

    Parameters allow injecting an already-loaded (df, schema, report) — used by
    the offline test harness. In normal use they are None and the dataset is
    downloaded from HuggingFace.
    """
    config.ensure_dirs()
    log: dict = {}

    # ---- 1. load + inspect --------------------------------------------------
    if df is None:
        df, ds = data_loader.load_raw_dataframe()
        schema = data_loader.detect_schema(
            [c for c in df.columns if c != "__split__"])
        report = data_loader.inspect_and_report(df, ds, schema)

    # ---- 2. sampling policy -------------------------------------------------
    total_records = len(df)
    use_subset = config.MAX_SAMPLES is not None and total_records > config.MAX_SAMPLES
    print("\n" + "=" * 78)
    print("SAMPLING POLICY")
    print("=" * 78)
    if use_subset:
        df = df.sample(n=config.MAX_SAMPLES, random_state=config.RANDOM_SEED) \
               .reset_index(drop=True)
        print(f"  Using a RESEARCH SUBSET: {config.MAX_SAMPLES:,} of "
              f"{total_records:,} records (seed={config.RANDOM_SEED}).")
    else:
        print(f"  Using the COMPLETE dataset: {total_records:,} records "
              f"(MAX_SAMPLES={config.MAX_SAMPLES}).")
    log["total_records"] = int(total_records)
    log["used_subset"] = bool(use_subset)
    log["records_used"] = int(len(df))

    # ---- 3. canonicalise + clean + filter -----------------------------------
    canon = pp.build_canonical_frame(df, schema)
    canon = pp.clean_and_filter_records(canon, schema, log)
    canon = pp.label_positive_interactions(canon, schema, log)
    canon = pp.deduplicate_pairs(canon, log)
    filtered = pp.filter_interactions(canon, log)

    if len(filtered) == 0:
        print("\n[ERROR] No interactions remain after filtering. "
              "Lower MIN_USER_INTERACTIONS / MIN_ITEM_INTERACTIONS in config.py.")
        # still write an (empty) statistics file for provenance
        stats_blob = {"config": config.as_dict(), "schema_report": report,
                      "pipeline_log": log, "interaction_statistics": {},
                      "provenance": _provenance()}
        with open(config.STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats_blob, f, indent=2, default=str)
        return stats_blob

    # ---- 4. encode ----------------------------------------------------------
    encoded, user2idx, item2idx = pp.encode_ids(filtered)

    # ---- 5. statistics ------------------------------------------------------
    stats = pp.compute_statistics(encoded, schema)
    pp.print_statistics(stats)

    # ---- 6. split -----------------------------------------------------------
    train, val, test = pp.per_user_split(encoded, schema, log)
    pp.print_split(log)

    # ---- 7. persist ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("WRITING ARTEFACTS")
    print("=" * 78)

    keep_cols = [c for c in ["user", "item", "user_idx", "item_idx", "rating",
                             "timestamp", "language", "review", "query",
                             "interaction", "orig_split"] if c in encoded.columns]
    written = []
    written += _save_frame(encoded[keep_cols], "interactions_full")
    written += _save_frame(train[keep_cols], "train")
    if len(val):
        written += _save_frame(val[keep_cols], "validation")
    if len(test):
        written += _save_frame(test[keep_cols], "test")

    # id mappings
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    map_paths = {
        "user2idx.json": {str(k): int(v) for k, v in user2idx.items()},
        "item2idx.json": {str(k): int(v) for k, v in item2idx.items()},
        "idx2user.json": {str(k): str(v) for k, v in idx2user.items()},
        "idx2item.json": {str(k): str(v) for k, v in idx2item.items()},
        "schema_map.json": schema.to_dict(),
    }
    for fname, obj in map_paths.items():
        p = os.path.join(config.MAPPINGS_DIR, fname)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        written.append(p)

    # statistics + provenance
    stats_blob = {
        "config": config.as_dict(),
        "schema_report": report,
        "pipeline_log": log,
        "interaction_statistics": stats,
        "split_sizes": {"train": log.get("train_size", 0),
                        "validation": log.get("val_size", 0),
                        "test": log.get("test_size", 0)},
        "provenance": _provenance(),
    }
    with open(config.STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_blob, f, indent=2, default=str)
    written.append(config.STATS_PATH)

    for p in written:
        print(f"  wrote {p}")

    _print_explanations(schema, report, stats, log)
    return stats_blob


def _provenance() -> dict:
    import platform
    prov = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import datasets
        prov["datasets"] = datasets.__version__
    except Exception:
        prov["datasets"] = None
    return prov


def _print_explanations(schema, report, stats, log) -> None:
    print("\n" + "#" * 78)
    print("# PHASE 1 SUMMARY")
    print("#" * 78)

    print(f"\n(1) ACTUAL schema detected (source={config.DATASET_SOURCE}):")
    print(f"    columns : {report['columns']}")
    print(f"    dtypes  : {report['dtypes']}")

    print("\n(2) Field role assignment (adaptation layer):")
    print(f"    user      -> {schema.user}")
    print(f"    item      -> {schema.item}")
    print(f"    review    -> {schema.review}")
    print(f"    rating    -> {schema.rating}")
    print(f"    timestamp -> {schema.timestamp}")
    print(f"    language  -> {schema.language}")
    _q = "Amazon-C4 synthetic complex query" if config.DATASET_SOURCE == "amazon_c4" else "n/a for this source"
    print(f"    query     -> {schema.query}  ({_q})")

    print("\n(3) Interaction matrix R in {0,1}^(M x N):")
    print(f"    mode = {log.get('interaction_mode')}; M={stats['num_users']:,} users, "
          f"N={stats['num_items']:,} items, {stats['num_interactions']:,} positives.")
    print(f"    density={stats['density']:.3e}, sparsity={stats['sparsity']:.10f}")

    print("\n(4) Split:")
    print(f"    {log.get('split_strategy')} per-user split -> "
          f"train={log.get('train_size',0):,}, val={log.get('val_size',0):,}, "
          f"test={log.get('test_size',0):,}")

    print("\n(5) Artefacts under results/: processed/, mappings/, "
          "dataset_statistics.json")
    print("#" * 78)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        print("\nIf this is a network error, ensure the environment can reach "
              "huggingface.co to download 'McAuley-Lab/Amazon-C4'.",
              file=sys.stderr)
        sys.exit(1)
