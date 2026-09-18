#!/usr/bin/env python3
"""
main.py
=======
Single entry point for the Emotion-Aware Multilingual Hybrid Recommender
(LightGCN + XLM-RoBERTa) — runs the complete experimental pipeline.

Architecture
------------
    Dataset -> Preprocessing -> split
        ├── User-Item interactions ── LightGCN ─── collaborative embeddings x_u,x_i
        └── Multilingual reviews ──── XLM-RoBERTa ─ semantic h_ui
                                         + sentiment s_ui
                                         + emotion   e_ui   (pseudo-labels)
              -> projections W_c/W_t -> hybrid fusion -> fused h_u,h_i
              -> preference score ŷ = h_uᵀh_i -> Top-K ranking -> evaluation

Stages (mode=all): data → LightGCN → XLM-R features → sentiment → emotion →
emotion viz → hybrid → baseline comparison → ablation/sensitivity → final
multilingual & emotion analysis.

CLI examples
------------
    python main.py --max_samples 100000
    python main.py --max_samples 500000
    python main.py --full_dataset
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode ablation
    python main.py --mode verify           # correctness checks only (no HF download)
    python main.py --device cpu --epochs 10 --skip_transformers

Honesty
-------
No value is fabricated. Every statistic, metric, emotion label, language
distribution, and improvement is produced by actual execution of the stages
below. Emotion labels are model-derived PSEUDO-labels; language labels are
INFERRED when the dataset lacks a language field.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


# ----------------------------------------------------------------------------- #
# CLI                                                                            #
# ----------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Emotion-Aware Multilingual Hybrid Recommender — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", default="all",
                   choices=["all", "train", "evaluate", "ablation", "final", "verify"],
                   help="which part of the pipeline to run")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--max_samples", type=int, default=None,
                   help="cap on interactions used (overrides config.MAX_SAMPLES)")
    g.add_argument("--full_dataset", action="store_true",
                   help="use the complete dataset (MAX_SAMPLES=None)")
    p.add_argument("--dataset_source", default=None,
                   choices=["google_local", "amazon_c4"])
    p.add_argument("--epochs", type=int, default=None,
                   help="override training epochs for all trainable stages")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--strict", action="store_true",
                   help="abort at the first failing stage (default: continue)")
    p.add_argument("--skip_transformers", action="store_true",
                   help="skip XLM-R/sentiment/emotion stages and everything that "
                        "needs review features (use when no network/HF access)")
    return p.parse_args()


def apply_overrides(config, args):
    if args.full_dataset:
        config.MAX_SAMPLES = None
    elif args.max_samples is not None:
        config.MAX_SAMPLES = args.max_samples
    if args.dataset_source:
        config.DATASET_SOURCE = args.dataset_source
    if args.seed is not None:
        config.RANDOM_SEED = args.seed
    if args.epochs is not None:
        for k in ("EPOCHS", "HYBRID_EPOCHS", "SENTIMENT_EPOCHS", "COMPARE_EPOCHS"):
            if hasattr(config, k):
                setattr(config, k, args.epochs)


# ----------------------------------------------------------------------------- #
# Stage definitions: (name, fn, needs_transformers)                             #
# ----------------------------------------------------------------------------- #
def _stage_data():
    import phase1_dataset
    phase1_dataset.run()


def _stage_lightgcn():
    import lightgcn_train
    lightgcn_train.train()


def _stage_xlmr():
    import review_embeddings
    review_embeddings.extract_and_cache()


def _stage_sentiment():
    import xlmr_sentiment
    xlmr_sentiment.run()


def _stage_emotion():
    import emotion_features, emotion_representation
    emotion_features.extract_and_cache()
    emotion_representation.build_representations()


def _stage_emotion_viz():
    import emotion_evaluation
    emotion_evaluation.run()


def _stage_hybrid():
    import hybrid_train
    hybrid_train.train()


def _stage_baselines():
    import run_comparison
    run_comparison.run()


def _stage_ablation():
    import ablation
    ablation.run()


def _stage_final():
    import final_analysis
    final_analysis.run()


# name -> (fn, needs_transformers)
REGISTRY = {
    "data":         (_stage_data, False),
    "lightgcn":     (_stage_lightgcn, False),
    "xlmr":         (_stage_xlmr, True),
    "sentiment":    (_stage_sentiment, True),
    "emotion":      (_stage_emotion, True),
    "emotion_viz":  (_stage_emotion_viz, True),
    "hybrid":       (_stage_hybrid, True),
    "baselines":    (_stage_baselines, True),
    "ablation":     (_stage_ablation, True),
    "final":        (_stage_final, True),
}

MODE_PLANS = {
    "all":      ["data", "lightgcn", "xlmr", "sentiment", "emotion",
                 "emotion_viz", "hybrid", "baselines", "ablation", "final"],
    "train":    ["data", "lightgcn", "xlmr", "sentiment", "emotion", "hybrid"],
    "evaluate": ["baselines", "final"],
    "ablation": ["ablation"],
    "final":    ["final"],
}


# ----------------------------------------------------------------------------- #
# Runner                                                                         #
# ----------------------------------------------------------------------------- #
def run_plan(plan, strict, skip_transformers):
    results = []
    for name in plan:
        fn, needs_tf = REGISTRY[name]
        if skip_transformers and needs_tf:
            print(f"\n===== SKIP stage '{name}' (--skip_transformers) =====")
            results.append((name, "skipped", 0.0))
            continue
        print(f"\n{'='*70}\n>>> STAGE: {name}\n{'='*70}")
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            results.append((name, "ok", dt))
            print(f"--- stage '{name}' done in {dt:.1f}s ---")
        except Exception as exc:
            dt = time.time() - t0
            results.append((name, f"FAILED: {exc}", dt))
            print(f"!!! stage '{name}' FAILED after {dt:.1f}s: {exc}", file=sys.stderr)
            traceback.print_exc()
            if strict:
                break
    return results


def print_summary(results):
    print("\n" + "#" * 70)
    print("# PIPELINE SUMMARY")
    print("#" * 70)
    for name, status, dt in results:
        tag = "OK   " if status == "ok" else ("SKIP " if status == "skipped" else "FAIL ")
        print(f"  [{tag}] {name:<14} {dt:6.1f}s   {'' if status in ('ok','skipped') else status}")
    n_fail = sum(1 for _, s, _ in results if s not in ("ok", "skipped"))
    print("#" * 70)
    print(f"# {len(results)} stages, {n_fail} failed. All reported numbers come "
          f"from actual execution.")
    print("#" * 70)
    return n_fail


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU before torch init

    import config
    apply_overrides(config, args)

    print("#" * 70)
    print("# Emotion-Aware Multilingual Hybrid Recommender (LightGCN + XLM-RoBERTa)")
    print(f"# mode={args.mode}  dataset_source={config.DATASET_SOURCE}  "
          f"max_samples={config.MAX_SAMPLES}  seed={config.RANDOM_SEED}")
    print("# Emotion = model-derived pseudo-labels; language = inferred when no "
          "field exists. No values are fabricated.")
    print("#" * 70)

    if args.mode == "verify":
        import verify_pipeline
        return verify_pipeline.main()

    plan = MODE_PLANS[args.mode]
    results = run_plan(plan, args.strict, args.skip_transformers)
    n_fail = print_summary(results)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
