#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Proposed hybrid on Google Local (real data only)")
    p.add_argument("--data", default=os.path.join("data", "google_local",
                                                  "review-California.json.gz"),
                   help="path to a Google Local review .json.gz file")
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap on interactions used (default: config.MAX_SAMPLES)")
    p.add_argument("--full_dataset", action="store_true",
                   help="use the complete file (no MAX_SAMPLES cap)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU before torch import

    import config
    if args.full_dataset:
        config.MAX_SAMPLES = None
    elif args.max_samples is not None:
        config.MAX_SAMPLES = args.max_samples
    if args.epochs is not None:
        for k in ("HYBRID_EPOCHS", "SENTIMENT_EPOCHS"):
            setattr(config, k, args.epochs)

    from dataset_google import load_google_local
    # read a little more than the sample cap so filtering has room; None = all
    read_cap = None if config.MAX_SAMPLES is None else max(config.MAX_SAMPLES * 5,
                                                           config.MAX_SAMPLES)
    df = load_google_local(args.data, max_rows=read_cap)   # errors if file missing
    print(f"[google] loaded {len(df):,} real reviews from {args.data}")

    from _hybrid_pipeline import run_proposed_hybrid
    run_proposed_hybrid(df)


if __name__ == "__main__":
    main()
