from __future__ import annotations

import ast
import gzip
import json
import os

import pandas as pd


def _parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)      # single-quoted dict (2018 dumps)
        except Exception:
            return None


def load_google_local(path: str, max_rows: int | None = None) -> pd.DataFrame:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Google Local dataset file not found.\n"
            f"  expected at: {os.path.abspath(path)}\n"
            "  Download a per-state review file (e.g. review-California.json.gz) from\n"
            "  https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/\n"
            "  and place it there, or pass --data <path-to-file>."
        )

    opener = (gzip.open(path, "rt", encoding="utf-8", errors="replace")
              if path.lower().endswith(".gz")
              else open(path, "r", encoding="utf-8", errors="replace"))
    rows = []
    with opener as f:
        for line in f:
            if max_rows is not None and len(rows) >= max_rows:
                break
            rec = _parse_line(line)
            if isinstance(rec, dict):
                rec.pop("pics", None)
                rec.pop("resp", None)
                rows.append(rec)

    if not rows:
        raise ValueError(
            f"No records parsed from {path}. Is this a Google Local review "
            f"JSON-lines (.json.gz) file?")
    return pd.DataFrame(rows)
