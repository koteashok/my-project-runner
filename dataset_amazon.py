
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
            return ast.literal_eval(line)
        except Exception:
            return None


def _read_jsonish(path: str, max_rows: int | None):
    opener = (gzip.open(path, "rt", encoding="utf-8", errors="replace")
              if path.lower().endswith(".gz")
              else open(path, "r", encoding="utf-8", errors="replace"))
    rows = []
    with opener as f:
        head = f.read(2048).lstrip()
        f.seek(0)
        if head[:1] == "[":                       # a single JSON array
            data = json.load(f)
            data = data if isinstance(data, list) else [data]
            rows = data[:max_rows] if max_rows is not None else data
        else:                                     # JSON-lines
            for line in f:
                if max_rows is not None and len(rows) >= max_rows:
                    break
                rec = _parse_line(line)
                if isinstance(rec, dict):
                    rows.append(rec)
    return rows


def load_amazon(path: str, max_rows: int | None = None) -> pd.DataFrame:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Amazon dataset file not found.\n"
            f"  expected at: {os.path.abspath(path)}\n"
            "  Place a real Amazon review file there (Amazon-C4 parquet/jsonl, or an\n"
            "  Amazon Reviews .jsonl.gz with user / item / rating / review columns),\n"
            "  or pass --data <path-to-file>."
        )

    ext = path.lower()
    if ext.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.head(max_rows) if max_rows is not None else df
    if ext.endswith(".csv"):
        return pd.read_csv(path, nrows=max_rows)
    if ext.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz", ".gz")):
        rows = _read_jsonish(path, max_rows)
        if not rows:
            raise ValueError(f"No records parsed from {path}.")
        return pd.DataFrame(rows)

    raise ValueError(
        f"Unsupported Amazon file type: {path}. Use .parquet, .csv, .json(.gz) "
        f"or .jsonl(.gz).")
