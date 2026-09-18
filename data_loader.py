
from __future__ import annotations

import ast
import gzip
import json
import sys
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd

import config



ROLE_CANDIDATES: dict[str, list[str]] = {
    "user": ["user_id", "gplususerid", "userid", "reviewerid", "reviewer_id",
             "uid", "user"],
    "item": ["item_id", "gmap_id", "gplusplaceid", "parent_asin", "asin",
             "product_id", "productid", "business_id", "itemid", "pid", "item"],
    "review": ["ori_review", "reviewtext", "review_text", "review_body",
               "reviewbody", "review", "text", "content", "body"],
    "rating": ["ori_rating", "rating", "overall", "stars", "star_rating", "score"],
    "timestamp": ["timestamp", "unixreviewtime", "time", "review_time",
                  "reviewtime", "date", "ts", "unix_time"],
    "language": ["language", "lang", "language_code", "lang_code", "locale"],
    "query": ["query", "complex_query", "instruction", "context"],
}
OPTIONAL_ROLES = {"rating", "timestamp", "language", "query", "review"}


@dataclass
class SchemaMap:
    user: Optional[str] = None
    item: Optional[str] = None
    review: Optional[str] = None
    rating: Optional[str] = None
    timestamp: Optional[str] = None
    language: Optional[str] = None
    query: Optional[str] = None
    all_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def has(self, role: str) -> bool:
        return getattr(self, role, None) is not None


def detect_schema(columns) -> SchemaMap:

    columns = list(columns)
    lower_to_real = {c.lower(): c for c in columns}
    resolved: dict[str, Optional[str]] = {}

    def match_role(candidates: list[str]) -> Optional[str]:
        for cand in candidates:                       # exact case-insensitive
            if cand.lower() in lower_to_real:
                return lower_to_real[cand.lower()]
        for cand in candidates:                       # conservative substring
            for real_lower, real in lower_to_real.items():
                if len(cand) >= 4 and cand.lower() in real_lower:
                    return real
        return None

    for role, cands in ROLE_CANDIDATES.items():
        resolved[role] = match_role(cands)

    if resolved.get("query") and resolved.get("query") == resolved.get("review"):
        resolved["query"] = None
    return SchemaMap(all_columns=columns, **resolved)



def _parse_json_line(raw: str):

    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None


def _open_gzip_lines(path: Optional[str], url: Optional[str]):

    if path:
        fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        try:
            for line in fh:
                yield line
        finally:
            fh.close()
        return
    if url:
        req = urllib.request.Request(url, headers={"User-Agent": "phase1-loader"})
        resp = urllib.request.urlopen(req, timeout=config.NETWORK_TIMEOUT_SECONDS)
        gz = gzip.GzipFile(fileobj=resp)
        try:
            for bline in gz:
                yield bline.decode("utf-8", errors="replace")
        finally:
            gz.close()
            resp.close()
        return
    raise ValueError("No Google Local review path or URL provided.")


def _load_google_local_category_allowset():

    cat = config.GOOGLE_LOCAL_CATEGORY
    if not cat:
        return None
    meta_path = config.GOOGLE_LOCAL_META_PATH
    meta_url = config.GOOGLE_LOCAL_META_URL
    if not meta_path and not meta_url and config.GOOGLE_LOCAL_VERSION == "2021":
        state = config.GOOGLE_LOCAL_STATE.strip().replace(" ", "_")
        base = config.GOOGLE_LOCAL_BASE_URL.rstrip("/")
        meta_url = f"{base}/meta-{state}.json.gz"
    allow = set()
    want = cat.strip().lower()
    n = 0
    for line in _open_gzip_lines(meta_path, meta_url):
        rec = _parse_json_line(line)
        n += 1
        if n > config.GOOGLE_LOCAL_META_MAX_ROWS:
            break
        if not rec:
            continue
        gid = rec.get("gmap_id")
        cats = rec.get("category") or []
        if isinstance(cats, str):
            cats = [cats]
        if gid and any(want in str(c).lower() for c in cats):
            allow.add(gid)
    print(f"  category filter '{cat}': {len(allow):,} matching businesses")
    return allow


def load_google_local():

    path = config.GOOGLE_LOCAL_REVIEW_PATH
    url = config.resolve_google_local_review_url()
    if not path and not url:
        raise RuntimeError(
            "Google Local 2018/global selected but no review file specified.\n"
            "Set GOOGLE_LOCAL_REVIEW_URL or GOOGLE_LOCAL_REVIEW_PATH in config.py.")

    src_desc = path if path else url
    print(f"  streaming Google Local reviews from: {src_desc}")
    print(f"  version={config.GOOGLE_LOCAL_VERSION}  "
          f"state={config.GOOGLE_LOCAL_STATE}  "
          f"raw_cap={config.GOOGLE_LOCAL_MAX_RAW_ROWS:,}")

    allow = _load_google_local_category_allowset()

    rows = []
    cap = config.GOOGLE_LOCAL_MAX_RAW_ROWS
    read = 0
    kept = 0
    try:
        for line in _open_gzip_lines(path, url):
            read += 1
            rec = _parse_json_line(line)
            if not rec or not isinstance(rec, dict):
                continue
            if allow is not None:
                gid = rec.get("gmap_id") or rec.get("gPlusPlaceId")
                if gid not in allow:
                    continue
            # drop bulky/nested fields we do not need in Phase 1
            rec.pop("pics", None)
            rec.pop("resp", None)
            rows.append(rec)
            kept += 1
            if cap and kept >= cap:
                break
    except Exception as exc:  # pragma: no cover - network dependent
        if not rows:
            raise RuntimeError(
                f"Failed to stream Google Local reviews from {src_desc}. "
                f"Ensure the URL/path is correct and reachable.\n    {exc}") from exc
        print(f"  [warn] stream ended early after {kept:,} rows: {exc}")

    df = pd.DataFrame(rows)
    df["__split__"] = f"google_local_{config.GOOGLE_LOCAL_VERSION}_{config.GOOGLE_LOCAL_STATE}"
    print(f"  parsed {kept:,} review records (scanned {read:,} lines).")
    return df



def load_amazon_c4():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install datasets") from exc
    try:
        ds = (load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
              if config.DATASET_CONFIG else load_dataset(config.DATASET_NAME))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to download '{config.DATASET_NAME}' (needs huggingface.co). "
            f"\n    {exc}") from exc
    frames = []
    for split_name in ds.keys():
        part = ds[split_name].to_pandas()
        part["__split__"] = split_name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return df, ds


def load_raw_dataframe():
    """Return (df, ds_or_None). ds is a HuggingFace DatasetDict for Amazon-C4,
    or None for Google Local (no HF object)."""
    if config.DATASET_SOURCE == "amazon_c4":
        df, ds = load_amazon_c4()
        return df, ds
    elif config.DATASET_SOURCE == "google_local":
        df = load_google_local()
        return df, None
    raise ValueError(f"Unknown DATASET_SOURCE: {config.DATASET_SOURCE}")



def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def inspect_and_report(df: pd.DataFrame, ds, schema: SchemaMap) -> dict:
    _print_header("1. DATASET SPLITS & RECORD COUNTS")
    split_counts = {}
    try:
        for split_name in ds.keys():
            n = ds[split_name].num_rows
            split_counts[split_name] = int(n)
            print(f"  split '{split_name}': {n:,} records")
    except Exception:
        for split_name, sub in df.groupby("__split__"):
            split_counts[str(split_name)] = int(len(sub))
            print(f"  split '{split_name}': {len(sub):,} records")
    print(f"  TOTAL (combined working frame): {len(df):,} records")

    _print_header("2. ACTUAL COLUMN NAMES")
    real_columns = [c for c in df.columns if c != "__split__"]
    print(f"  {real_columns}")

    _print_header("3. FEATURE / DATA TYPES")
    try:
        first_split = next(iter(ds.keys()))
        hf_features = ds[first_split].features
        print("  HuggingFace feature schema:")
        for name, feat in hf_features.items():
            print(f"    - {name}: {feat}")
    except Exception:
        print("  (no HuggingFace feature schema for this source)")
    print("  pandas dtypes:")
    for name in real_columns:
        print(f"    - {name}: {df[name].dtype}")

    _print_header(f"4. {config.N_SAMPLE_RECORDS} SAMPLE RECORDS")
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(df[real_columns].head(config.N_SAMPLE_RECORDS).to_string())

    _print_header("5. MISSING-VALUE STATISTICS")
    missing = {}
    n = len(df)
    for name in real_columns:
        n_null = int(df[name].isna().sum())
        n_empty = 0
        if df[name].dtype == object:
            s = df[name].astype("string")
            n_empty = max(int((s.fillna("").str.strip() == "").sum()) - n_null, 0)
        pct = (n_null / n * 100.0) if n else 0.0
        missing[name] = {"nulls": n_null, "empty_strings": n_empty,
                         "pct_null": round(pct, 4)}
        print(f"  {name:<16} nulls={n_null:<8} empty_str={n_empty:<8} "
              f"({pct:.3f}% null)")

    _print_header("6. AVAILABLE METADATA")
    meta = {}
    try:
        info = ds[next(iter(ds.keys()))].info
        meta["description"] = (info.description or "").strip()[:500]
        meta["citation"] = (info.citation or "").strip()[:300]
        for k, v in meta.items():
            print(f"  {k}: {v if v else '(none)'}")
    except Exception:
        meta = {"source": config.DATASET_SOURCE,
                "google_local_version": config.GOOGLE_LOCAL_VERSION
                if config.DATASET_SOURCE == "google_local" else None}
        print(f"  source: {config.DATASET_SOURCE}")
        if config.DATASET_SOURCE == "google_local":
            print(f"  google_local_version: {config.GOOGLE_LOCAL_VERSION}")
            print(f"  region/state: {config.GOOGLE_LOCAL_STATE}")

    _print_header("7. FIELD PRESENCE (schema-adaptation result)")
    presence = {
        "user_id_exists":   schema.has("user"),
        "item_id_exists":   schema.has("item"),
        "review_exists":    schema.has("review"),
        "rating_exists":    schema.has("rating"),
        "timestamp_exists": schema.has("timestamp"),
        "language_exists":  schema.has("language"),
        "query_exists":     schema.has("query"),
    }
    print(f"  user id exists?         {presence['user_id_exists']!s:<6} -> {schema.user}")
    print(f"  item / product id?      {presence['item_id_exists']!s:<6} -> {schema.item}")
    print(f"  review text exists?     {presence['review_exists']!s:<6} -> {schema.review}")
    print(f"  rating exists?          {presence['rating_exists']!s:<6} -> {schema.rating}")
    print(f"  timestamp exists?       {presence['timestamp_exists']!s:<6} -> {schema.timestamp}")
    print(f"  language exists?        {presence['language_exists']!s:<6} -> {schema.language}")
    print(f"  (extra) query exists?   {presence['query_exists']!s:<6} -> {schema.query}")

    if schema.has("timestamp"):
        print("\n  NOTE: timestamp present -> CHRONOLOGICAL per-user split will be used.")
    else:
        print("\n  NOTE: no timestamp -> reproducible per-user RANDOM split will be used.")
    if not schema.has("language"):
        print("  NOTE: no language column -> language treated as unavailable "
              "(later phase can language-detect from review text).")

    return {
        "split_counts": split_counts,
        "combined_records": int(len(df)),
        "columns": real_columns,
        "dtypes": {c: str(df[c].dtype) for c in real_columns},
        "missing": missing,
        "metadata": meta,
        "field_presence": presence,
        "schema_map": schema.to_dict(),
        "dataset_source": config.DATASET_SOURCE,
    }


def load_and_inspect():
    df, ds = load_raw_dataframe()
    schema = detect_schema([c for c in df.columns if c != "__split__"])
    report = inspect_and_report(df, ds, schema)
    return df, schema, report


if __name__ == "__main__":  # pragma: no cover
    try:
        load_and_inspect()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
