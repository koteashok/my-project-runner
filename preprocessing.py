

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_loader import SchemaMap



_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NOISE_RES = [re.compile(p) for p in config.NOISE_TOKEN_PATTERNS]


def clean_text(value) -> str:

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = html.unescape(text)               # &amp; -> &, &#39; -> '
    text = _HTML_TAG_RE.sub(" ", text)       # <br /> and friends
    for rex in _NOISE_RES:                    # [[VIDEOID:...]] etc.
        text = rex.sub(" ", text)
    # Unicode normalisation preserves language/emoji, normalises width/compat forms.
    text = unicodedata.normalize(config.UNICODE_NORMALIZATION_FORM, text)
    text = _WS_RE.sub(" ", text).strip()
    if config.LOWERCASE_TEXT:
        text = text.lower()
    return text



def build_canonical_frame(df: pd.DataFrame, schema: SchemaMap) -> pd.DataFrame:

    out = pd.DataFrame(index=df.index)
    out["user"] = df[schema.user].astype("string") if schema.has("user") else pd.NA
    out["item"] = df[schema.item].astype("string") if schema.has("item") else pd.NA

    if schema.has("review"):
        out["review_raw"] = df[schema.review]
    if schema.has("query"):
        out["query"] = df[schema.query]
    if schema.has("rating"):
        out["rating"] = pd.to_numeric(df[schema.rating], errors="coerce")
    if schema.has("timestamp"):
        out["timestamp"] = pd.to_numeric(df[schema.timestamp], errors="coerce")
    if schema.has("language"):
        out["language"] = df[schema.language].astype("string")
    if "__split__" in df.columns:
        out["orig_split"] = df["__split__"]
    return out



def clean_and_filter_records(frame: pd.DataFrame, schema: SchemaMap,
                             log: dict) -> pd.DataFrame:

    n0 = len(frame)

    # --- essential-key missing-value handling: user & item are mandatory ------
    frame = frame.copy()
    for key in ("user", "item"):
        frame[key] = frame[key].astype("string").str.strip()
        frame.loc[frame[key].isin(["", "nan", "None", "<NA>"]), key] = pd.NA
    frame = frame.dropna(subset=["user", "item"])
    log["dropped_missing_keys"] = int(n0 - len(frame))

    # --- review cleaning ------------------------------------------------------
    if "review_raw" in frame.columns:
        frame["review"] = frame["review_raw"].map(clean_text)
        frame = frame.drop(columns=["review_raw"])
    else:
        frame["review"] = ""
    if "query" in frame.columns:
        frame["query"] = frame["query"].map(clean_text)

    # --- rating processing ----------------------------------------------------
    if "rating" in frame.columns:
        # keep numeric ratings only; leave NaN for rows without a parseable score
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")

    # --- timestamp processing -------------------------------------------------
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")

    # --- language processing --------------------------------------------------
    if "language" in frame.columns:
        frame["language"] = (frame["language"].astype("string")
                             .str.strip().str.lower().replace({"": pd.NA}))

    # --- exact duplicate removal ---------------------------------------------
    before = len(frame)
    subset = ["user", "item"]
    if "review" in frame.columns:
        subset_full = ["user", "item", "review"]
    else:
        subset_full = subset
    frame = frame.drop_duplicates(subset=subset_full, keep="first")
    log["dropped_exact_duplicates"] = int(before - len(frame))

    log["records_after_cleaning"] = int(len(frame))
    return frame.reset_index(drop=True)



def resolve_interaction_mode(schema: SchemaMap) -> str:
    mode = config.INTERACTION_MODE
    if mode == "auto":
        return "rating" if schema.has("rating") else "implicit"
    if mode == "rating" and not schema.has("rating"):
        return "implicit"
    return mode


def label_positive_interactions(frame: pd.DataFrame, schema: SchemaMap,
                                log: dict) -> pd.DataFrame:
    """
    Keep only positive interactions (R_ui = 1).

    - mode "rating"  : rating >= RATING_THRESHOLD
    - mode "implicit": a non-empty review exists
    """
    mode = resolve_interaction_mode(schema)
    log["interaction_mode"] = mode
    n0 = len(frame)

    if mode == "rating":
        mask = frame["rating"].notna() & (frame["rating"] >= config.RATING_THRESHOLD)
        log["rating_threshold"] = config.RATING_THRESHOLD
    else:  # implicit
        mask = frame["review"].astype("string").str.len() > 0

    frame = frame[mask].copy()
    frame["interaction"] = 1
    log["dropped_non_positive"] = int(n0 - len(frame))
    log["positive_interactions_before_kcore"] = int(len(frame))
    return frame.reset_index(drop=True)



def deduplicate_pairs(frame: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Collapse repeated (user, item) pairs to a single binary interaction."""
    if not config.DEDUPLICATE_INTERACTIONS:
        return frame
    before = len(frame)

    if config.REVIEW_KEEP_STRATEGY == "longest" and "review" in frame.columns:
        frame = frame.assign(_len=frame["review"].astype("string").str.len().fillna(0))
        frame = (frame.sort_values("_len", ascending=False)
                      .drop_duplicates(subset=["user", "item"], keep="first")
                      .drop(columns="_len"))
    elif config.REVIEW_KEEP_STRATEGY == "last":
        frame = frame.drop_duplicates(subset=["user", "item"], keep="last")
    else:  # "first"
        frame = frame.drop_duplicates(subset=["user", "item"], keep="first")

    log["dropped_duplicate_pairs"] = int(before - len(frame))
    return frame.reset_index(drop=True)



def kcore_filter(frame: pd.DataFrame, min_user: int, min_item: int,
                 log: dict) -> pd.DataFrame:
    """
    Iteratively drop users with < min_user interactions and items with
    < min_item interactions until the frame is stable (a k-core).
    """
    prev = -1
    iterations = 0
    cur = frame
    while len(cur) != prev and len(cur) > 0:
        prev = len(cur)
        iterations += 1
        uc = cur["user"].value_counts()
        keep_users = uc[uc >= min_user].index
        cur = cur[cur["user"].isin(keep_users)]
        ic = cur["item"].value_counts()
        keep_items = ic[ic >= min_item].index
        cur = cur[cur["item"].isin(keep_items)]
    log["kcore_iterations"] = iterations
    log["records_after_kcore"] = int(len(cur))
    return cur.reset_index(drop=True)


def filter_interactions(frame: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    Apply the configured k-core. If it empties the set and RELAX_FILTER_IF_EMPTY
    is on, retry with RELAXED_MIN_INTERACTIONS (reported loudly).
    """
    filtered = kcore_filter(frame, config.MIN_USER_INTERACTIONS,
                            config.MIN_ITEM_INTERACTIONS, log)
    log["kcore_min_user"] = config.MIN_USER_INTERACTIONS
    log["kcore_min_item"] = config.MIN_ITEM_INTERACTIONS
    log["kcore_relaxed"] = False

    if len(filtered) == 0 and config.RELAX_FILTER_IF_EMPTY:
        r = config.RELAXED_MIN_INTERACTIONS
        print("\n" + "!" * 78)
        print("WARNING: the {u}-core / {i}-core filter removed ALL interactions."
              .format(u=config.MIN_USER_INTERACTIONS,
                      i=config.MIN_ITEM_INTERACTIONS))
        print("This is expected for Amazon-C4: it is a complex-product-search")
        print("EVALUATION set built by uniformly sampling ~22k individual user")
        print("reviews, so almost every user appears only once and cannot")
        print("survive a 5-core. Retrying with a relaxed threshold of "
              f"min_user=min_item={r}.")
        print("Adjust MIN_USER_INTERACTIONS / MIN_ITEM_INTERACTIONS in config.py,")
        print("or switch to the full Amazon-Reviews-2023 dataset in a later phase")
        print("if a denser collaborative signal is required.")
        print("!" * 78)
        relaxed_log = {}
        filtered = kcore_filter(frame, r, r, relaxed_log)
        log["kcore_relaxed"] = True
        log["kcore_relaxed_min"] = r
        log["kcore_iterations_relaxed"] = relaxed_log.get("kcore_iterations")
        log["records_after_kcore"] = int(len(filtered))

    return filtered



def encode_ids(frame: pd.DataFrame):

    users = sorted(frame["user"].unique().tolist())
    items = sorted(frame["item"].unique().tolist())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: j for j, it in enumerate(items)}
    frame = frame.copy()
    frame["user_idx"] = frame["user"].map(user2idx).astype("int64")
    frame["item_idx"] = frame["item"].map(item2idx).astype("int64")
    return frame, user2idx, item2idx



def compute_statistics(frame: pd.DataFrame, schema: SchemaMap) -> dict:
    n_users = int(frame["user"].nunique())
    n_items = int(frame["item"].nunique())
    n_inter = int(len(frame))
    possible = n_users * n_items

    stats = {
        "num_users": n_users,
        "num_items": n_items,
        "num_interactions": n_inter,
        "avg_interactions_per_user": round(n_inter / n_users, 6) if n_users else 0.0,
        "avg_interactions_per_item": round(n_inter / n_items, 6) if n_items else 0.0,
        "density": round(n_inter / possible, 10) if possible else 0.0,
        "sparsity": round(1.0 - (n_inter / possible), 10) if possible else 0.0,
        "num_reviews": int((frame["review"].astype("string").str.len() > 0).sum())
                       if "review" in frame.columns else 0,
    }

    if "rating" in frame.columns and frame["rating"].notna().any():
        dist = (frame["rating"].dropna().round(1)
                .value_counts().sort_index())
        stats["rating_distribution"] = {str(k): int(v) for k, v in dist.items()}
        stats["rating_min"] = float(frame["rating"].min())
        stats["rating_max"] = float(frame["rating"].max())
        stats["rating_mean"] = round(float(frame["rating"].mean()), 6)
    else:
        stats["rating_distribution"] = None

    if "language" in frame.columns and frame["language"].notna().any():
        lang = frame["language"].dropna().value_counts()
        stats["language_distribution"] = {str(k): int(v) for k, v in lang.items()}
    else:
        stats["language_distribution"] = None

    return stats


def print_statistics(stats: dict) -> None:
    print("\n" + "=" * 78)
    print("INTERACTION STATISTICS")
    print("=" * 78)
    print(f"  users ............................ {stats['num_users']:,}")
    print(f"  items ............................ {stats['num_items']:,}")
    print(f"  interactions ..................... {stats['num_interactions']:,}")
    print(f"  avg interactions / user .......... {stats['avg_interactions_per_user']}")
    print(f"  avg interactions / item .......... {stats['avg_interactions_per_item']}")
    print(f"  density .......................... {stats['density']:.3e}")
    print(f"  sparsity ......................... {stats['sparsity']:.10f}")
    print(f"  reviews (non-empty) .............. {stats['num_reviews']:,}")
    if stats.get("rating_distribution"):
        print("  rating distribution:")
        for k, v in stats["rating_distribution"].items():
            print(f"      rating {k}: {v:,}")
        print(f"  rating mean/min/max .............. "
              f"{stats.get('rating_mean')}/{stats.get('rating_min')}/{stats.get('rating_max')}")
    else:
        print("  rating distribution .............. (no rating column)")
    if stats.get("language_distribution"):
        print("  language distribution:")
        for k, v in list(stats["language_distribution"].items())[:20]:
            print(f"      {k}: {v:,}")


def per_user_split(frame: pd.DataFrame, schema: SchemaMap, log: dict):

    use_time = schema.has("timestamp") and "timestamp" in frame.columns \
        and frame["timestamp"].notna().any()
    log["split_strategy"] = "chronological" if use_time else "per_user_random"

    rng = np.random.RandomState(config.RANDOM_SEED)
    train_parts, val_parts, test_parts = [], [], []
    min_train = config.MIN_TRAIN_INTERACTIONS_FOR_EVAL

    for _user, grp in frame.groupby("user", sort=False):
        g = grp
        if use_time:
            g = g.sort_values("timestamp", kind="mergesort")
        else:
            order = rng.permutation(len(g))
            g = g.iloc[order]

        n = len(g)
        if n == 1:
            train_parts.append(g)
            continue
        if n == 2:
            train_parts.append(g.iloc[:1])
            if len(g.iloc[:1]) >= min_train:
                test_parts.append(g.iloc[1:2])
            else:
                train_parts.append(g.iloc[1:2])
            continue
        # n >= 3
        train_g = g.iloc[:-2]
        val_g = g.iloc[-2:-1]
        test_g = g.iloc[-1:]
        if len(train_g) >= min_train:
            train_parts.append(train_g)
            if config.VALIDATION_ENABLED:
                val_parts.append(val_g)
            else:
                train_parts.append(val_g)
            test_parts.append(test_g)
        else:
            train_parts.append(g)  # not enough train history to evaluate

    def _cat(parts):
        return (pd.concat(parts, ignore_index=True)
                if parts else frame.iloc[0:0].copy())

    train = _cat(train_parts)
    val = _cat(val_parts)
    test = _cat(test_parts)

    # cold-item guard: drop eval interactions whose item never appears in train
    if config.DROP_COLD_ITEMS_IN_EVAL and len(train) > 0:
        train_items = set(train["item"].unique())
        train_users = set(train["user"].unique())
        for name, part in (("val", val), ("test", test)):
            if len(part) == 0:
                continue
            before = len(part)
            part = part[part["item"].isin(train_items) & part["user"].isin(train_users)]
            log[f"dropped_cold_{name}"] = int(before - len(part))
            if name == "val":
                val = part
            else:
                test = part

    log["train_size"] = int(len(train))
    log["val_size"] = int(len(val))
    log["test_size"] = int(len(test))
    log["users_in_train"] = int(train["user"].nunique()) if len(train) else 0
    log["users_in_val"] = int(val["user"].nunique()) if len(val) else 0
    log["users_in_test"] = int(test["user"].nunique()) if len(test) else 0
    return train, val, test


def print_split(log: dict) -> None:
    print("\n" + "=" * 78)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 78)
    print(f"  strategy ......................... {log.get('split_strategy')}")
    print(f"  train interactions ............... {log.get('train_size', 0):,} "
          f"({log.get('users_in_train', 0):,} users)")
    print(f"  val   interactions ............... {log.get('val_size', 0):,} "
          f"({log.get('users_in_val', 0):,} users)")
    print(f"  test  interactions ............... {log.get('test_size', 0):,} "
          f"({log.get('users_in_test', 0):,} users)")
    if log.get("dropped_cold_test") or log.get("dropped_cold_val"):
        print(f"  dropped cold-item val ............ {log.get('dropped_cold_val', 0):,}")
        print(f"  dropped cold-item test ........... {log.get('dropped_cold_test', 0):,}")
    if log.get("val_size", 0) == 0 and log.get("test_size", 0) == 0:
        print("\n  NOTE: no val/test examples were produced. With Amazon-C4 most")
        print("  users have a single interaction, so leave-one-out yields only a")
        print("  training set. Lower the k-core thresholds or use a denser source")
        print("  to obtain evaluation splits.")
