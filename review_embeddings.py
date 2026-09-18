
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

import config
from xlmr_model import build_encoder, load_tokenizer



def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _read_split(basename: str):
    import pandas as pd
    p_parq = os.path.join(config.PROCESSED_DIR, basename + ".parquet")
    p_csv = os.path.join(config.PROCESSED_DIR, basename + ".csv")
    if os.path.exists(p_parq):
        return pd.read_parquet(p_parq)
    if os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    return None


def load_splits():
    train = _read_split("train")
    if train is None:
        raise FileNotFoundError(
            f"No processed splits under {config.PROCESSED_DIR}. Run Phase 1 first.")
    return train, _read_split("validation"), _read_split("test")


def _reviews(df) -> List[str]:
    if df is None:
        return []
    if "review" in df.columns:
        return df["review"].fillna("").astype(str).tolist()
    return ["" for _ in range(len(df))]



@torch.no_grad()
def extract_embeddings(encoder,
                       tokenizer,
                       texts: List[str],
                       device: torch.device,
                       batch_size: int = 32,
                       max_length: int = 256,
                       use_amp: bool = False,
                       fp16_store: bool = True) -> torch.Tensor:
    """
    Tokenise (dynamic padding per batch, truncation to `max_length`) and return a
    (len(texts) × H) embedding tensor. Runs entirely under no_grad.
    """
    encoder.eval()
    out_chunks = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pooled = encoder.encode(input_ids, attention_mask, use_amp=use_amp)  # (B,H)
        pooled = pooled.detach().cpu()
        if fp16_store:
            pooled = pooled.half()
        out_chunks.append(pooled)
    if not out_chunks:
        H = encoder.hidden_size or config.XLMR_HIDDEN
        return torch.empty(0, H)
    return torch.cat(out_chunks, dim=0)



def aggregate_user_item(embeddings: torch.Tensor,
                        user_idx: np.ndarray,
                        item_idx: np.ndarray,
                        num_users: int,
                        num_items: int) -> Dict[str, torch.Tensor]:
    """
    z_u = mean over the user's rows; z_i = mean over the item's rows.

    IMPORTANT: pass ONLY training rows here. Users/items with no training review
    get a zero vector (and count 0) — callers can detect these via the counts.
    """
    emb = embeddings.float()
    H = emb.shape[1]
    z_u = torch.zeros(num_users, H)
    z_i = torch.zeros(num_items, H)
    cu = torch.zeros(num_users, 1)
    ci = torch.zeros(num_items, 1)

    ut = torch.from_numpy(np.asarray(user_idx, dtype=np.int64))
    it = torch.from_numpy(np.asarray(item_idx, dtype=np.int64))
    z_u.index_add_(0, ut, emb)
    z_i.index_add_(0, it, emb)
    cu.index_add_(0, ut, torch.ones(len(ut), 1))
    ci.index_add_(0, it, torch.ones(len(it), 1))

    z_u = z_u / cu.clamp(min=1.0)
    z_i = z_i / ci.clamp(min=1.0)
    return {"z_user": z_u, "z_item": z_i,
            "user_count": cu.squeeze(1), "item_count": ci.squeeze(1)}



def _infer_sizes(train, val, test):
    import json
    um = os.path.join(config.MAPPINGS_DIR, "user2idx.json")
    im = os.path.join(config.MAPPINGS_DIR, "item2idx.json")
    if os.path.exists(um) and os.path.exists(im):
        with open(um) as f:
            M = len(json.load(f))
        with open(im) as f:
            N = len(json.load(f))
        return M, N
    mu = mi = -1
    for df in (train, val, test):
        if df is not None and len(df):
            mu = max(mu, int(df["user_idx"].max()))
            mi = max(mi, int(df["item_idx"].max()))
    return mu + 1, mi + 1


def extract_and_cache(encoder=None, tokenizer=None, device=None):
    """Extract embeddings for every split, cache them, and build TRAIN-only
    user/item review representations. Returns the cache dict."""
    config.ensure_cache_dir()
    device = device or get_device()
    if tokenizer is None:
        tokenizer = load_tokenizer(config.XLMR_MODEL_NAME)
    if encoder is None:
        encoder = build_encoder(config.XLMR_MODEL_NAME, device=device, freeze=True)

    train, val, test = load_splits()
    M, N = _infer_sizes(train, val, test)
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"

    cache = {"model_name": config.XLMR_MODEL_NAME,
             "hidden": encoder.hidden_size or config.XLMR_HIDDEN,
             "max_length": config.XLMR_MAX_LENGTH,
             "num_users": M, "num_items": N, "splits": {}}

    for name, df in (("train", train), ("validation", val), ("test", test)):
        if df is None or len(df) == 0:
            continue
        texts = _reviews(df)
        emb = extract_embeddings(encoder, tokenizer, texts, device,
                                 config.XLMR_EXTRACT_BATCH_SIZE,
                                 config.XLMR_MAX_LENGTH, use_amp,
                                 fp16_store=config.CACHE_FP16)
        has_text = np.array([len(t.strip()) > 0 for t in texts], dtype=bool)
        cache["splits"][name] = {
            "embeddings": emb,
            "user_idx": df["user_idx"].to_numpy().astype(np.int64),
            "item_idx": df["item_idx"].to_numpy().astype(np.int64),
            "rating": (df["rating"].to_numpy().astype(np.float32)
                       if "rating" in df.columns else None),
            "has_text": has_text,
        }
        print(f"  [{name}] embeddings {tuple(emb.shape)} "
              f"(with-text {int(has_text.sum())}/{len(has_text)})")

    torch.save(cache, config.REVIEW_EMB_CACHE)
    print(f"  cached review embeddings -> {config.REVIEW_EMB_CACHE}")

    # --- TRAIN-only user/item representations (leakage-safe) ---------------
    tr = cache["splits"].get("train")
    if tr is not None:
        agg = aggregate_user_item(tr["embeddings"], tr["user_idx"],
                                  tr["item_idx"], M, N)
        torch.save({"z_user": agg["z_user"], "user_count": agg["user_count"],
                    "source": "train_only"}, config.USER_REVIEW_EMB_CACHE)
        torch.save({"z_item": agg["z_item"], "item_count": agg["item_count"],
                    "source": "train_only"}, config.ITEM_REVIEW_EMB_CACHE)
        cov_u = int((agg["user_count"] > 0).sum())
        cov_i = int((agg["item_count"] > 0).sum())
        print(f"  user reps z_u {tuple(agg['z_user'].shape)} "
              f"(covered {cov_u}/{M}); item reps z_i {tuple(agg['z_item'].shape)} "
              f"(covered {cov_i}/{N})  [TRAIN reviews only]")

    return cache


def load_cache():
    if not os.path.exists(config.REVIEW_EMB_CACHE):
        raise FileNotFoundError(
            f"No cache at {config.REVIEW_EMB_CACHE}. Run review_embeddings.py first.")
    return torch.load(config.REVIEW_EMB_CACHE, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    dev = get_device()
    print(f"[review_embeddings] device={dev} model={config.XLMR_MODEL_NAME}")
    extract_and_cache(device=dev)
