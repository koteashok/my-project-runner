
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

import config
from emotion_model import build_emotion_classifier, load_emotion_tokenizer


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _read_split(basename: str):
    import pandas as pd
    for ext, rd in ((".parquet", "read_parquet"), (".csv", "read_csv")):
        p = os.path.join(config.PROCESSED_DIR, basename + ext)
        if os.path.exists(p):
            return getattr(pd, rd)(p)
    return None


def load_splits():
    train = _read_split("train")
    if train is None:
        raise FileNotFoundError(f"No processed splits under {config.PROCESSED_DIR}.")
    return train, _read_split("validation"), _read_split("test")


def _reviews(df) -> List[str]:
    if df is None:
        return []
    if "review" in df.columns:
        return df["review"].fillna("").astype(str).tolist()
    return ["" for _ in range(len(df))]


@torch.no_grad()
def extract_emotion_probs(clf, tokenizer, texts: List[str], device,
                          batch_size=32, max_length=192, use_amp=False) -> torch.Tensor:
    clf.eval()
    chunks = []
    for s in range(0, len(texts), batch_size):
        bt = texts[s:s + batch_size]
        enc = tokenizer(bt, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        probs = clf.predict(enc["input_ids"].to(device),
                            enc["attention_mask"].to(device), use_amp=use_amp)
        chunks.append(probs.detach().cpu())
    if not chunks:
        return torch.empty(0, clf.num_labels)
    return torch.cat(chunks, dim=0)


def _neutral_index(labels: List[str]) -> Optional[int]:
    for i, l in enumerate(labels):
        ll = l.strip().lower()
        if ll in ("neutral", "none", "none of them", "no emotion", "other"):
            return i
    return None


def compute_emotion_features(probs: torch.Tensor, labels: List[str]) -> Dict:
    """Return a dict of per-review features from the (N, C) probability matrix."""
    e = probs.float().numpy()
    N, C = e.shape
    eps = config.EMOTION_ENTROPY_EPS

    # distribution p (normalise so entropy is well-defined even for sigmoid probs)
    row_sum = e.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    p = e / row_sum

    dominant = e.argmax(axis=1)
    max_prob = e.max(axis=1)
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    norm_entropy = entropy / (np.log(C) if C > 1 else 1.0)

    mode = config.EMOTION_INTENSITY_MODE
    n_idx = _neutral_index(labels)
    if mode == "max_prob":
        intensity = max_prob.copy()
    elif mode == "peakedness":
        intensity = 1.0 - norm_entropy
    else:  # one_minus_neutral (default), fallback to max_prob if no neutral label
        if n_idx is not None:
            intensity = 1.0 - p[:, n_idx]
        else:
            intensity = max_prob.copy()

    return {
        "probs": probs.float(),                 # (N, C) tensor e_ui
        "dominant_idx": dominant.astype(np.int64),
        "dominant_label": np.array([labels[i] for i in dominant]),
        "max_prob": max_prob.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "norm_entropy": norm_entropy.astype(np.float32),
        "intensity": intensity.astype(np.float32),
    }


def extract_and_cache(clf=None, tokenizer=None, device=None):
    config.ensure_cache_dir()
    device = device or get_device()
    if tokenizer is None:
        tokenizer = load_emotion_tokenizer(config.EMOTION_MODEL_NAME)
    if clf is None:
        clf = build_emotion_classifier(config.EMOTION_MODEL_NAME, device=device,
                                       activation=config.EMOTION_ACTIVATION)

    prov = clf.verify_multilingual()
    print(f"  emotion model: {prov['model_name']} (type={prov['model_type']}, "
          f"multilingual~{prov['looks_multilingual']}, activation={prov['activation']})")
    print(f"  emotion labels (C={prov['num_labels']}): {prov['labels']}")
    if not prov["looks_multilingual"]:
        print("  [warn] model_type does not look multilingual — confirm the model "
              "card supports your languages before trusting cross-lingual outputs.")

    train, val, test = load_splits()
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"

    cache = {"model_name": clf.model_name, "labels": clf.labels,
             "num_labels": clf.num_labels, "activation": clf.activation,
             "is_pseudo_label": True,
             "note": "Auxiliary model-derived emotion pseudo-labels; NOT ground truth.",
             "splits": {}}

    for name, df in (("train", train), ("validation", val), ("test", test)):
        if df is None or len(df) == 0:
            continue
        texts = _reviews(df)
        probs = extract_emotion_probs(clf, tokenizer, texts, device,
                                      config.EMOTION_EXTRACT_BATCH_SIZE,
                                      config.EMOTION_MAX_LENGTH, use_amp)
        feats = compute_emotion_features(probs, clf.labels)
        feats["user_idx"] = df["user_idx"].to_numpy().astype(np.int64)
        feats["item_idx"] = df["item_idx"].to_numpy().astype(np.int64)
        feats["rating"] = (df["rating"].to_numpy().astype(np.float32)
                           if "rating" in df.columns else None)
        feats["language"] = (df["language"].astype(str).to_numpy()
                             if "language" in df.columns else None)
        feats["has_text"] = np.array([len(t.strip()) > 0 for t in texts], dtype=bool)
        cache["splits"][name] = feats
        print(f"  [{name}] emotion probs {tuple(probs.shape)} "
              f"dominant-dist={np.bincount(feats['dominant_idx'], minlength=clf.num_labels).tolist()}")

    torch.save(cache, config.EMOTION_FEATURES_CACHE)
    print(f"  cached emotion features -> {config.EMOTION_FEATURES_CACHE}")
    return cache


def load_emotion_cache():
    if not os.path.exists(config.EMOTION_FEATURES_CACHE):
        raise FileNotFoundError(
            f"No emotion cache at {config.EMOTION_FEATURES_CACHE}. "
            f"Run emotion_features.py first.")
    return torch.load(config.EMOTION_FEATURES_CACHE, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    dev = get_device()
    print(f"[emotion_features] device={dev}")
    extract_and_cache(device=dev)
