
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np

import config



def _concat_split_field(cache: Dict, field: str):
    parts = []
    for name in ("train", "validation", "test"):
        s = cache["splits"].get(name)
        if s is None or s.get(field) is None:
            continue
        v = s[field]
        parts.append(v.numpy() if hasattr(v, "numpy") else np.asarray(v))
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def rating_to_sentiment(ratings: np.ndarray) -> np.ndarray:
    r = np.asarray(ratings, dtype=np.float64)
    lab = np.full(r.shape, -1, dtype=np.int64)
    lab[r <= config.SENTIMENT_NEG_MAX] = 0
    lab[r == config.SENTIMENT_NEU_VALUE] = 1
    lab[r >= config.SENTIMENT_NEU_VALUE + 1] = 2
    return lab


def emotion_distribution(cache: Dict) -> Dict:
    """Mean probability mass per emotion + dominant-emotion counts (all splits)."""
    labels = cache["labels"]
    probs = _concat_split_field(cache, "probs")
    dom = _concat_split_field(cache, "dominant_idx")
    C = len(labels)
    mean_mass = probs.mean(axis=0) if probs is not None else np.zeros(C)
    dom_counts = np.bincount(dom, minlength=C) if dom is not None else np.zeros(C, int)
    return {"labels": labels, "mean_prob": mean_mass, "dominant_counts": dom_counts,
            "n": 0 if probs is None else len(probs)}


def dominant_by_language(cache: Dict) -> Optional[Dict]:
    lang = _concat_split_field(cache, "language")
    dom = _concat_split_field(cache, "dominant_idx")
    if lang is None or dom is None:
        return None
    labels = cache["labels"]
    table: Dict[str, np.ndarray] = {}
    for lg in np.unique(lang):
        mask = lang == lg
        table[str(lg)] = np.bincount(dom[mask], minlength=len(labels))
    return {"labels": labels, "by_language": table}


def emotion_sentiment_relationship(cache: Dict) -> Optional[Dict]:
    """Cross-tab: dominant emotion (rows) x rating-derived sentiment (cols)."""
    rating = _concat_split_field(cache, "rating")
    dom = _concat_split_field(cache, "dominant_idx")
    if rating is None or dom is None:
        return None
    sent = rating_to_sentiment(rating)
    keep = sent >= 0
    dom, sent = dom[keep], sent[keep]
    labels = cache["labels"]
    C, S = len(labels), config.SENTIMENT_NUM_CLASSES
    table = np.zeros((C, S), dtype=np.int64)
    for d, s in zip(dom, sent):
        table[d, s] += 1
    return {"emotion_labels": labels,
            "sentiment_labels": config.SENTIMENT_LABELS, "table": table}

def plot_emotion_distribution(dist: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = dist["labels"]
    mean_prob = dist["mean_prob"]
    dom = dist["dominant_counts"]
    dom_frac = dom / dom.sum() if dom.sum() > 0 else dom

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.1), 4.5))
    axes[0].bar(x, mean_prob, color="#4C72B0")
    axes[0].set_title("Mean emotion probability mass")
    axes[0].set_ylabel("mean probability")
    axes[1].bar(x, dom_frac, color="#DD8452")
    axes[1].set_title("Dominant-emotion share")
    axes[1].set_ylabel("fraction of reviews")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Emotion distribution (auxiliary model-derived pseudo-labels — "
                 "NOT ground truth)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_statistics_csv(path: str, dist: Dict, rel: Optional[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = max(dist["n"], 1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# auxiliary model-derived emotion pseudo-labels; not ground truth"])
        w.writerow(["emotion", "mean_probability", "dominant_count",
                    "dominant_fraction"])
        for i, lab in enumerate(dist["labels"]):
            w.writerow([lab, f"{dist['mean_prob'][i]:.6f}",
                        int(dist["dominant_counts"][i]),
                        f"{dist['dominant_counts'][i] / n:.6f}"])
        if rel is not None:
            w.writerow([])
            w.writerow(["# emotion x sentiment (rating-derived) counts"])
            w.writerow(["emotion"] + list(rel["sentiment_labels"]))
            for i, lab in enumerate(rel["emotion_labels"]):
                w.writerow([lab] + [int(v) for v in rel["table"][i]])


def print_reports(cache, dist, rel, bylang, samples=None):
    print("\n=== EMOTION DISTRIBUTION (auxiliary pseudo-labels, not ground truth) ===")
    order = np.argsort(-dist["mean_prob"])
    for i in order:
        print(f"  {dist['labels'][i]:<12} mean_p={dist['mean_prob'][i]:.4f}  "
              f"dominant={int(dist['dominant_counts'][i])}")
    if rel is not None:
        print("\n=== EMOTION x SENTIMENT (rating-derived) ===")
        header = "  " + " " * 12 + " ".join(f"{s[:7]:>8}" for s in rel["sentiment_labels"])
        print(header)
        for i, lab in enumerate(rel["emotion_labels"]):
            print(f"  {lab:<12}" + " ".join(f"{v:>8}" for v in rel["table"][i]))
    if bylang is not None:
        print("\n=== DOMINANT EMOTION BY LANGUAGE ===")
        for lg, counts in bylang["by_language"].items():
            top = int(np.argmax(counts))
            print(f"  {lg:<8} top={bylang['labels'][top]} counts={counts.tolist()}")
    else:
        print("\n(no language column available -> emotion-by-language skipped)")
    if samples:
        print("\n=== SAMPLE REVIEW -> SENTIMENT + EMOTION ===")
        for s in samples:
            print(f"  review : {s['review'][:60]}")
            print(f"     dominant emotion={s['dominant']} (p={s['max_prob']:.2f}) "
                  f"entropy={s['entropy']:.2f} intensity={s['intensity']:.2f} "
                  f"| rating-sentiment={s['sentiment']}")


def collect_samples(cache, k=5):
    s = cache["splits"].get("train")
    if s is None:
        return []
    import os as _os
    # re-read review text from Phase 1 for display
    import pandas as pd
    p = None
    for ext in (".parquet", ".csv"):
        cand = _os.path.join(config.PROCESSED_DIR, "train" + ext)
        if _os.path.exists(cand):
            p = cand; break
    reviews = None
    if p is not None:
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        reviews = df["review"].fillna("").astype(str).tolist() if "review" in df else None
    rating = s.get("rating")
    sent = rating_to_sentiment(rating) if rating is not None else None
    out = []
    for i in range(min(k, len(s["dominant_idx"]))):
        out.append({
            "review": reviews[i] if reviews else "(text unavailable)",
            "dominant": str(s["dominant_label"][i]),
            "max_prob": float(s["max_prob"][i]),
            "entropy": float(s["entropy"][i]),
            "intensity": float(s["intensity"][i]),
            "sentiment": (config.SENTIMENT_LABELS[sent[i]]
                          if sent is not None and sent[i] >= 0 else "n/a"),
        })
    return out


def run():
    from emotion_features import load_emotion_cache
    cache = load_emotion_cache()
    dist = emotion_distribution(cache)
    rel = emotion_sentiment_relationship(cache)
    bylang = dominant_by_language(cache)
    samples = collect_samples(cache)

    print_reports(cache, dist, rel, bylang, samples)
    config.ensure_figures_dir()
    plot_emotion_distribution(dist, config.EMOTION_FIG)
    write_statistics_csv(config.EMOTION_STATS_CSV, dist, rel)
    print(f"\nwrote {config.EMOTION_STATS_CSV}")
    print(f"wrote {config.EMOTION_FIG}")
    return dist, rel


if __name__ == "__main__":
    run()
