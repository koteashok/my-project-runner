"""
language_utils.py
=================
Language identification for Phase 8.

Policy
------
1. If the processed splits already carry a language column, that field is used
   as-is (and marked ``inferred=False``).
2. Otherwise language is INFERRED per review by a configurable detector and every
   downstream artefact is clearly marked ``inferred=True``. The actual set of
   languages is determined empirically from the data — none are assumed.

Detector limitations (documented): short or code-mixed reviews are unreliable
(e.g. a brief French phrase may be tagged Catalan), so a `min_chars` guard labels
very short strings as ``unknown``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import config


class LanguageDetector:
    def __init__(self, backend: Optional[str] = None, seed: int = 0,
                 min_chars: int = 1):
        self.backend = backend or config.LANG_DETECT_BACKEND
        self.min_chars = min_chars
        self._impl = None
        self._init_backend(seed)

    def _init_backend(self, seed):
        if self.backend == "langdetect":
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = seed
            self._detect = detect
        elif self.backend == "py3langid":
            import py3langid as langid
            self._detect = lambda t: langid.classify(t)[0]
        else:
            raise ValueError(f"unknown LANG_DETECT_BACKEND: {self.backend}")

    def detect_one(self, text: str) -> str:
        if text is None:
            return "unknown"
        t = str(text).strip()
        if len(t) < self.min_chars:
            return "unknown"
        try:
            return self._detect(t)
        except Exception:
            return "unknown"

    def detect_many(self, texts: List[str], cap: Optional[int] = None) -> np.ndarray:
        out = []
        for i, t in enumerate(texts):
            if cap is not None and i >= cap:
                out.append("unknown")
            else:
                out.append(self.detect_one(t))
        return np.array(out, dtype=object)


def assign_languages(df, detector: LanguageDetector):
    """Return (lang_array, inferred_flag) for a split DataFrame.
    Uses an existing 'language' column if present, else infers from 'review'."""
    if df is None or len(df) == 0:
        return np.array([], dtype=object), False
    if "language" in df.columns and df["language"].notna().any():
        return df["language"].astype(str).to_numpy(), False
    texts = (df["review"].fillna("").astype(str).tolist()
             if "review" in df.columns else ["" for _ in range(len(df))])
    return detector.detect_many(texts, cap=config.LANG_DETECT_MAX_PER_SPLIT), True
