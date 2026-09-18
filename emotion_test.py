"""
emotion_test.py
===============
Self-contained test/demo of the Phase 4 emotion module. The real multilingual
emotion model needs network access; offline we inject a SHAPE-IDENTICAL stand-in
(exposes `.config.id2label` / `.config.problem_type` and returns `.logits`). The
real run swaps in EMOTION_MODEL_NAME with no other change.

Confirms (real numbers from stand-ins; emotion outputs are PSEUDO-LABELS):
  - emotion labels are READ FROM the model config (not hardcoded)
  - probability vectors are valid; entropy / dominant / max / intensity correct
  - z_ui = [h || e || s] has the right concat dim and projects to FUSION_LATENT_DIM
  - distributions, emotion x sentiment table, by-language report
  - emotion_statistics.csv + emotion_distribution.png are written
  - sample review -> sentiment + emotion predictions
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

import config


# --------------------------------------------------------------------------- #
# Stand-ins (NOT the real models; offline plumbing only)                       #
# --------------------------------------------------------------------------- #
EMO_LABELS = ["anger", "contempt", "disgust", "fear", "frustration", "gratitude",
              "joy", "love", "neutral", "sadness", "surprise"]  # tabularisai set


class _Cfg:
    def __init__(self, labels, problem_type=None, model_type="xlm-roberta"):
        self.id2label = {i: l for i, l in enumerate(labels)}
        self.num_labels = len(labels)
        self.problem_type = problem_type
        self.model_type = model_type


class DummyEmotionBackbone(nn.Module):
    def __init__(self, labels, vocab=4096):
        super().__init__()
        self.config = _Cfg(labels)                      # softmax path; has 'neutral'
        self.emb = nn.Embedding(vocab, len(labels))
    def forward(self, input_ids, attention_mask=None):
        class _O: pass
        o = _O()
        # crude pooled logits = mean token-embedding rows
        o.logits = self.emb(input_ids).mean(dim=1)
        return o


class DummyXLMRBackbone(nn.Module):
    def __init__(self, hidden=768, vocab=4096):
        super().__init__()
        class _C: pass
        self.config = _C(); self.config.hidden_size = hidden
        self.emb = nn.Embedding(vocab, hidden)
    def forward(self, input_ids, attention_mask=None):
        class _O: pass
        o = _O(); o.last_hidden_state = self.emb(input_ids); return o


class DummyTokenizer:
    def __init__(self, vocab=4096): self.vocab = vocab
    def _ids(self, t):
        toks = t.split()
        ids = [1] + [2 + (hash(w) % (self.vocab - 2)) for w in toks]
        return ids if len(ids) > 1 else [1, 1]
    def __call__(self, texts, padding=True, truncation=True, max_length=256,
                 return_tensors="pt"):
        seqs = [self._ids(t)[:max_length] for t in texts]
        L = max(len(s) for s in seqs)
        ii = [s + [0] * (L - len(s)) for s in seqs]
        am = [[1] * len(s) + [0] * (L - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am)}


def route_outputs(tmp):
    for k, v in {
        "RESULTS_DIR": tmp, "PROCESSED_DIR": f"{tmp}/processed",
        "MAPPINGS_DIR": f"{tmp}/mappings", "CACHE_DIR": f"{tmp}/cache",
        "CHECKPOINT_DIR": f"{tmp}/checkpoints", "FIGURES_DIR": f"{tmp}/figures",
        "REVIEW_EMB_CACHE": f"{tmp}/cache/review_embeddings.pt",
        "USER_REVIEW_EMB_CACHE": f"{tmp}/cache/user_review_emb.pt",
        "ITEM_REVIEW_EMB_CACHE": f"{tmp}/cache/item_review_emb.pt",
        "XLMR_SENTIMENT_BEST": f"{tmp}/checkpoints/xlmr_sentiment_best.pt",
        "XLMR_SENTIMENT_RESULTS": f"{tmp}/xlmr_sentiment_results.csv",
        "XLMR_SENTIMENT_HISTORY": f"{tmp}/xlmr_sentiment_history.json",
        "EMOTION_FEATURES_CACHE": f"{tmp}/cache/emotion_features.pt",
        "EMOTION_REPR_CACHE": f"{tmp}/cache/emotion_review_representation.pt",
        "EMOTION_STATS_CSV": f"{tmp}/emotion_statistics.csv",
        "EMOTION_FIG": f"{tmp}/figures/emotion_distribution.png",
    }.items():
        setattr(config, k, v)
    for d in ("processed", "mappings", "cache", "checkpoints", "figures"):
        os.makedirs(f"{tmp}/{d}", exist_ok=True)


def write_fake_phase1(M=40, N=25, seed=5):
    import pandas as pd, json
    rng = np.random.RandomState(seed)
    revs = ["I love this, amazing!", "Muy buen producto", "Sehr schlecht!",
            "とても良い", "Terrible, broke fast", "average nothing special",
            "Chất lượng tạm ổn", "C'est correct"]
    langs = ["en", "es", "de", "ja", "en", "en", "vi", "fr"]
    rows = []
    for u in range(M):
        for _ in range(rng.randint(3, 7)):
            j = rng.randint(0, len(revs))
            rows.append((u, rng.randint(0, N), int(rng.randint(1, 6)), revs[j],
                         langs[j], int(rng.randint(1_500_000_000, 1_700_000_000))))
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "review",
                                     "language", "timestamp"])
    tr, va, te = [], [], []
    for u, g in df.groupby("user_idx"):
        g = g.sort_values("timestamp")
        if len(g) >= 3: tr.append(g.iloc[:-2]); va.append(g.iloc[-2:-1]); te.append(g.iloc[-1:])
        elif len(g) == 2: tr.append(g.iloc[:1]); te.append(g.iloc[1:2])
        else: tr.append(g)
    for name, d in [("train", pd.concat(tr)), ("validation", pd.concat(va)),
                    ("test", pd.concat(te))]:
        d = d.copy(); d["user"] = d["user_idx"]; d["item"] = d["item_idx"]
        d.to_parquet(f"{config.PROCESSED_DIR}/{name}.parquet", index=False)
    json.dump({str(i): i for i in range(M)}, open(f"{config.MAPPINGS_DIR}/user2idx.json", "w", encoding="utf-8"))
    json.dump({str(i): i for i in range(N)}, open(f"{config.MAPPINGS_DIR}/item2idx.json", "w", encoding="utf-8"))


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")

    # ------------------------------------------------ 1. labels read from config
    from emotion_model import build_emotion_classifier
    clf = build_emotion_classifier("stand-in", device=device,
                                   backbone=DummyEmotionBackbone(EMO_LABELS),
                                   activation="auto")
    prov = clf.verify_multilingual()
    assert clf.labels == EMO_LABELS, "labels must come from model config"
    assert clf.num_labels == 11 and clf.activation == "softmax"
    print(f"[1] labels from config OK: C={clf.num_labels} labels={clf.labels}")
    print(f"    provenance: type={prov['model_type']} multilingual~{prov['looks_multilingual']}")

    # ------------------------------------------------ 2. feature math correctness
    from emotion_features import compute_emotion_features
    probs = torch.tensor([[0.7, 0.1, 0.2] + [0.0] * 8,       # peaked
                          [1/11] * 11])                       # uniform
    feats = compute_emotion_features(probs, EMO_LABELS)
    # manual entropy of uniform over 11 = log(11); peaked row lower
    assert abs(feats["entropy"][1] - np.log(11)) < 1e-4, "uniform entropy wrong"
    assert feats["entropy"][0] < feats["entropy"][1], "peaked should have lower H"
    assert feats["dominant_label"][0] == "anger" and abs(feats["max_prob"][0] - 0.7) < 1e-6
    # intensity default = 1 - P(neutral); neutral idx=8 -> ~1 - small
    assert 0.0 <= feats["intensity"][0] <= 1.0
    print(f"[2] feature math OK: H(uniform)={feats['entropy'][1]:.3f}=log11; "
          f"H(peaked)={feats['entropy'][0]:.3f}; dominant/max/intensity valid")

    # ------------------------------------------------ 3-7 full pipeline
    route_outputs("results_t4")
    write_fake_phase1()

    # Phase 3 h-cache + sentiment head (stand-in XLM-R)
    import review_embeddings as re_mod
    from xlmr_model import build_encoder
    tok = DummyTokenizer()
    enc = build_encoder("stand-in", device=device,
                        backbone=DummyXLMRBackbone(config.XLMR_HIDDEN), freeze=True)
    re_mod.extract_and_cache(encoder=enc, tokenizer=tok, device=device)
    import xlmr_sentiment; xlmr_sentiment.run()

    # Phase 4 emotion features (stand-in emotion model)
    import emotion_features as ef
    ecache = ef.extract_and_cache(clf=clf, tokenizer=tok, device=device)
    assert os.path.exists(config.EMOTION_FEATURES_CACHE)
    tr = ecache["splits"]["train"]
    assert tuple(tr["probs"].shape)[1] == 11
    print(f"[3] emotion features OK: cached {config.EMOTION_FEATURES_CACHE}; "
          f"is_pseudo_label={ecache['is_pseudo_label']}")

    # z_ui = [h || e || s] projection
    import emotion_representation as er
    rep = er.build_representations(device=device)
    zt = rep["splits"]["train"]
    exp_concat = config.XLMR_HIDDEN + 11 + config.SENTIMENT_NUM_CLASSES
    assert rep["in_dim"] == exp_concat, (rep["in_dim"], exp_concat)
    assert tuple(zt["z"].shape)[1] == config.FUSION_LATENT_DIM
    print(f"[4] z_ui OK: concat dim={rep['in_dim']} (=768+11+3) -> "
          f"projected latent={config.FUSION_LATENT_DIM}; shape {tuple(zt['z'].shape)}")

    # reports + figure + csv
    import emotion_evaluation as ee
    dist, rel = ee.run()
    assert os.path.exists(config.EMOTION_STATS_CSV) and os.path.exists(config.EMOTION_FIG)
    assert len(dist["labels"]) == 11 and dist["mean_prob"].shape == (11,)
    assert rel is not None and rel["table"].shape == (11, 3)
    print(f"[5] reports OK: wrote {config.EMOTION_STATS_CSV} and {config.EMOTION_FIG}; "
          f"emotion×sentiment table {rel['table'].shape}")

    bylang = ee.dominant_by_language(ecache)
    assert bylang is not None, "by-language report should be produced (language col present)"
    print(f"[6] by-language OK: languages={list(bylang['by_language'].keys())}")

    samples = ee.collect_samples(ecache, k=3)
    print("\n[7] sample review -> sentiment + emotion (stand-in weights; "
          "emotion = auxiliary PSEUDO-labels, not ground truth):")
    for s in samples:
        print(f"    '{s['review'][:34]:<34}' -> emotion={s['dominant']:<10} "
              f"p={s['max_prob']:.2f} H={s['entropy']:.2f} intensity={s['intensity']:.2f} "
              f"| sentiment={s['sentiment']}")

    print("\nALL PHASE 4 SELF-TESTS PASSED.")


if __name__ == "__main__":
    main()
