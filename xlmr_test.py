

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

import config
from xlmr_model import masked_mean_pool, XLMREncoder, build_encoder
import review_embeddings as re_mod
import sentiment_evaluation as seval


class DummyConfig:
    def __init__(self, hidden): self.hidden_size = hidden


class DummyBackbone(nn.Module):
    """Maps token ids -> per-token vectors, mimicking AutoModel's
    `.last_hidden_state` (B, L, H) contract."""
    def __init__(self, vocab=4096, hidden=768):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.config = DummyConfig(hidden)

    def forward(self, input_ids, attention_mask=None):
        class _Out:  # minimal object with .last_hidden_state
            pass
        o = _Out()
        o.last_hidden_state = self.emb(input_ids)
        return o


class DummyTokenizer:
    """Whitespace + hashing tokenizer with the HF call contract (padding,
    truncation, max_length, return_tensors)."""
    def __init__(self, vocab=4096): self.vocab = vocab

    def _ids(self, text):
        toks = text.split()
        ids = [1] + [2 + (hash(t) % (self.vocab - 2)) for t in toks]  # 1 = BOS
        return ids if len(ids) > 1 else [1, 1]

    def __call__(self, texts, padding=True, truncation=True, max_length=256,
                 return_tensors="pt"):
        seqs = [self._ids(t)[:max_length] for t in texts]
        L = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = L - len(s)
            input_ids.append(s + [0] * pad)
            attn.append([1] * len(s) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long)}


# ----------------------------------------------------------------------------- #
def route_outputs_to(tmp):
    config.RESULTS_DIR = tmp
    config.PROCESSED_DIR = f"{tmp}/processed"
    config.MAPPINGS_DIR = f"{tmp}/mappings"
    config.CACHE_DIR = f"{tmp}/cache"
    config.CHECKPOINT_DIR = f"{tmp}/checkpoints"
    config.REVIEW_EMB_CACHE = f"{tmp}/cache/review_embeddings.pt"
    config.USER_REVIEW_EMB_CACHE = f"{tmp}/cache/user_review_emb.pt"
    config.ITEM_REVIEW_EMB_CACHE = f"{tmp}/cache/item_review_emb.pt"
    config.XLMR_SENTIMENT_BEST = f"{tmp}/checkpoints/xlmr_sentiment_best.pt"
    config.XLMR_SENTIMENT_RESULTS = f"{tmp}/xlmr_sentiment_results.csv"
    config.XLMR_SENTIMENT_HISTORY = f"{tmp}/xlmr_sentiment_history.json"
    for d in (config.PROCESSED_DIR, config.MAPPINGS_DIR, config.CACHE_DIR,
              config.CHECKPOINT_DIR):
        os.makedirs(d, exist_ok=True)


def write_fake_phase1(M=40, N=25, seed=3):
    import pandas as pd, json
    rng = np.random.RandomState(seed)
    multiling = [
        "This product is amazing, absolutely love it",          # en +
        "Muy buen producto, lo recomiendo totalmente",           # es +
        "Sehr schlecht, funktioniert überhaupt nicht",           # de -
        "とても良い商品です",                                       # ja +
        "Terrible quality, broke immediately",                   # en -
        "C'est correct, sans plus",                              # fr neutral-ish
        "average, nothing special",                              # en neutral
        "Chất lượng tạm ổn",                                     # vi neutral
    ]
    rows = []
    for u in range(M):
        for _ in range(rng.randint(3, 7)):
            i = rng.randint(0, N)
            rating = int(rng.randint(1, 6))
            rows.append((u, i, rating, multiling[rng.randint(0, len(multiling))],
                         int(rng.randint(1_500_000_000, 1_700_000_000))))
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating",
                                     "review", "timestamp"])
    # per-user chronological LOO split (mirrors Phase 1)
    tr, va, te = [], [], []
    for u, g in df.groupby("user_idx"):
        g = g.sort_values("timestamp")
        if len(g) >= 3: tr.append(g.iloc[:-2]); va.append(g.iloc[-2:-1]); te.append(g.iloc[-1:])
        elif len(g) == 2: tr.append(g.iloc[:1]); te.append(g.iloc[1:2])
        else: tr.append(g)
    import pandas as pd
    for name, d in [("train", pd.concat(tr)), ("validation", pd.concat(va)),
                    ("test", pd.concat(te))]:
        d = d.copy(); d["user"] = d["user_idx"]; d["item"] = d["item_idx"]
        d.to_parquet(f"{config.PROCESSED_DIR}/{name}.parquet", index=False)
    json.dump({str(i): i for i in range(M)},
              open(f"{config.MAPPINGS_DIR}/user2idx.json", "w", encoding="utf-8"))
    json.dump({str(i): i for i in range(N)},
              open(f"{config.MAPPINGS_DIR}/item2idx.json", "w", encoding="utf-8"))
    return M, N


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    H = config.XLMR_HIDDEN
    tok = DummyTokenizer()
    backbone = DummyBackbone(hidden=H)
    encoder = build_encoder("stand-in", device=device, backbone=backbone, freeze=True)
    assert encoder.hidden_size == H
    print(f"stand-in encoder hidden={encoder.hidden_size} (real xlm-roberta-base=768)")

    # -------------------------------------------------- 1. pooling correctness
    lhs = torch.randn(1, 5, H)
    am = torch.tensor([[1, 1, 1, 0, 0]])
    manual = lhs[0, :3].mean(0)
    pooled = masked_mean_pool(lhs, am)[0]
    assert torch.allclose(pooled, manual, atol=1e-6), "masked mean pooling wrong"
    # padding invariance: same seq with extra pad tokens -> same result
    lhs_pad = torch.cat([lhs, torch.randn(1, 3, H)], dim=1)
    am_pad = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    pooled_pad = masked_mean_pool(lhs_pad, am_pad)[0]
    assert torch.allclose(pooled, pooled_pad, atol=1e-6), "pooling not padding-invariant"
    print("[1] masked mean pooling OK: matches manual mean; padding-invariant "
          "(uses tokens+mask, not [CLS])")

    # -------------------------------------------------- 2. extraction shapes
    sample_reviews = ["great product love it", "muy malo no funciona",
                      "とても良い", "average nothing special"]
    emb = re_mod.extract_embeddings(encoder, tok, sample_reviews, device,
                                    batch_size=2, max_length=config.XLMR_MAX_LENGTH,
                                    use_amp=False, fp16_store=False)
    assert tuple(emb.shape) == (len(sample_reviews), H)
    print(f"[2] extraction OK: reviews={len(sample_reviews)} -> embeddings "
          f"{tuple(emb.shape)}")

    # -------------------------------------------------- 3. aggregation shapes
    u_idx = np.array([0, 0, 1, 2]); i_idx = np.array([0, 1, 1, 2])
    agg = re_mod.aggregate_user_item(emb, u_idx, i_idx, num_users=3, num_items=3)
    assert tuple(agg["z_user"].shape) == (3, H) and tuple(agg["z_item"].shape) == (3, H)
    # user 0 rep == mean of its two reviews
    assert torch.allclose(agg["z_user"][0], emb[:2].float().mean(0), atol=1e-5)
    print(f"[3] aggregation OK: z_u{tuple(agg['z_user'].shape)} "
          f"z_i{tuple(agg['z_item'].shape)}; user-0 rep == mean of its reviews")

    # -------------------------------------------------- 4. rating -> label map
    from xlmr_sentiment import rating_to_sentiment
    labs = rating_to_sentiment(np.array([1, 2, 3, 4, 5, np.nan]))
    assert labs.tolist() == [0, 0, 1, 2, 2, -1], labs.tolist()
    print(f"[4] rating->weak-label map OK: [1,2,3,4,5,NaN] -> {labs.tolist()} "
          f"(0=neg,1=neu,2=pos,-1=drop)")

    # -------------------------------- 5. head trains, loss decreases (synthetic)
    from xlmr_sentiment import SentimentHead
    C = config.SENTIMENT_NUM_CLASSES
    rng = np.random.RandomState(0)
    per = 300
    centers = torch.randn(C, H) * 3.0
    Xs, ys = [], []
    for c in range(C):
        Xs.append(centers[c] + torch.randn(per, H) * 0.5)
        ys += [c] * per
    Xs = torch.cat(Xs); ys = torch.tensor(ys)
    head = SentimentHead(H, C)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    first = last = None
    for ep in range(40):
        opt.zero_grad(); logit = head(Xs); loss = crit(logit, ys)
        loss.backward(); opt.step()
        if ep == 0: first = loss.item()
        last = loss.item()
    assert tuple(head(Xs).shape) == (len(ys), C)
    assert last < first, f"CE loss did not decrease ({first:.3f}->{last:.3f})"
    acc = (head(Xs).argmax(1) == ys).float().mean().item()
    print(f"[5] head OK: logits {tuple(head(Xs).shape)}; CE {first:.3f}->{last:.3f}; "
          f"train acc on separable synthetic data={acc:.3f}")

    # -------------------------------------------------- 6. metrics + confusion
    yp = head(Xs).argmax(1).numpy()
    m = seval.compute_metrics(ys.numpy(), yp, C, config.SENTIMENT_LABELS)
    assert np.array(m["confusion_matrix"]).shape == (C, C)
    assert set(m.keys()) >= {"accuracy", "macro_f1", "weighted_f1"}
    print(f"[6] metrics OK: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
          f"weightedF1={m['weighted_f1']:.3f}; confusion matrix "
          f"{np.array(m['confusion_matrix']).shape}")

    # ---------------------------- 7. end-to-end cache + checkpoint + results.csv
    route_outputs_to("results_t3")
    M, N = write_fake_phase1()
    re_mod.extract_and_cache(encoder=encoder, tokenizer=tok, device=device)
    assert os.path.exists(config.REVIEW_EMB_CACHE)
    import xlmr_sentiment
    xlmr_sentiment.run()
    assert os.path.exists(config.XLMR_SENTIMENT_BEST)
    assert os.path.exists(config.XLMR_SENTIMENT_RESULTS)
    print(f"[7] pipeline OK: wrote {config.REVIEW_EMB_CACHE}, "
          f"{config.XLMR_SENTIMENT_BEST}, {config.XLMR_SENTIMENT_RESULTS}")

    # -------------------------------------------------- 8. sample predictions
    from xlmr_sentiment import SentimentHead as SH
    ckpt = torch.load(config.XLMR_SENTIMENT_BEST, map_location="cpu", weights_only=False)
    trained = SH(ckpt["hidden"], ckpt["num_classes"]); trained.load_state_dict(ckpt["state_dict"])
    demo = ["This product is amazing, absolutely love it",
            "Sehr schlecht, funktioniert überhaupt nicht",
            "average, nothing special"]
    demo_emb = re_mod.extract_embeddings(encoder, tok, demo, device,
                                         batch_size=8, max_length=256,
                                         use_amp=False, fp16_store=False)
    pred, probs = trained.predict(demo_emb.float())
    print("\n[8] sample review -> embedding -> sentiment (stand-in weights; "
          "labels illustrate the flow, not real XLM-R quality):")
    for r, e, p, pr in zip(demo, demo_emb, pred.tolist(), probs.tolist()):
        lab = config.SENTIMENT_LABELS[p]
        print(f"    review: {r[:44]:<44} | emb dim={e.numel()} "
              f"|emb|={e.norm():.2f} | pred={lab} p={max(pr):.2f}")

    print("\nALL PHASE 3 SELF-TESTS PASSED.")


if __name__ == "__main__":
    main()
