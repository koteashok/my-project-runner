from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import config
from emotion_features import load_emotion_cache, get_device



def _sentiment_rep(h: torch.Tensor) -> torch.Tensor:
    path = config.XLMR_SENTIMENT_BEST
    C = config.SENTIMENT_NUM_CLASSES
    if not os.path.exists(path):
        print(f"  [warn] no sentiment head at {path}; using zero s_ui ({C}-d).")
        return torch.zeros(h.shape[0], C)
    from xlmr_sentiment import SentimentHead
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = SentimentHead(ckpt["hidden"], ckpt["num_classes"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(h.float())
        if config.SENTIMENT_REP == "logits":
            return logits
        return torch.softmax(logits, dim=-1)



class ReviewFusionProjector(nn.Module):


    def __init__(self, in_dim: int, latent_dim: int,
                 use_layernorm: bool = True, activation: str = "gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.proj = nn.Linear(in_dim, latent_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU(),
                    "none": nn.Identity()}.get(activation, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(self.norm(x)))



def build_representations(device=None) -> Dict:
    device = device or get_device()
    from review_embeddings import load_cache as load_h_cache
    h_cache = load_h_cache()
    e_cache = load_emotion_cache()

    hidden = h_cache["hidden"]
    C_emotion = e_cache["num_labels"]
    C_sent = config.SENTIMENT_NUM_CLASSES
    in_dim = hidden + C_emotion + C_sent
    latent = config.FUSION_LATENT_DIM

    torch.manual_seed(config.RANDOM_SEED)
    projector = ReviewFusionProjector(
        in_dim, latent, config.FUSION_USE_LAYERNORM, config.FUSION_ACTIVATION
    ).to(device).eval()

    out = {"in_dim": in_dim, "hidden": hidden, "emotion_dim": C_emotion,
           "sentiment_dim": C_sent, "latent_dim": latent,
           "emotion_labels": e_cache["labels"],
           "is_pseudo_label": True,
           "note": "z_ui=[h||e||s]; emotion part is auxiliary pseudo-labels, not GT.",
           "splits": {}}

    for name in ("train", "validation", "test"):
        hs = h_cache["splits"].get(name)
        es = e_cache["splits"].get(name)
        if hs is None or es is None:
            continue
        h = hs["embeddings"].float()                         # (n, hidden)
        e = es["probs"].float()                              # (n, C_emotion)
        # alignment sanity: same rows / order across the two caches
        assert len(h) == len(e), f"{name}: h/e length mismatch"
        assert np.array_equal(hs["user_idx"], es["user_idx"]) and \
               np.array_equal(hs["item_idx"], es["item_idx"]), \
               f"{name}: h/e row alignment mismatch"

        s = _sentiment_rep(h)                                # (n, C_sent)
        z_in = torch.cat([h, e, s], dim=1)                   # (n, in_dim)
        with torch.no_grad():
            z = projector(z_in.to(device)).cpu()
        out["splits"][name] = {
            "z": z, "z_concat_dim": in_dim,
            "user_idx": hs["user_idx"], "item_idx": hs["item_idx"],
        }
        print(f"  [{name}] z_ui: concat [h{tuple(h.shape)} | e{tuple(e.shape)} | "
              f"s{tuple(s.shape)}] = ({len(z)},{in_dim}) -> projected ({len(z)},{latent})")

    config.ensure_cache_dir()
    torch.save(out, config.EMOTION_REPR_CACHE)
    print(f"  cached review representations -> {config.EMOTION_REPR_CACHE}")
    return out


if __name__ == "__main__":
    dev = get_device()
    print(f"[emotion_representation] device={dev} latent_dim={config.FUSION_LATENT_DIM}")
    build_representations(dev)
