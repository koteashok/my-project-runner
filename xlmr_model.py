
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- #
# Pooling                                                                        #
# ----------------------------------------------------------------------------- #
def masked_mean_pool(last_hidden_state: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state : (B, L, H) float
    attention_mask    : (B, L)    {0,1}
    returns           : (B, H)    masked mean over valid tokens
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                          # (B, 1)
    return summed / counts


# ----------------------------------------------------------------------------- #
# Factories (lazy transformers import)                                           #
# ----------------------------------------------------------------------------- #
def load_tokenizer(model_name: str):
    """Return the XLM-R tokenizer (downloads on first use)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def load_backbone(model_name: str):
    """Return the raw XLM-R AutoModel (downloads on first use)."""
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_name)


# ----------------------------------------------------------------------------- #
# Encoder                                                                        #
# ----------------------------------------------------------------------------- #
class XLMREncoder(nn.Module):
    """
    Wraps an XLM-R backbone and exposes `encode()` returning masked-mean-pooled
    review embeddings. `backbone` can be injected (tests / custom checkpoints);
    otherwise it is loaded from `model_name`.

    The backbone is expected to return an object exposing `.last_hidden_state`
    of shape (B, L, H) — exactly the HuggingFace AutoModel contract.
    """

    def __init__(self,
                 model_name: str = "xlm-roberta-base",
                 backbone: Optional[nn.Module] = None,
                 freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone if backbone is not None else load_backbone(model_name)
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        # hidden size (768 for xlm-roberta-base); read from config when present
        self.hidden_size = getattr(getattr(self.backbone, "config", None),
                                   "hidden_size", None)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") \
            else out[0]
        return masked_mean_pool(last_hidden, attention_mask)

    @torch.no_grad()
    def encode(self,
               input_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               use_amp: bool = False) -> torch.Tensor:
        """Frozen feature extraction (no grad). Uses CUDA autocast fp16 when
        `use_amp` and a CUDA device is in play."""
        self.backbone.eval()
        device_type = input_ids.device.type
        if use_amp and device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pooled = self.forward(input_ids, attention_mask)
            return pooled.float()
        return self.forward(input_ids, attention_mask)


def build_encoder(model_name: str = "xlm-roberta-base",
                  device: Optional[torch.device] = None,
                  backbone: Optional[nn.Module] = None,
                  freeze: bool = True) -> XLMREncoder:
    enc = XLMREncoder(model_name=model_name, backbone=backbone, freeze=freeze)
    if device is not None:
        enc = enc.to(device)
    return enc
