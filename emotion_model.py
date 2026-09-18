from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn



def load_emotion_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def load_emotion_backbone(model_name: str):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_name)


def labels_from_config(model_config) -> List[str]:

    id2label = getattr(model_config, "id2label", None)
    if not id2label:
        n = getattr(model_config, "num_labels", 0)
        return [f"LABEL_{i}" for i in range(n)]
    items = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])
    return [v for _, v in items]


def resolve_activation(model_config, override: str = "auto") -> str:
    if override in ("softmax", "sigmoid"):
        return override
    ptype = getattr(model_config, "problem_type", None)
    if ptype == "multi_label_classification":
        return "sigmoid"
    return "softmax"


class EmotionClassifier(nn.Module):

    def __init__(self,
                 model_name: str = "tabularisai/multilingual-emotion-classification",
                 backbone: Optional[nn.Module] = None,
                 activation: str = "auto"):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone if backbone is not None \
            else load_emotion_backbone(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.config = getattr(self.backbone, "config", None)
        self.labels = labels_from_config(self.config)
        self.num_labels = len(self.labels)
        self.activation = resolve_activation(self.config, activation)

    def verify_multilingual(self) -> dict:
         mtype = getattr(self.config, "model_type", "unknown")
        info = {"model_name": self.model_name, "model_type": mtype,
                "num_labels": self.num_labels, "labels": self.labels,
                "activation": self.activation}
        multilingual_bases = {"xlm-roberta", "bert",  # mbert reports 'bert'
                              "xlm", "camembert", "rembert", "mdeberta-v2"}
        info["looks_multilingual"] = mtype in multilingual_bases
        return info

    @torch.no_grad()
    def predict(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                use_amp: bool = False) -> torch.Tensor:
        self.backbone.eval()
        if use_amp and input_ids.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out[0]
            logits = logits.float()
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits if hasattr(out, "logits") else out[0]
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)


def build_emotion_classifier(model_name=None, device=None, backbone=None,
                             activation="auto") -> EmotionClassifier:
    import config as _cfg
    model_name = model_name or _cfg.EMOTION_MODEL_NAME
    clf = EmotionClassifier(model_name=model_name, backbone=backbone,
                            activation=activation)
    if device is not None:
        clf = clf.to(device)
    return clf
