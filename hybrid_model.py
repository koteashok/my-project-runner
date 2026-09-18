"""
hybrid_model.py
===============
Hybrid recommender (Phase 5). ACTUALLY COMBINES two branches:

  collaborative : x_u, x_i  = LightGCN(graph)          (Phase 2)
  review-based  : z_u, z_i  = aggregated [h || e || s] (Phases 3-4, TRAIN-only)

Projections into a common latent dim d_common:
    x~ = W_c x        z~ = W_t z            (shared W_c across users/items,
                                             shared W_t across users/items)

Fusion (configurable):
    weighted            h = α x~ + (1-α) z~                 (α fixed, ALPHA)
    weighted_learnable  α = σ(α_logit)  (learnable scalar)
    gate                g = σ(W_g[x~||z~]+b_g);  h = g⊙x~ + (1-g)⊙z~
    concat              h = W_o[x~ || z~]

Score:  ŷ_ui = h_uᵀ h_i.  Trained with BPR (+ L2 on ego embeddings). Emotion
probabilities are INPUT FEATURES (inside z), never a supervised loss.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightgcn import LightGCN


class HybridRecommender(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 norm_adj: torch.Tensor,
                 z_user: torch.Tensor,
                 z_item: torch.Tensor,
                 lgcn_dim: int = 64,
                 common_dim: int = 64,
                 num_layers: int = 3,
                 fusion: str = "weighted",
                 alpha: float = 0.5,
                 reg_lambda: float = 1e-4,
                 proj_bias: bool = False,
                 emb_init_std: float = 0.1,
                 norm_R: Optional[torch.Tensor] = None,
                 lightgcn: Optional[LightGCN] = None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.lgcn_dim = lgcn_dim
        self.text_dim = int(z_user.shape[1])
        self.common_dim = common_dim
        self.fusion = fusion
        self.reg_lambda = reg_lambda

        # --- collaborative branch (LightGCN submodule) -----------------------
        self.lightgcn = lightgcn or LightGCN(
            num_users, num_items, norm_adj, embedding_dim=lgcn_dim,
            num_layers=num_layers, reg_lambda=reg_lambda,
            emb_init_std=emb_init_std, norm_R=norm_R)

        # --- review branch (frozen precomputed features) ---------------------
        self.register_buffer("z_user", z_user.float())
        self.register_buffer("z_item", z_item.float())

        # --- projections into common space -----------------------------------
        self.W_c = nn.Linear(lgcn_dim, common_dim, bias=proj_bias)
        self.W_t = nn.Linear(self.text_dim, common_dim, bias=proj_bias)
        nn.init.xavier_uniform_(self.W_c.weight)
        nn.init.xavier_uniform_(self.W_t.weight)

        # --- fusion parameters ------------------------------------------------
        if fusion == "weighted":
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        elif fusion == "weighted_learnable":
            a = min(max(alpha, 1e-4), 1 - 1e-4)
            self.alpha_logit = nn.Parameter(torch.tensor(math.log(a / (1 - a))))
        elif fusion == "gate":
            self.gate = nn.Linear(2 * common_dim, common_dim)
        elif fusion == "concat":
            self.W_o = nn.Linear(2 * common_dim, common_dim)
        else:
            raise ValueError(f"unknown fusion strategy: {fusion}")

    # -- current fusion weight (for logging) ---------------------------------
    def current_alpha(self) -> Optional[float]:
        if self.fusion == "weighted":
            return float(self.alpha)
        if self.fusion == "weighted_learnable":
            return float(torch.sigmoid(self.alpha_logit))
        return None

    # -- fuse two aligned projected reps -------------------------------------
    def _fuse(self, x_tilde: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        if self.fusion == "weighted":
            a = self.alpha
            return a * x_tilde + (1.0 - a) * z_tilde
        if self.fusion == "weighted_learnable":
            a = torch.sigmoid(self.alpha_logit)
            return a * x_tilde + (1.0 - a) * z_tilde
        if self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([x_tilde, z_tilde], dim=-1)))
            return g * x_tilde + (1.0 - g) * z_tilde
        # concat
        return self.W_o(torch.cat([x_tilde, z_tilde], dim=-1))

    # -- full user/item hybrid representations -------------------------------
    def compute_repr(self):
        x_u, x_i = self.lightgcn.propagate()          # (M,d_lgcn), (N,d_lgcn)
        xt_u, xt_i = self.W_c(x_u), self.W_c(x_i)     # (M,d_common), (N,d_common)
        zt_u, zt_i = self.W_t(self.z_user), self.W_t(self.z_item)
        h_u = self._fuse(xt_u, zt_u)                  # (M,d_common)
        h_i = self._fuse(xt_i, zt_i)                  # (N,d_common)
        return h_u, h_i

    def forward(self):
        return self.compute_repr()

    # -- BPR loss ------------------------------------------------------------
    def bpr_loss(self, users, pos_items, neg_items):
        h_u, h_i = self.compute_repr()
        u = h_u[users]
        pi = h_i[pos_items]
        ni = h_i[neg_items]
        pos_scores = (u * pi).sum(dim=-1)
        neg_scores = (u * ni).sum(dim=-1)
        bpr = F.softplus(neg_scores - pos_scores).mean()

        # L2 on ego (layer-0) embeddings Θ (BPR reg term); Adam weight_decay
        # covers the projection / fusion parameters.
        u0 = self.lightgcn.user_embedding(users)
        p0 = self.lightgcn.item_embedding(pos_items)
        n0 = self.lightgcn.item_embedding(neg_items)
        reg = (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) \
            / (2.0 * users.shape[0])
        reg = self.reg_lambda * reg
        return bpr + reg, bpr.detach(), reg.detach()

    # -- scoring for evaluation ---------------------------------------------
    @torch.no_grad()
    def get_all_embeddings(self):
        self.eval()
        return self.compute_repr()

    @torch.no_grad()
    def score_users(self, user_ids, all_users_emb=None, all_items_emb=None):
        if all_users_emb is None or all_items_emb is None:
            all_users_emb, all_items_emb = self.compute_repr()
        return torch.matmul(all_users_emb[user_ids], all_items_emb.t())

    # -- dimensions / param report -------------------------------------------
    def dims(self) -> dict:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "lightgcn_dim": self.lgcn_dim,
            "xlmr_text_dim": self.text_dim,
            "projected_dim": self.common_dim,
            "fused_dim": self.common_dim,
            "trainable_params": int(n_train),
            "fusion": self.fusion,
            "alpha": self.current_alpha(),
        }
