"""
lightgcn.py
===========
LightGCN collaborative-filtering model (Phase 2), implemented from scratch in
PyTorch. Uses ONLY the user-item interaction graph — no text / XLM-RoBERTa.

Contents
--------
- set_seed()                          reproducibility across random/numpy/torch
- build_norm_adjacency()              sparse Â = D^{-1/2} A D^{-1/2}
- LightGCN(nn.Module)                 embeddings, propagation, prediction, BPR loss

Graph
-----
Given the binary interaction matrix R ∈ R^{M×N}, the bipartite adjacency is

        A = [[0, R], [Rᵀ, 0]]   (size (M+N)×(M+N))

and the symmetrically-normalised operator is Â = D^{-1/2} A D^{-1/2}, where D is
the diagonal degree matrix of A. Â is stored as a sparse COO tensor.

Propagation
-----------
E^{(k+1)} = Â E^{(k)}, with E^{(0)} = [user_emb ; item_emb] ∈ R^{(M+N)×d}.
Because of A's block structure, one matrix step is *exactly* the two per-node
equations

    x_u^{(k+1)} = Σ_{i∈N(u)} 1/sqrt(|N(u)||N(i)|) · x_i^{(k)}
    x_i^{(k+1)} = Σ_{u∈N(i)} 1/sqrt(|N(i)||N(u)|) · x_u^{(k)}

which is provided explicitly in `propagate_explicit` and cross-checked in the
self-test. Layers are combined with uniform weights α_k = 1/(K+1):

    x = Σ_{k=0}^{K} α_k x^{(k)}.

Prediction: ŷ_ui = x_uᵀ x_i.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- #
# Reproducibility                                                                #
# ----------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favour determinism (small perf cost); safe on CPU and GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- #
# Graph construction                                                             #
# ----------------------------------------------------------------------------- #
def build_norm_adjacency(user_idx: np.ndarray,
                         item_idx: np.ndarray,
                         num_users: int,
                         num_items: int,
                         device: torch.device):
    """
    Build the sparse normalised adjacency Â (size (M+N)×(M+N)) and the sparse
    normalised bipartite block R̂ = D_u^{-1/2} R D_i^{-1/2} (size M×N).

    Only the given (training) edges are used. Nodes with zero degree simply have
    no incident edges (no 1/sqrt(0) is ever computed).

    Returns
    -------
    norm_adj  : torch.sparse_coo_tensor  (M+N, M+N)   — Â
    norm_R    : torch.sparse_coo_tensor  (M, N)       — R̂ (for the explicit form)
    deg_user  : torch.Tensor (M,)                     — user degrees
    deg_item  : torch.Tensor (N,)                     — item degrees
    """
    user_idx = np.asarray(user_idx, dtype=np.int64)
    item_idx = np.asarray(item_idx, dtype=np.int64)
    assert user_idx.shape == item_idx.shape

    deg_user = np.bincount(user_idx, minlength=num_users).astype(np.float64)
    deg_item = np.bincount(item_idx, minlength=num_items).astype(np.float64)

    # edge normalisation value 1/sqrt(deg_u * deg_i); guard against 0 (unused edges)
    du = deg_user[user_idx]
    di = deg_item[item_idx]
    denom = np.sqrt(du * di)
    denom[denom == 0.0] = 1.0
    vals = (1.0 / denom).astype(np.float32)

    # --- normalised R̂ (M×N) -------------------------------------------------
    R_indices = torch.from_numpy(np.stack([user_idx, item_idx], axis=0))
    R_values = torch.from_numpy(vals)
    norm_R = torch.sparse_coo_tensor(
        R_indices, R_values, size=(num_users, num_items)
    ).coalesce().to(device)

    # --- full Â (M+N × M+N): top-right = R̂, bottom-left = R̂ᵀ ---------------
    top_rows = user_idx                      # u
    top_cols = item_idx + num_users          # M + i
    bot_rows = item_idx + num_users          # M + i
    bot_cols = user_idx                      # u
    rows = np.concatenate([top_rows, bot_rows])
    cols = np.concatenate([top_cols, bot_cols])
    all_vals = np.concatenate([vals, vals])

    A_indices = torch.from_numpy(np.stack([rows, cols], axis=0))
    A_values = torch.from_numpy(all_vals.astype(np.float32))
    n = num_users + num_items
    norm_adj = torch.sparse_coo_tensor(
        A_indices, A_values, size=(n, n)
    ).coalesce().to(device)

    return (norm_adj, norm_R,
            torch.from_numpy(deg_user.astype(np.float32)).to(device),
            torch.from_numpy(deg_item.astype(np.float32)).to(device))


# ----------------------------------------------------------------------------- #
# Model                                                                          #
# ----------------------------------------------------------------------------- #
class LightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 norm_adj: torch.Tensor,
                 embedding_dim: int = 64,
                 num_layers: int = 3,
                 reg_lambda: float = 1e-4,
                 emb_init_std: float = 0.1,
                 norm_R: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg_lambda = reg_lambda

        # Â and R̂ are buffers (moved with .to(device), not trained).
        self.register_buffer("_norm_adj_indices", norm_adj._indices())
        self.register_buffer("_norm_adj_values", norm_adj._values())
        self._norm_adj_size = (norm_adj.size(0), norm_adj.size(1))
        self._norm_R = norm_R  # optional, kept as plain attribute (sparse)

        # E^(0): ego embeddings (Θ) — the only trainable parameters.
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=emb_init_std)
        nn.init.normal_(self.item_embedding.weight, std=emb_init_std)

    # -- helpers -------------------------------------------------------------
    @property
    def norm_adj(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._norm_adj_indices, self._norm_adj_values, size=self._norm_adj_size
        ).coalesce()

    def ego_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embedding.weight,
                          self.item_embedding.weight], dim=0)

    # -- propagation (matrix form E^{(k+1)} = Â E^{(k)}) --------------------
    def propagate(self):
        """Return (final_user_emb, final_item_emb) using layer combination
        α_k = 1/(K+1). Also the standard LightGCN light graph convolution."""
        e = self.ego_embeddings()                     # (M+N, d)
        layer_embs = [e]
        adj = self.norm_adj
        for _ in range(self.num_layers):
            e = torch.sparse.mm(adj, e)               # Â E^{(k)}
            layer_embs.append(e)
        # α_k = 1/(K+1): simple mean over the K+1 stacked layers
        out = torch.stack(layer_embs, dim=0).mean(dim=0)
        users, items = torch.split(out, [self.num_users, self.num_items], dim=0)
        return users, items

    # -- propagation (explicit per-node user/item equations) ----------------
    def propagate_explicit(self):
        """Equivalent formulation using R̂ directly:
            x_u^{(k+1)} = R̂  x_i^{(k)}
            x_i^{(k+1)} = R̂ᵀ x_u^{(k)}
        Provided to make the user/item equations explicit; cross-checked against
        `propagate` in the self-test. Requires norm_R to have been supplied."""
        assert self._norm_R is not None, "norm_R required for explicit propagation"
        R = self._norm_R
        Rt = R.transpose(0, 1).coalesce()
        xu = self.user_embedding.weight
        xi = self.item_embedding.weight
        u_layers, i_layers = [xu], [xi]
        for _ in range(self.num_layers):
            new_u = torch.sparse.mm(R, xi)            # (M,d)
            new_i = torch.sparse.mm(Rt, xu)           # (N,d)
            xu, xi = new_u, new_i
            u_layers.append(xu)
            i_layers.append(xi)
        users = torch.stack(u_layers, dim=0).mean(dim=0)
        items = torch.stack(i_layers, dim=0).mean(dim=0)
        return users, items

    # -- prediction ----------------------------------------------------------
    def forward(self):
        """Convenience: return final (user, item) embeddings."""
        return self.propagate()

    def score_pairs(self, user_e, pos_e):
        return torch.sum(user_e * pos_e, dim=-1)

    # -- BPR loss ------------------------------------------------------------
    def bpr_loss(self, users, pos_items, neg_items):
        """
        L_BPR = -mean log σ(ŷ_ui - ŷ_uj) + λ‖Θ‖²  (Θ = layer-0 embeddings).

        users/pos_items/neg_items are 1-D LongTensors of equal length.
        Returns (total_loss, bpr_term, reg_term).
        """
        all_users, all_items = self.propagate()
        u = all_users[users]
        pi = all_items[pos_items]
        ni = all_items[neg_items]

        pos_scores = self.score_pairs(u, pi)
        neg_scores = self.score_pairs(u, ni)

        # numerically stable -log σ(x) = softplus(-x)
        bpr = torch.nn.functional.softplus(neg_scores - pos_scores).mean()

        # L2 on the ego (layer-0) embeddings only, averaged over the batch
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        reg = (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) \
            / (2.0 * users.shape[0])
        reg = self.reg_lambda * reg

        return bpr + reg, bpr.detach(), reg.detach()

    # -- scoring for recommendation -----------------------------------------
    @torch.no_grad()
    def get_all_embeddings(self):
        self.eval()
        return self.propagate()

    @torch.no_grad()
    def score_users(self, user_ids: torch.Tensor,
                    all_items_emb: Optional[torch.Tensor] = None,
                    all_users_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return a (len(user_ids) × num_items) score matrix ŷ = x_u · x_iᵀ."""
        if all_users_emb is None or all_items_emb is None:
            all_users_emb, all_items_emb = self.propagate()
        u = all_users_emb[user_ids]
        return torch.matmul(u, all_items_emb.t())
