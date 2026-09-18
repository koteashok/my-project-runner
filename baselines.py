

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class PopularityRecommender:
    """Most-popular-item recommender. Same ranking for every user."""

    def __init__(self, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.pop = None

    def fit(self, item_idx: np.ndarray):
        self.pop = np.bincount(np.asarray(item_idx, dtype=np.int64),
                               minlength=self.num_items).astype(np.float32)

    def popularity_tensor(self, device) -> torch.Tensor:
        return torch.from_numpy(self.pop).to(device)


class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, dim=64, reg_lambda=1e-4, init_std=0.1):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.reg_lambda = reg_lambda
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.item_embedding.weight, std=init_std)

    def get_all_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def bpr_loss(self, users, pos, neg):
        u = self.user_embedding(users)
        pi = self.item_embedding(pos)
        ni = self.item_embedding(neg)
        pos_s = (u * pi).sum(-1)
        neg_s = (u * ni).sum(-1)
        bpr = F.softplus(neg_s - pos_s).mean()
        reg = (u.pow(2).sum() + pi.pow(2).sum() + ni.pow(2).sum()) / (2.0 * users.shape[0])
        reg = self.reg_lambda * reg
        return bpr + reg, bpr.detach(), reg.detach()



class TextRecommender(nn.Module):
    """Ranks by projected review-representation similarity. Only the projection
    W_t is trained (z_u/z_i are frozen TRAIN-only review features)."""

    def __init__(self, z_user, z_item, common_dim=64, reg_lambda=1e-4, bias=False):
        super().__init__()
        self.num_users = z_user.shape[0]
        self.num_items = z_item.shape[0]
        self.reg_lambda = reg_lambda
        self.register_buffer("z_user", z_user.float())
        self.register_buffer("z_item", z_item.float())
        self.W_t = nn.Linear(z_user.shape[1], common_dim, bias=bias)
        nn.init.xavier_uniform_(self.W_t.weight)

    def get_all_embeddings(self):
        return self.W_t(self.z_user), self.W_t(self.z_item)

    def bpr_loss(self, users, pos, neg):
        hu, hi = self.get_all_embeddings()
        u, pi, ni = hu[users], hi[pos], hi[neg]
        pos_s = (u * pi).sum(-1)
        neg_s = (u * ni).sum(-1)
        bpr = F.softplus(neg_s - pos_s).mean()
        reg = self.reg_lambda * self.W_t.weight.pow(2).sum()
        return bpr + reg, bpr.detach(), reg.detach()
