
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch

NEG_INF = -1e9


@torch.no_grad()
def evaluate_full(score_provider: Callable[[torch.Tensor], torch.Tensor],
                  test_pos: Dict[int, np.ndarray],
                  exclude: Dict[int, np.ndarray],
                  num_items: int,
                  ks: List[int],
                  device: torch.device,
                  user_batch: int = 1024) -> Dict[int, Dict[str, float]]:
    users = np.asarray(sorted(test_pos.keys()), dtype=np.int64)
    kmax = max(ks)
    disc = 1.0 / torch.log2(torch.arange(2, kmax + 2, device=device).float())  # (kmax,)
    disc_cumsum = torch.cumsum(disc, dim=0)                                     # (kmax,)

    agg = {k: {m: 0.0 for m in ("precision", "recall", "f1", "hitrate",
                                "ndcg", "map", "mrr")} for k in ks}
    cat = {k: torch.zeros(num_items, dtype=torch.bool, device=device) for k in ks}
    rec_cov = {k: 0 for k in ks}
    score_sum = {k: 0.0 for k in ks}
    score_cnt = {k: 0 for k in ks}
    n_users = 0

    for start in range(0, len(users), user_batch):
        batch = users[start:start + user_batch]
        uids = torch.from_numpy(batch).to(device)
        scores = score_provider(uids).clone().float()          # (B, N)

        gt = torch.zeros_like(scores, dtype=torch.bool)
        for row, u in enumerate(batch):
            if exclude and u in exclude:
                scores[row, torch.from_numpy(exclude[u]).to(device)] = NEG_INF
            gt[row, torch.from_numpy(test_pos[u]).to(device)] = True

        n_gt = gt.sum(dim=1).clamp(min=1).float()
        n_cand = (scores > NEG_INF / 2).sum(dim=1)
        topv, topi = torch.topk(scores, kmax, dim=1)           # (B, kmax)
        rel = torch.gather(gt.float(), 1, topi)                # (B, kmax)
        valid_top = topv > NEG_INF / 2                         # real candidate slots

        for k in ks:
            relk = rel[:, :k]
            validk = valid_top[:, :k]
            hits = relk.sum(dim=1)
            prec = hits / k
            recl = hits / n_gt
            f1 = torch.where((prec + recl) > 0, 2 * prec * recl / (prec + recl),
                             torch.zeros_like(prec))
            hr = (hits > 0).float()

            dcg = (relk * disc[:k]).sum(dim=1)
            m = torch.minimum(n_gt, torch.full_like(n_gt, k)).long().clamp(min=1)
            idcg = disc_cumsum[m - 1].clamp(min=1e-12)
            ndcg = dcg / idcg

            cum = torch.cumsum(relk, dim=1)
            ranks = torch.arange(1, k + 1, device=device).float()
            prec_at_r = cum / ranks
            ap = (prec_at_r * relk).sum(dim=1) / m.float()
            has = hits > 0
            first = torch.argmax(relk, dim=1).float()
            rr = torch.where(has, 1.0 / (first + 1.0), torch.zeros_like(first))

            agg[k]["precision"] += prec.sum().item()
            agg[k]["recall"] += recl.sum().item()
            agg[k]["f1"] += f1.sum().item()
            agg[k]["hitrate"] += hr.sum().item()
            agg[k]["ndcg"] += ndcg.sum().item()
            agg[k]["map"] += ap.sum().item()
            agg[k]["mrr"] += rr.sum().item()

            recommended = topi[:, :k][validk]
            cat[k][recommended] = True
            rec_cov[k] += int((n_cand >= k).sum().item())
            score_sum[k] += float(topv[:, :k][validk].sum().item())
            score_cnt[k] += int(validk.sum().item())

        n_users += len(batch)

    out = {}
    for k in ks:
        out[k] = {m: agg[k][m] / n_users for m in agg[k]}
        out[k]["catalog_coverage"] = float(cat[k].sum().item()) / num_items
        out[k]["rec_coverage"] = rec_cov[k] / n_users
        out[k]["avg_score"] = (score_sum[k] / score_cnt[k]) if score_cnt[k] else 0.0
        out[k]["n_users"] = n_users
    return out



def embedding_score_provider(user_emb: torch.Tensor, item_emb: torch.Tensor):
    """Closure scoring users by dot product against all items."""
    def provider(user_ids: torch.Tensor) -> torch.Tensor:
        return torch.matmul(user_emb[user_ids], item_emb.t())
    return provider


def popularity_score_provider(pop_vector: torch.Tensor):
    def provider(user_ids: torch.Tensor) -> torch.Tensor:
        return pop_vector.unsqueeze(0).expand(user_ids.shape[0], -1)
    return provider
