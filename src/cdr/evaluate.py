"""Leave-one-out sampled ranking evaluation (HW1 §1.3 protocol).

Cohort: cold-start cross-vertical users = held-out overlap users (unseen by the mapping),
with their target history hidden (scored only through the mapping).
Per user: 1 held-out target test positive + ``eval_n_neg`` sampled unseen negatives,
ranked together. Metrics averaged over users, then over negative-sampling seeds.

Models compared on the identical candidate sets:
  emcdr            : û_t = f(U_src);  score = û_t · V_tgt
  baseline_target_mf : mean target-user vector · V_tgt  (single-vertical MF on a cold user)
  baseline_mostpop : target train popularity            (the deployed "popular this week")
  baseline_random  : random scores                      (floor)
  baseline_feature_transfer : cosine(mean source liked-item text, target item text) [optional]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import VerticalData
from .models import HybridRecommender, MappingMLP
from .utils import log


def _held_out_positives(tgt_vd: VerticalData, tgt_users: np.ndarray) -> dict:
    """One held-out target test positive per user (latest by timestamp)."""
    tp = tgt_vd.test_positives(restrict_users=tgt_users)
    if len(tp) == 0:
        return {}
    last = tp.sort_values("timestamp").groupby("user_idx", as_index=False).tail(1)
    return dict(zip(last["user_idx"].to_numpy(), last["item_idx"].to_numpy()))


def _sample_candidates(tgt_users, held_items, tgt_seen, n_items, n_neg, rng):
    """For each user build [held_item, neg_0, ..., neg_{n_neg-1}] (held at column 0)."""
    cands = np.empty((len(tgt_users), 1 + n_neg), dtype=np.int64)
    for k, u in enumerate(tgt_users):
        seen = tgt_seen.get(u, frozenset())
        negs, need = [], n_neg
        while need > 0:
            draw = rng.integers(0, n_items, size=need * 2 + 8)
            for it in draw:
                if it not in seen:
                    negs.append(it)
                    if len(negs) == n_neg:
                        break
            need = n_neg - len(negs)
        cands[k, 0] = held_items[k]
        cands[k, 1:] = negs[:n_neg]
    return cands


def _rank_metrics(scores: np.ndarray, ks) -> dict:
    """scores: (n_users, C) with the positive at column 0. Strict-greater ranking."""
    pos = scores[:, :1]
    rank = 1 + (scores > pos).sum(axis=1)          # 1 = top
    out = {}
    for k in ks:
        hit = (rank <= k).astype(np.float64)
        out[f"Recall@{k}"] = float(hit.mean())
        ndcg = np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)
        out[f"NDCG@{k}"] = float(ndcg.mean())
    out["HR@10"] = out.get("Recall@10", out[f"Recall@{ks[0]}"])
    return out


def _source_text_profiles(src_vd: VerticalData, src_users: np.ndarray) -> np.ndarray:
    """Mean MiniLM text embedding of each user's source liked items (shared encoder space)."""
    pos = src_vd.train_positives(restrict_users=src_users)
    df = pd.DataFrame(pos, columns=["user_idx", "item_idx"])
    text = src_vd.store.text_emb
    prof = {u: text[g["item_idx"].to_numpy()].mean(0) for u, g in df.groupby("user_idx")}
    out = np.zeros((len(src_users), text.shape[1]), dtype=np.float32)
    for k, u in enumerate(src_users):
        if u in prof:
            out[k] = prof[u]
    return out


def evaluate_pair(f: MappingMLP, src_model: HybridRecommender, tgt_model: HybridRecommender,
                  src_vd: VerticalData, tgt_vd: VerticalData, eval_df: pd.DataFrame,
                  cfg: Config, ablation: str) -> dict:
    device = tgt_model.device
    store = tgt_vd.store
    n_items = tgt_vd.n_items

    # restrict cohort to users that actually have a held-out target test positive
    held = _held_out_positives(tgt_vd, eval_df["tgt_idx"].to_numpy())
    ev = eval_df[eval_df["tgt_idx"].isin(set(held))].reset_index(drop=True)
    if cfg.max_eval_users:
        ev = ev.iloc[:cfg.max_eval_users].reset_index(drop=True)
    if len(ev) == 0:
        raise RuntimeError("no cold-eval users with a held-out target test positive")

    tgt_users = ev["tgt_idx"].to_numpy()
    src_users = ev["src_idx"].to_numpy()
    held_items = np.array([held[u] for u in tgt_users], dtype=np.int64)
    tgt_seen = tgt_vd.user_seen_sets(restrict_users=tgt_users)

    # precompute model ingredients
    V = tgt_model.all_item_embeddings(store).to(device)                     # (n_items, d)
    pop = tgt_vd.item_popularity()                                          # (n_items,)
    with torch.no_grad():
        Us = src_model.user_emb(torch.as_tensor(src_users, dtype=torch.long, device=device))
        mapped = f(Us).detach()                                            # (n_ev, d)
        mean_user = tgt_model.user_emb.weight.mean(0, keepdim=True).detach()  # (1, d)

    want_ft = cfg.run_baselines
    if want_ft:
        prof = torch.from_numpy(_source_text_profiles(src_vd, src_users)).to(device)  # (n_ev,384)
        prof = torch.nn.functional.normalize(prof, dim=-1)
        tgt_text = torch.nn.functional.normalize(
            torch.from_numpy(store.text_emb).to(device), dim=-1)            # (n_items,384)

    models = ["emcdr"]
    if cfg.run_baselines:
        models += ["baseline_target_mf", "baseline_mostpop",
                   "baseline_random", "baseline_feature_transfer"]

    log(f"evaluate [{src_vd.vertical}->{tgt_vd.vertical}/{ablation}] cohort={len(ev)} "
        f"n_neg={cfg.eval_n_neg} seeds={cfg.eval_seeds}")

    per_seed = {m: [] for m in models}
    n_ev = len(ev)
    chunk = 2048
    for seed in cfg.eval_seeds:
        rng = np.random.default_rng(1000 + seed)
        cands = _sample_candidates(tgt_users, held_items, tgt_seen, n_items, cfg.eval_n_neg, rng)
        scores = {m: np.empty((n_ev, cands.shape[1]), dtype=np.float64) for m in models}
        for s in range(0, n_ev, chunk):
            e = min(s + chunk, n_ev)
            c = torch.as_tensor(cands[s:e], dtype=torch.long, device=device)   # (b, C)
            cand_emb = V[c]                                                     # (b, C, d)
            # emcdr
            scores["emcdr"][s:e] = (mapped[s:e].unsqueeze(1) * cand_emb).sum(-1).cpu().numpy()
            if cfg.run_baselines:
                scores["baseline_target_mf"][s:e] = (mean_user.unsqueeze(1) * cand_emb).sum(-1).cpu().numpy()
                scores["baseline_mostpop"][s:e] = pop[cands[s:e]]
                scores["baseline_random"][s:e] = rng.random((e - s, cands.shape[1]))
                ct = torch.as_tensor(cands[s:e], dtype=torch.long, device=device)
                scores["baseline_feature_transfer"][s:e] = (
                    (prof[s:e].unsqueeze(1) * tgt_text[ct]).sum(-1).cpu().numpy())
        for m in models:
            per_seed[m].append(_rank_metrics(scores[m], cfg.eval_ks))

    # aggregate mean/std across seeds
    summary = {}
    metric_names = list(per_seed[models[0]][0].keys())
    for m in models:
        summary[m] = {}
        for name in metric_names:
            vals = np.array([d[name] for d in per_seed[m]], dtype=np.float64)
            summary[m][name] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                "per_seed": vals.tolist()}
    return {"n_eval_users": int(n_ev), "metrics": summary}
