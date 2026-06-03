"""History-grounded cross-domain bridge (the mapping redesign).

Replaces the old free-embedding EMCDR mapping. Design decisions (made with the user):
  - Source-user rep = learned POOL (mean or attention) over the embeddings of the user's
    source items -> a function of history, so it (a) enables sub-history augmentation,
    (b) is robust for thin/cold users, (c) has no fragile free U_src.
  - Bridge trained END-TO-END on the target ranking task (BPR over real target interactions),
    NOT by regressing onto a noisy learned U_tgt.
  - Sub-history augmentation: each overlap user yields many examples per epoch.
  - Semi-supervised CORAL alignment: the large source-only / target-only populations pull the
    bridge's output distribution toward the real target-user distribution (helps thin overlap).
  - Validation-based early stopping + best-on-val model selection (per-user temporal hold-out).

Reuses the per-vertical recommenders only for their (frozen) item embeddings, and the existing
eval helpers (`_sample_candidates`, `_rank_metrics`, `_source_text_profiles`).

NOTE: written but not yet execution-tested (PyTorch was iCloud-evicted at authoring time).
Run once `.venv` is restored:
  PYTHONPATH=src python -m cdr.bridge --config configs/bridge_smoke.yaml
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import PairData, VerticalData, align_overlap
from .evaluate import _rank_metrics, _sample_candidates, _source_text_profiles
from .train_recommender import train_recommender
from .utils import log, save_json, select_device, set_seed


# ------------------------------------------------------------------------- models
class SourceEncoder(nn.Module):
    """Pool a user's item embeddings into one vector. mean (parameter-free) or attention.

    Ragged sets -> flat gather + segment reduction via index_add_ (MPS-safe). Used for both the
    source-history input and (in the semi-supervised term) the target-history rep.
    """

    def __init__(self, d: int, mode: str = "mean"):
        super().__init__()
        self.d, self.mode = d, mode
        if mode == "attention":
            self.attn = nn.Linear(d, 1)

    def _flat_seg(self, hist_lists, device):
        B = len(hist_lists)
        lens = np.fromiter((len(h) for h in hist_lists), dtype=np.int64, count=B)
        if lens.sum() == 0:
            return None, None, B
        flat = np.concatenate([h for h in hist_lists if len(h)])
        seg = np.repeat(np.arange(B), lens)
        return (torch.as_tensor(flat, dtype=torch.long, device=device),
                torch.as_tensor(seg, dtype=torch.long, device=device), B)

    def forward(self, V: torch.Tensor, hist_lists, device) -> torch.Tensor:
        flat, seg, B = self._flat_seg(hist_lists, device)
        if flat is None:
            return torch.zeros(B, self.d, device=device)
        vecs = V[flat]                                                   # (total, d)
        if self.mode == "attention":
            s = self.attn(vecs).squeeze(-1)
            s = s - s.max()                                             # global shift for stability
            w = s.exp()
            denom = torch.zeros(B, device=device).index_add_(0, seg, w)
            w = w / denom[seg].clamp(min=1e-9)                          # segment-softmax
            return torch.zeros(B, self.d, device=device).index_add_(0, seg, vecs * w.unsqueeze(1))
        agg = torch.zeros(B, self.d, device=device).index_add_(0, seg, vecs)
        cnt = torch.zeros(B, device=device).index_add_(0, seg, torch.ones(seg.shape[0], device=device))
        return agg / cnt.clamp(min=1).unsqueeze(1)


class Bridge(nn.Module):
    """MLP from the pooled source-user rep into target user-space (optional low-dim bottleneck)."""

    def __init__(self, d: int, hidden, dropout: float, bottleneck=None):
        super().__init__()
        layers, prev = [], d
        for h in ([bottleneck] if bottleneck else []) + list(hidden):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def coral(ms: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """CORAL domain-alignment: match mean + covariance of two reps (semi-supervised term)."""
    d = ms.shape[1]
    dm = ms.mean(0) - ts.mean(0)
    mc, tc = ms - ms.mean(0, keepdim=True), ts - ts.mean(0, keepdim=True)
    cov_m = mc.t() @ mc / max(ms.shape[0] - 1, 1)
    cov_t = tc.t() @ tc / max(ts.shape[0] - 1, 1)
    return (dm * dm).sum() + ((cov_m - cov_t) ** 2).sum() / (4 * d * d)


# ------------------------------------------------------------------------- data helpers
def _positives(vd: VerticalData, users):
    m = vd.inter["label"] == 1
    df = vd.inter.loc[m, ["user_idx", "item_idx", "timestamp"]]
    return df[df["user_idx"].isin(np.asarray(users))]


def history_map(vd: VerticalData, users) -> dict:
    """user_idx -> np.array(positive item_idx) — used for both source histories and target reps."""
    return {u: g["item_idx"].to_numpy() for u, g in _positives(vd, users).groupby("user_idx")}


def target_loo_test(vd: VerticalData, users) -> dict:
    """user_idx -> latest target positive item_idx (per-user temporal hold-out for eval/val)."""
    df = _positives(vd, users).sort_values(["user_idx", "timestamp"])
    last = df.groupby("user_idx", as_index=False).tail(1)
    return dict(zip(last["user_idx"].to_numpy(), last["item_idx"].to_numpy()))


def role_user_idx(pair: PairData, vd: VerticalData, role: str) -> np.ndarray:
    """Per-vertical user_idx for users in a given pair role (source_only / target_only)."""
    ru = pair.roles.loc[pair.roles["role"] == role, "user_idx"].to_numpy()
    uid = pair.pair_user_map.set_index("user_idx")["user_id"].reindex(ru)
    m = vd.user_map.set_index("user_id")["user_idx"]
    return uid.map(m).dropna().astype("int64").to_numpy()


def _subsample_hist(h, rng, cfg):
    if cfg.aug_k <= 0 or len(h) <= 1:
        return h
    k = max(1, int(round(len(h) * rng.uniform(cfg.aug_min_frac, 1.0))))
    return rng.choice(h, size=k, replace=False)


def _sample_neg(seen, n_items, rng):
    while True:
        c = int(rng.integers(0, n_items))
        if c not in seen:
            return c


# ------------------------------------------------------------------------- validation
def _val_recall(cfg, enc, bridge, V_src, V_tgt, val_users, src_hist, val_latest,
                tgt_seen, n_items, device, k=10):
    enc.eval()
    bridge.eval()
    rng = np.random.default_rng(777)                                   # fixed: comparable across epochs
    tgt_users = np.array([t for _, t in val_users], dtype=np.int64)
    held = np.array([val_latest[t] for _, t in val_users], dtype=np.int64)
    with torch.no_grad():
        mapped = bridge(enc(V_src, [src_hist[s] for s, _ in val_users], device))
    cands = _sample_candidates(tgt_users, held, tgt_seen, n_items, cfg.eval_n_neg, rng)
    hit = 0
    for s in range(0, len(val_users), 2048):
        e = min(s + 2048, len(val_users))
        c = torch.as_tensor(cands[s:e], dtype=torch.long, device=device)
        sc = (mapped[s:e].unsqueeze(1) * V_tgt[c]).sum(-1).cpu().numpy()
        hit += int(((1 + (sc > sc[:, :1]).sum(1)) <= k).sum())
    return hit / max(len(val_users), 1)


# ------------------------------------------------------------------------- orchestration
def run_bridge_pair(cfg: Config, src: str, tgt: str) -> dict:
    device = select_device(cfg.device)
    set_seed(cfg.seed)
    ablation = cfg.ablation

    src_vd = VerticalData(cfg.data_root, src, device)
    tgt_vd = VerticalData(cfg.data_root, tgt, device)
    pair = PairData(cfg.data_root, src, tgt)
    align = align_overlap(pair, src_vd, tgt_vd)
    if cfg.subsample_users:
        align = align.head(cfg.subsample_users).reset_index(drop=True)
    log(f"bridge {src}->{tgt} aligned overlap={len(align)} (meta {pair.meta.get('overlap')})")

    # recommenders -> frozen item embeddings
    src_model = train_recommender(src_vd, cfg, ablation, include_users=align["src_idx"].unique())
    tgt_model = train_recommender(tgt_vd, cfg, ablation, include_users=align["tgt_idx"].unique())
    with torch.no_grad():
        V_src = src_model.all_item_embeddings(src_vd.store).to(device)
        V_tgt = tgt_model.all_item_embeddings(tgt_vd.store).to(device)

    # split overlap users -> fit / val (early stop) / cold-eval (all disjoint)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(align))
    n_tr = int(cfg.map_train_frac * len(align))
    tr_idx, ev_idx = perm[:n_tr], perm[n_tr:]
    use_val = bool(cfg.early_stop_patience) and len(tr_idx) > 50
    n_val = int(0.1 * len(tr_idx)) if use_val else 0
    fit_df = align.iloc[tr_idx[n_val:]].reset_index(drop=True)
    val_df = align.iloc[tr_idx[:n_val]].reset_index(drop=True)
    ev_df = align.iloc[ev_idx].reset_index(drop=True)

    src_hist = history_map(src_vd, align["src_idx"].unique())
    tgt_pos = history_map(tgt_vd, fit_df["tgt_idx"].unique())
    tgt_seen = tgt_vd.user_seen_sets(restrict_users=align["tgt_idx"].unique())
    n_items_tgt = tgt_vd.n_items
    fit_users = [(int(r.src_idx), int(r.tgt_idx)) for r in fit_df.itertuples()
                 if r.src_idx in src_hist and r.tgt_idx in tgt_pos]
    val_latest = target_loo_test(tgt_vd, val_df["tgt_idx"].unique()) if n_val else {}
    val_users = [(int(r.src_idx), int(r.tgt_idx)) for r in val_df.itertuples()
                 if r.src_idx in src_hist and r.tgt_idx in val_latest]
    log(f"bridge fit={len(fit_users)} val={len(val_users)} cold_eval~{len(ev_df)}")

    # semi-supervised pools (non-overlap users), capped + cached
    semisup = cfg.semisup_weight > 0
    so_hist = to_hist = so_keys = to_keys = None
    if semisup:
        so = role_user_idx(pair, src_vd, "source_only")
        to = role_user_idx(pair, tgt_vd, "target_only")
        rng.shuffle(so)
        rng.shuffle(to)
        so_hist = history_map(src_vd, so[:cfg.semisup_pool_cap])
        to_hist = history_map(tgt_vd, to[:cfg.semisup_pool_cap])
        so_keys = np.array(list(so_hist.keys()))
        to_keys = np.array(list(to_hist.keys()))
        log(f"semisup pools: source_only={len(so_keys)} target_only={len(to_keys)} w={cfg.semisup_weight}")

    enc = SourceEncoder(cfg.d, cfg.pool).to(device)
    bridge = Bridge(cfg.d, cfg.bridge_hidden, cfg.bridge_dropout, cfg.bridge_bottleneck).to(device)
    opt = torch.optim.Adam(list(bridge.parameters()) + list(enc.parameters()),
                           lr=cfg.bridge_lr, weight_decay=cfg.bridge_weight_decay)
    bs = cfg.bridge_batch
    best_val, best_state, bad = -1.0, None, 0

    for epoch in range(cfg.bridge_epochs):
        enc.train()
        bridge.train()
        rng.shuffle(fit_users)
        total, nb = 0.0, 0
        for b in range(0, len(fit_users), bs):
            batch = fit_users[b:b + bs]
            hist = [_subsample_hist(src_hist[s], rng, cfg) for s, _ in batch]
            mapped = bridge(enc(V_src, hist, device))
            pos = np.fromiter((rng.choice(tgt_pos[t]) for _, t in batch), np.int64, len(batch))
            neg = np.fromiter((_sample_neg(tgt_seen.get(t, frozenset()), n_items_tgt, rng)
                               for _, t in batch), np.int64, len(batch))
            sp = (mapped * V_tgt[torch.as_tensor(pos, device=device)]).sum(-1)
            sn = (mapped * V_tgt[torch.as_tensor(neg, device=device)]).sum(-1)
            loss = -F.logsigmoid(sp - sn).mean()
            if semisup and len(so_keys) > 1 and len(to_keys) > 1:
                si = rng.choice(so_keys, size=min(cfg.semisup_batch, len(so_keys)), replace=False)
                ti = rng.choice(to_keys, size=min(cfg.semisup_batch, len(to_keys)), replace=False)
                m_rep = bridge(enc(V_src, [so_hist[u] for u in si], device))
                t_rep = enc(V_tgt, [to_hist[u] for u in ti], device)   # target-user rep in V_tgt space
                loss = loss + cfg.semisup_weight * coral(m_rep, t_rep)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        msg = f"  [bridge {src}->{tgt}/{ablation}] epoch {epoch + 1}/{cfg.bridge_epochs} loss={total / max(nb, 1):.4f}"
        if val_users:
            vr = _val_recall(cfg, enc, bridge, V_src, V_tgt, val_users, src_hist,
                             val_latest, tgt_seen, n_items_tgt, device)
            msg += f" val_recall@10={vr:.4f}"
            if vr > best_val:
                best_val, bad = vr, 0
                best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(bridge.state_dict()))
            else:
                bad += 1
        if val_users or (epoch + 1) % max(cfg.bridge_epochs // 8, 1) == 0:
            log(msg)
        if cfg.early_stop_patience and bad >= cfg.early_stop_patience:
            log(f"  early stop @ epoch {epoch + 1} (best val_recall@10={best_val:.4f})")
            break

    if best_state is not None:                                          # restore best-on-val
        enc.load_state_dict(best_state[0])
        bridge.load_state_dict(best_state[1])

    res = _evaluate(cfg, enc, bridge, V_src, V_tgt, src_vd, tgt_vd, ev_df, src_hist, tgt_seen)
    out = {"src": src, "tgt": tgt, "ablation": ablation, "arch": "history_bridge",
           "overlap_aligned": int(len(align)),
           "best_val_recall@10": (best_val if best_state is not None else None),
           "config": cfg.to_dict(), **res}
    save_json(out, cfg.bridge_result_path(src, tgt, ablation))
    log(f"wrote {cfg.bridge_result_path(src, tgt, ablation)}")
    return out


def _evaluate(cfg, enc, bridge, V_src, V_tgt, src_vd, tgt_vd, ev_df, src_hist, tgt_seen) -> dict:
    device = V_tgt.device
    n_items = tgt_vd.n_items
    test_item = target_loo_test(tgt_vd, ev_df["tgt_idx"].unique())
    ev = [(int(r.src_idx), int(r.tgt_idx)) for r in ev_df.itertuples()
          if r.src_idx in src_hist and r.tgt_idx in test_item]
    if cfg.max_eval_users:
        ev = ev[:cfg.max_eval_users]
    if not ev:
        raise RuntimeError("no cold-eval users with source history + target test positive")

    tgt_users = np.array([t for _, t in ev], dtype=np.int64)
    src_users = np.array([s for s, _ in ev], dtype=np.int64)
    held = np.array([test_item[t] for t in tgt_users], dtype=np.int64)

    enc.eval()
    bridge.eval()
    with torch.no_grad():
        mapped = bridge(enc(V_src, [src_hist[s] for s in src_users], device))

    pop = tgt_vd.item_popularity()
    models = ["bridge"]
    if cfg.run_baselines:
        models += ["baseline_mostpop", "baseline_random", "baseline_feature_transfer"]
        prof = F.normalize(torch.from_numpy(_source_text_profiles(src_vd, src_users)).to(device), dim=-1)
        tgt_text = F.normalize(torch.from_numpy(tgt_vd.store.text_emb).to(device), dim=-1)

    log(f"evaluate bridge [{src_vd.vertical}->{tgt_vd.vertical}] cohort={len(ev)} "
        f"n_neg={cfg.eval_n_neg} seeds={cfg.eval_seeds}")
    per_seed = {m: [] for m in models}
    n_ev = len(ev)
    for seed in cfg.eval_seeds:
        rng = np.random.default_rng(1000 + seed)
        cands = _sample_candidates(tgt_users, held, tgt_seen, n_items, cfg.eval_n_neg, rng)
        scores = {m: np.empty((n_ev, cands.shape[1])) for m in models}
        for s in range(0, n_ev, 2048):
            e = min(s + 2048, n_ev)
            c = torch.as_tensor(cands[s:e], dtype=torch.long, device=device)
            cand_emb = V_tgt[c]
            scores["bridge"][s:e] = (mapped[s:e].unsqueeze(1) * cand_emb).sum(-1).cpu().numpy()
            if cfg.run_baselines:
                scores["baseline_mostpop"][s:e] = pop[cands[s:e]]
                scores["baseline_random"][s:e] = rng.random((e - s, cands.shape[1]))
                scores["baseline_feature_transfer"][s:e] = (
                    (prof[s:e].unsqueeze(1) * tgt_text[c]).sum(-1).cpu().numpy())
        for m in models:
            per_seed[m].append(_rank_metrics(scores[m], cfg.eval_ks))

    summary, names = {}, list(per_seed[models[0]][0].keys())
    for m in models:
        summary[m] = {}
        for nm in names:
            vals = np.array([d[nm] for d in per_seed[m]])
            summary[m][nm] = {"mean": float(vals.mean()), "std": float(vals.std()),
                              "per_seed": vals.tolist()}
    return {"n_eval_users": int(n_ev), "metrics": summary}


def print_summary(out: dict) -> None:
    metrics = out["metrics"]
    names = list(next(iter(metrics.values())).keys())
    bv = out.get("best_val_recall@10")
    print(f"\n=== [bridge] {out['src']} -> {out['tgt']}  [{out['ablation']}]  "
          f"cohort={out['n_eval_users']}  best_val@10={bv} ===")
    hdr = "model".ljust(28) + "".join(n.ljust(16) for n in names)
    print(hdr)
    print("-" * len(hdr))
    for m, md in metrics.items():
        print(m.ljust(28) + "".join(f"{md[n]['mean']:.4f}±{md[n]['std']:.3f}".ljust(16) for n in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--tgt", default=None)
    a = ap.parse_args()
    cfg = Config.from_yaml(a.config)
    pairs = [(a.src, a.tgt)] if (a.src and a.tgt) else [tuple(p) for p in cfg.pairs]

    results = []
    for s, t in pairs:
        out = run_bridge_pair(cfg, s, t)
        print_summary(out)
        results.append({"src": s, "tgt": t, "n_eval_users": out["n_eval_users"],
                        "best_val_recall@10": out.get("best_val_recall@10"), "metrics": out["metrics"]})
    if len(pairs) > 1:
        from pathlib import Path
        grid = Path(cfg.results_root) / f"BRIDGE_GRID__{cfg.tag}.json"
        save_json({"pairs": [list(p) for p in pairs], "results": results}, grid)
        log(f"wrote {grid}")


if __name__ == "__main__":
    main()
