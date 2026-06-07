"""C2 — residual content-conditioned bridge for the thin Video-Games pairs.

Why: on the remaining pairs where content-transfer still beats EMCDR (Movies/Toys -> Video Games),
content is the winning signal but EMCDR only injects content on the *item* side (B1, content-grounded
V_tgt). The *user* side is still a pure-collaborative U_src. C2 closes the loop: it feeds the user's
source content profile into the bridge and keeps the content-cosine as an explicit floor term, then
learns a collaborative residual on top — trained on ranking, so the two terms are jointly calibrated
(unlike the C1 score-blend, which destructively interfered).

    score(u, i) = content_scale * cos(profile_src(u), text_tgt(i))        # content floor (= feat-transfer)
                + ( Linear[ U_src(u) ; profile_src(u) ] ) · V_tgt(i)       # learned collaborative residual

Trained with BPR on overlap-TRAIN users' target positives (overlap users have target history);
evaluated on the held-out cold-eval overlap slice with the SAME split / candidates / seeds as
evaluate_pair, so C2 / content / EMCDR(B1) numbers are directly comparable. Reuses cached
recommenders (full sources + the B1 Video_Games target via rec_tag_overrides).

  PYTHONPATH=src python -m cdr.run_c2 --config configs/improve_c2.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .data import PairData, VerticalData, align_overlap
from .evaluate import (_held_out_positives, _rank_metrics, _sample_candidates,
                       _source_text_profiles)
from .train_recommender import train_recommender
from .utils import log, save_json, select_device, set_seed


class C2Bridge(nn.Module):
    """Content floor + learned collaborative residual. Zero-init residual -> starts at content."""

    def __init__(self, d: int, text_dim: int):
        super().__init__()
        self.map = nn.Linear(d + text_dim, d)
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)
        self.content_scale = nn.Parameter(torch.tensor(1.0))

    def mapped(self, u_src: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.map(torch.cat([u_src, profile], dim=-1))


def _norm_profiles(src_vd, users, device) -> torch.Tensor:
    prof = _source_text_profiles(src_vd, users)                 # (N, 384) mean source liked-item text
    return F.normalize(torch.from_numpy(prof).to(device), dim=-1)


def run_c2_pair(cfg: Config, src: str, tgt: str, src_vd: VerticalData,
                tgt_vd: VerticalData, device) -> dict:
    # cached recommenders: full source, B1 content-grounded target (via rec_tag_overrides)
    src_model = train_recommender(src_vd, cfg, "full")
    tgt_model = train_recommender(tgt_vd, cfg, "full")
    store = tgt_vd.store
    n_items = tgt_vd.n_items

    # overlap split — identical to train_mapping so the cold-eval cohort matches the EMCDR/B1 run
    pair = PairData(cfg.data_root, src, tgt)
    align = align_overlap(pair, src_vd, tgt_vd)
    n = len(align)
    perm = np.random.default_rng(cfg.seed).permutation(n)
    n_train = int(cfg.map_train_frac * n)
    tr_df = align.iloc[perm[:n_train]].reset_index(drop=True)
    ev_df = align.iloc[perm[n_train:]].reset_index(drop=True)

    V = tgt_model.all_item_embeddings(store).to(device)                       # (n_items, d), B1-grounded
    tgt_text = F.normalize(torch.from_numpy(store.text_emb).to(device), dim=-1)  # (n_items, 384)

    # ---- train the residual bridge on overlap-train users' target positives (BPR) ----
    tr_src = tr_df["src_idx"].to_numpy()
    tr_tgt = tr_df["tgt_idx"].to_numpy()
    prof_tr = _norm_profiles(src_vd, tr_src, device)                          # (Ntr, 384)
    with torch.no_grad():
        u_tr = src_model.user_emb(torch.as_tensor(tr_src, dtype=torch.long, device=device)).detach()
    row_of = {int(u): i for i, u in enumerate(tr_tgt)}

    tpos = tgt_vd.train_positives(restrict_users=tr_tgt)                      # (M, 2): user_idx, item_idx
    if len(tpos) == 0:
        raise RuntimeError(f"no overlap-train target positives for {src}->{tgt}")
    pos_user, pos_item = tpos[:, 0], tpos[:, 1]
    rows = np.fromiter((row_of[int(u)] for u in pos_user), dtype=np.int64, count=len(pos_user))

    bridge = C2Bridge(cfg.d, store.text_dim).to(device)
    opt = torch.optim.Adam(bridge.parameters(), lr=cfg.c2_lr, weight_decay=cfg.c2_l2)
    M, bs = len(rows), 4096
    rng = np.random.default_rng(cfg.seed)
    log(f"train C2 [{src}->{tgt}] overlap_train={len(tr_df)} target_pos={M} cold_eval={len(ev_df)}")
    bridge.train()
    for epoch in range(cfg.c2_epochs):
        p = rng.permutation(M)
        total, nb = 0.0, 0
        for b in range(0, M, bs):
            sel = p[b:b + bs]
            r = torch.as_tensor(rows[sel], dtype=torch.long, device=device)
            ip = torch.as_tensor(pos_item[sel], dtype=torch.long, device=device)
            ineg = torch.as_tensor(rng.integers(0, n_items, size=len(sel)), dtype=torch.long, device=device)
            U, P = u_tr[r], prof_tr[r]
            mapped = bridge.mapped(U, P)
            cs = bridge.content_scale
            s_pos = cs * (P * tgt_text[ip]).sum(-1) + (mapped * V[ip]).sum(-1)
            s_neg = cs * (P * tgt_text[ineg]).sum(-1) + (mapped * V[ineg]).sum(-1)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % max(cfg.c2_epochs // 5, 1) == 0:
            log(f"  [C2 {src}->{tgt}] epoch {epoch + 1}/{cfg.c2_epochs} "
                f"bpr={total / max(nb, 1):.4f} content_scale={float(bridge.content_scale):.3f}")

    # ---- evaluate on the cold-eval slice (same protocol/candidates as evaluate_pair) ----
    held = _held_out_positives(tgt_vd, ev_df["tgt_idx"].to_numpy())
    evf = ev_df[ev_df["tgt_idx"].isin(set(held))].reset_index(drop=True)
    tgt_users = evf["tgt_idx"].to_numpy()
    src_users = evf["src_idx"].to_numpy()
    held_items = np.array([held[u] for u in tgt_users], dtype=np.int64)
    tgt_seen = tgt_vd.user_seen_sets(restrict_users=tgt_users)
    prof_ev = _norm_profiles(src_vd, src_users, device)
    bridge.eval()
    with torch.no_grad():
        u_ev = src_model.user_emb(torch.as_tensor(src_users, dtype=torch.long, device=device))
        mapped_ev = bridge.mapped(u_ev, prof_ev).detach()
        cs = float(bridge.content_scale.detach())
        # warm-oracle: rank with the user's REAL target embedding (knows their target taste).
        # This is the ceiling for any cold-start collaborative method — if it loses to content,
        # no mapping/bridge can ever beat content on this pair.
        u_tgt_ev = tgt_model.user_emb(torch.as_tensor(tgt_users, dtype=torch.long, device=device)).detach()

    models_out = ["c2", "content", "c2_resid_only", "warm_oracle"]
    per_seed = {m: [] for m in models_out}
    n_ev, chunk = len(evf), 2048
    for seed in cfg.eval_seeds:
        rng_s = np.random.default_rng(1000 + seed)
        cands = _sample_candidates(tgt_users, held_items, tgt_seen, n_items, cfg.eval_n_neg, rng_s)
        sc = {m: np.empty((n_ev, cands.shape[1]), dtype=np.float64) for m in models_out}
        for s in range(0, n_ev, chunk):
            e = min(s + chunk, n_ev)
            c = torch.as_tensor(cands[s:e], dtype=torch.long, device=device)
            cont = (prof_ev[s:e].unsqueeze(1) * tgt_text[c]).sum(-1)
            resid = (mapped_ev[s:e].unsqueeze(1) * V[c]).sum(-1)
            sc["content"][s:e] = cont.cpu().numpy()
            sc["c2_resid_only"][s:e] = resid.cpu().numpy()
            sc["c2"][s:e] = (cs * cont + resid).cpu().numpy()
            sc["warm_oracle"][s:e] = (u_tgt_ev[s:e].unsqueeze(1) * V[c]).sum(-1).cpu().numpy()
        for m in models_out:
            per_seed[m].append(_rank_metrics(sc[m], cfg.eval_ks))

    summary = {}
    names = list(per_seed["c2"][0].keys())
    for m in models_out:
        summary[m] = {nm: {"mean": float(np.mean([d[nm] for d in per_seed[m]])),
                           "std": float(np.std([d[nm] for d in per_seed[m]]))} for nm in names}
    out = {"src": src, "tgt": tgt, "ablation": "full", "n_eval_users": int(n_ev),
           "content_scale": cs, "metrics": summary}
    save_json(out, cfg.result_path(src, tgt, "full"))
    log(f"wrote {cfg.result_path(src, tgt, 'full')}")
    return out


def _b1_emcdr(src: str, tgt: str) -> float | None:
    p = Path("results") / f"{src}__{tgt}__full__b1.json"
    if not p.exists():
        return None
    return json.load(open(p))["metrics"]["emcdr"]["Recall@10"]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = Config.from_yaml(a.config)
    device = select_device(cfg.device)
    set_seed(cfg.seed)

    vds: dict[str, VerticalData] = {}
    rows = []
    for src, tgt in cfg.pairs:
        for v in (src, tgt):
            vds.setdefault(v, VerticalData(cfg.data_root, v, device))
        out = run_c2_pair(cfg, src, tgt, vds[src], vds[tgt], device)
        m = out["metrics"]
        rows.append((src, tgt, m["content"]["Recall@10"]["mean"],
                     m["c2"]["Recall@10"]["mean"], m["warm_oracle"]["Recall@10"]["mean"],
                     _b1_emcdr(src, tgt), out["content_scale"]))

    print("\n=== C2 on Video-Games pairs — Recall@10 ===")
    hdr = (f"{'pair':26s}{'content':>9s}{'C2':>9s}{'EMCDR(B1)':>11s}{'warm-oracle':>12s}"
           f"{'c_scale':>9s}  collab can win?")
    print(hdr); print("-" * len(hdr))
    for src, tgt, cont, c2, oracle, b1, cscale in rows:
        b1s = f"{b1:.3f}" if b1 is not None else "  n/a"
        verdict = "YES (oracle≥content)" if oracle >= cont else "NO  (ceiling<content)"
        print(f"{src[:3]+'→'+tgt[:9]:26s}{cont:>9.3f}{c2:>9.3f}{b1s:>11s}{oracle:>12.3f}{cscale:>9.2f}  {verdict}")


if __name__ == "__main__":
    main()
