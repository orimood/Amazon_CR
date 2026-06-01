"""Train the EMCDR mapping f: source user space -> target user space.

Supervised on overlap users' aligned (U_src, U_tgt) factors. Overlap users are split
into mapping-train / cold-eval so the evaluation cohort is unseen by the mapping.
Returns the trained mapping and the cold-eval slice of the alignment table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .models import HybridRecommender, MappingMLP
from .utils import log


def train_mapping(src_model: HybridRecommender, tgt_model: HybridRecommender,
                  align_df: pd.DataFrame, cfg: Config, src: str, tgt: str,
                  ablation: str):
    device = src_model.device
    src_idx = align_df["src_idx"].to_numpy()
    tgt_idx = align_df["tgt_idx"].to_numpy()
    with torch.no_grad():
        Us = src_model.user_emb(torch.as_tensor(src_idx, dtype=torch.long, device=device)).detach()
        Ut = tgt_model.user_emb(torch.as_tensor(tgt_idx, dtype=torch.long, device=device)).detach()

    n = len(align_df)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_train = int(cfg.map_train_frac * n)
    tr, ev = perm[:n_train], perm[n_train:]
    Xtr, Ytr = Us[tr], Ut[tr]

    f = MappingMLP(cfg.d, cfg.map_hidden, cfg.map_dropout).to(device)
    opt = torch.optim.Adam(f.parameters(), lr=cfg.map_lr, weight_decay=cfg.map_weight_decay)
    bs = cfg.map_batch_size
    log(f"train mapping [{src}->{tgt}/{ablation}] overlap={n} map_train={len(tr)} cold_eval={len(ev)}")

    f.train()
    for epoch in range(cfg.map_epochs):
        p = rng.permutation(len(tr))
        total, nb = 0.0, 0
        for b in range(0, len(tr), bs):
            sel = p[b:b + bs]
            x, y = Xtr[sel], Ytr[sel]
            pred = f(x)
            loss = F.mse_loss(pred, y) + cfg.map_cos_weight * (1 - F.cosine_similarity(pred, y, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % max(cfg.map_epochs // 5, 1) == 0:
            log(f"  [map {src}->{tgt}/{ablation}] epoch {epoch + 1}/{cfg.map_epochs} loss={total / max(nb, 1):.4f}")

    mdir = cfg.map_dir(src, tgt, ablation)
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(f.state_dict(), mdir / "model.pt")

    eval_df = align_df.iloc[ev].reset_index(drop=True)
    return f, eval_df
