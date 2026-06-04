"""Train one per-vertical hybrid recommender with BPR (pairwise) loss.

In full mode (subsample_users=None) the trained model is cached to disk and reused
across every pair that touches this vertical. In smoke mode it is retrained on the
pair's (subsampled) overlap users so end-to-end validation is fast.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import VerticalData
from .models import HybridRecommender
from .utils import log, save_json

_EMPTY = frozenset()


def _sample_negatives(u, user_seen, n_items, rng, cdf=None, max_tries=3):
    """Per-positive negatives, excluding each user's seen items. cdf=None -> uniform; otherwise
    draw negatives ~ popularity (hard negatives) via inverse-CDF sampling."""
    def draw(size):
        if cdf is None:
            return rng.integers(0, n_items, size=size)
        return np.minimum(np.searchsorted(cdf, rng.random(size)), n_items - 1)

    neg = draw(len(u))
    for _ in range(max_tries):
        bad = np.fromiter(
            (neg[k] in user_seen.get(u[k], _EMPTY) for k in range(len(u))),
            dtype=bool, count=len(u),
        )
        if not bad.any():
            break
        neg[bad] = draw(int(bad.sum()))
    return neg


def train_recommender(vd: VerticalData, cfg: Config, ablation: str,
                      include_users=None, force: bool = False) -> HybridRecommender:
    device = vd.store.device
    rec_dir = cfg.rec_dir(vd.vertical, ablation)
    ckpt = rec_dir / "model.pt"
    model = HybridRecommender(
        vd.n_users, vd.n_items, vd.n_cats, cfg.d, ablation,
        cfg.use_pop, vd.store.text_dim, device,
    ).to(device)

    # cache reuse only when training the full vertical (not a pair-specific subsample)
    if ckpt.exists() and not force and cfg.subsample_users is None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        log(f"loaded cached recommender: {rec_dir}")
        return model

    restrict = include_users if cfg.subsample_users is not None else None
    train_pos = vd.train_positives(restrict_users=restrict)
    if len(train_pos) == 0:
        raise RuntimeError(f"no train positives for {vd.vertical} (restrict={restrict is not None})")
    user_seen = vd.user_seen_sets(restrict_users=restrict)

    cdf = None
    if cfg.neg_train == "popularity":
        w = vd.item_popularity() ** cfg.neg_train_power
        tot = float(w.sum())
        cdf = np.cumsum(w / tot) if tot > 0 else None

    opt = torch.optim.Adam(model.parameters(), lr=cfg.rec_lr, weight_decay=cfg.rec_weight_decay)
    rng = np.random.default_rng(cfg.seed)
    N, bs = len(train_pos), cfg.rec_batch_size
    log(f"train recommender [{vd.vertical}/{ablation}] users~{len(user_seen)} pos={N} "
        f"items={vd.n_items} d={cfg.d} neg_train={cfg.neg_train} device={device}")

    model.train()
    for epoch in range(cfg.rec_epochs):
        perm = rng.permutation(N)
        n_batches = (N + bs - 1) // bs
        if cfg.rec_steps_per_epoch:
            n_batches = min(n_batches, cfg.rec_steps_per_epoch)
        total = 0.0
        for b in range(n_batches):
            sel = perm[b * bs:(b + 1) * bs]
            if len(sel) == 0:
                break
            u, ip = train_pos[sel, 0], train_pos[sel, 1]
            ineg = _sample_negatives(u, user_seen, vd.n_items, rng, cdf)
            s_pos = model.score(u, ip, vd.store)
            s_neg = model.score(u, ineg, vd.store)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        log(f"  [{vd.vertical}/{ablation}] epoch {epoch + 1}/{cfg.rec_epochs} "
            f"bpr_loss={total / max(n_batches, 1):.4f}")

    rec_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    save_json(
        {"vertical": vd.vertical, "ablation": ablation, "d": cfg.d,
         "n_users": vd.n_users, "n_items": vd.n_items, "n_cats": vd.n_cats,
         "use_pop": model.use_pop, "tag": cfg.tag, "n_train_pos": int(N)},
        rec_dir / "meta.json",
    )
    return model
