"""Models: hybrid two-tower recommender + EMCDR mapping MLP.

Item tower (hybrid):  item_emb = id_factor + W_text·text_emb + cat_bag(categories) [+ pop_proj]
Ablation toggles the content branches; ``id`` is always present.
  full      = id + text + cat (+ pop)
  cats_only = id + cat (+ pop)
  text_only = id + text (+ pop)
  id_only   = id            (pure MF baseline; pop also dropped)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class HybridRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_cats: int, d: int,
                 ablation: str, use_pop: bool, text_dim: int, device: torch.device):
        super().__init__()
        self.d = d
        self.device = device
        self.ablation = ablation
        self.use_text = ablation in ("full", "text_only")
        self.use_cat = ablation in ("full", "cats_only")
        self.use_pop = bool(use_pop) and ablation != "id_only"

        self.user_emb = nn.Embedding(n_users, d)
        self.id_factor = nn.Embedding(n_items, d)
        if self.use_text:
            self.text_proj = nn.Linear(text_dim, d)
        if self.use_cat:
            self.cat_table = nn.Embedding(n_cats, d)
        if self.use_pop:
            self.pop_proj = nn.Linear(2, d)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.id_factor.weight, std=0.01)
        if self.use_cat:
            nn.init.normal_(self.cat_table.weight, std=0.01)

    # --- item tower ---
    def item_embedding(self, item_idx_np, store) -> torch.Tensor:
        idx_np = np.asarray(item_idx_np)
        idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
        e = self.id_factor(idx_t)
        if self.use_text:
            e = e + self.text_proj(store.text(idx_np))
        if self.use_cat:
            flat, seg, B = store.cat_segments(idx_np)
            agg = torch.zeros(B, self.d, device=self.device)
            if flat.numel() > 0:
                agg.index_add_(0, seg, self.cat_table(flat))
            e = e + agg
        if self.use_pop:
            e = e + self.pop_proj(store.pop(idx_np))
        return e

    # --- user tower ---
    def user_vec(self, user_idx_np) -> torch.Tensor:
        u = torch.as_tensor(np.asarray(user_idx_np), dtype=torch.long, device=self.device)
        return self.user_emb(u)

    def score(self, user_idx_np, item_idx_np, store) -> torch.Tensor:
        return (self.user_vec(user_idx_np) * self.item_embedding(item_idx_np, store)).sum(-1)

    @torch.no_grad()
    def all_item_embeddings(self, store, batch: int = 20000) -> torch.Tensor:
        """Materialize V = item embeddings for every item (for eval/inference)."""
        n = self.id_factor.num_embeddings
        out = []
        for s in range(0, n, batch):
            idx = np.arange(s, min(s + batch, n))
            out.append(self.item_embedding(idx, store).detach())
        return torch.cat(out, 0)


class MappingMLP(nn.Module):
    """EMCDR mapping f: source user space -> target user space."""

    def __init__(self, d: int, hidden, dropout: float):
        super().__init__()
        dims = [d] + list(hidden) + [d]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
