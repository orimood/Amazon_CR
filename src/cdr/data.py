"""Data layer: load Stage A per-vertical artifacts + Stage B pair bridges.

Schemas (confirmed against data/processed/DATASET_DESCRIPTION.md):
  {vertical}/interactions.parquet : user_idx,item_idx,label,weight,rating,timestamp,split
  {vertical}/item_features.npz    : text_emb(n,384), cat_indptr(n+1), cat_indices(nnz),
                                    cat_shape=[n_items,n_cats], avg_rating(n), rating_number_z(n), ...
  {vertical}/seen.parquet         : user_idx,item_idx  (all kept real interactions)
  {vertical}/id_maps/user.parquet : user_id,user_idx   (per-vertical contiguous)
  pairs/{SRC}__{TGT}/roles.parquet        : user_idx(pair-global), role
  pairs/{SRC}__{TGT}/id_maps/user.parquet : user_id,user_idx(pair-global)

The pair-global user_idx differs from per-vertical user_idx; the two are bridged by
``user_id`` (the same person across verticals by construction). ``align_overlap`` does that join.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class ItemFeatureStore:
    """Per-vertical item features + a device-agnostic category bag.

    EmbeddingBag is unimplemented on MPS, so category aggregation is done with a
    learned table (in the model) + ``index_add_`` over the segments returned here.
    """

    def __init__(self, npz_path: str | Path, device: torch.device):
        z = np.load(npz_path, allow_pickle=True)
        self.text_emb = np.ascontiguousarray(z["text_emb"], dtype=np.float32)   # (n,384)
        self.cat_indptr = z["cat_indptr"].astype(np.int64)                      # (n+1,)
        self.cat_indices = z["cat_indices"].astype(np.int64)                    # (nnz,)
        self.n_items, self.n_cats = (int(x) for x in z["cat_shape"])
        self.pop_feat = np.stack(
            [z["avg_rating"].astype(np.float32), z["rating_number_z"].astype(np.float32)],
            axis=1,
        )                                                                       # (n,2)
        self.text_dim = self.text_emb.shape[1]
        self.device = device
        self._text_t = torch.from_numpy(self.text_emb)                          # kept on CPU
        self._pop_t = torch.from_numpy(self.pop_feat)

    def text(self, item_idx_np) -> torch.Tensor:
        return self._text_t[item_idx_np].to(self.device, non_blocking=True)

    def pop(self, item_idx_np) -> torch.Tensor:
        return self._pop_t[item_idx_np].to(self.device, non_blocking=True)

    def cat_segments(self, item_idx_np):
        """Return (flat_cat_indices, segment_ids, batch_size) for a batch of items.

        ``index_add_(0, segment_ids, cat_table(flat))`` then sums each item's category
        vectors. Items with no categories contribute nothing (their row stays zero).
        """
        idx = np.asarray(item_idx_np)
        starts = self.cat_indptr[idx]
        lengths = self.cat_indptr[idx + 1] - starts
        total = int(lengths.sum())
        B = len(idx)
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, B
        seg = np.repeat(np.arange(B), lengths)
        seg_starts_out = np.repeat(np.cumsum(lengths) - lengths, lengths)
        src = np.repeat(starts, lengths) + (np.arange(total) - seg_starts_out)
        flat = self.cat_indices[src]
        return (
            torch.from_numpy(flat).to(self.device),
            torch.from_numpy(seg).to(self.device),
            B,
        )


class VerticalData:
    """Stage A artifacts for one vertical."""

    def __init__(self, root: str | Path, vertical: str, device: torch.device,
                 store: ItemFeatureStore | None = None):
        self.vertical = vertical
        self.vdir = Path(root) / vertical
        self.inter = pd.read_parquet(
            self.vdir / "interactions.parquet",
            columns=["user_idx", "item_idx", "label", "split", "timestamp"],
        )
        self.store = store or ItemFeatureStore(self.vdir / "item_features.npz", device)
        self.n_items = self.store.n_items
        self.n_cats = self.store.n_cats
        self.user_map = pd.read_parquet(self.vdir / "id_maps" / "user.parquet")
        self.n_users = int(self.user_map["user_idx"].max()) + 1

    def train_positives(self, restrict_users=None) -> np.ndarray:
        m = (self.inter["label"] == 1) & (self.inter["split"] == "train")
        sub = self.inter.loc[m, ["user_idx", "item_idx"]]
        if restrict_users is not None:
            sub = sub[sub["user_idx"].isin(np.asarray(restrict_users))]
        return sub.to_numpy()

    def user_seen_sets(self, restrict_users=None) -> dict:
        seen = pd.read_parquet(self.vdir / "seen.parquet")
        if restrict_users is not None:
            seen = seen[seen["user_idx"].isin(np.asarray(restrict_users))]
        return seen.groupby("user_idx")["item_idx"].agg(set).to_dict()

    def item_popularity(self) -> np.ndarray:
        m = (self.inter["label"] == 1) & (self.inter["split"] == "train")
        counts = self.inter.loc[m, "item_idx"].value_counts()
        pop = np.zeros(self.n_items, dtype=np.float64)
        pop[counts.index.to_numpy()] = counts.to_numpy()
        return pop

    def test_positives(self, restrict_users=None) -> pd.DataFrame:
        m = (self.inter["label"] == 1) & (self.inter["split"] == "test")
        sub = self.inter.loc[m, ["user_idx", "item_idx", "timestamp"]]
        if restrict_users is not None:
            sub = sub[sub["user_idx"].isin(np.asarray(restrict_users))]
        return sub


class PairData:
    """Stage B bridge for one directed pair."""

    def __init__(self, root: str | Path, src: str, tgt: str):
        self.src, self.tgt = src, tgt
        self.pdir = Path(root) / "pairs" / f"{src}__{tgt}"
        self.roles = pd.read_parquet(self.pdir / "roles.parquet")
        self.pair_user_map = pd.read_parquet(self.pdir / "id_maps" / "user.parquet")
        with open(self.pdir / "meta.json") as fh:
            self.meta = json.load(fh)

    def overlap_user_ids(self) -> np.ndarray:
        ov = self.roles.loc[self.roles["role"] == "overlap", "user_idx"].to_numpy()
        s = self.pair_user_map.set_index("user_idx")["user_id"]
        return s.loc[ov].to_numpy()


def align_overlap(pair: PairData, src_vd: VerticalData, tgt_vd: VerticalData) -> pd.DataFrame:
    """Map each overlap user_id to its per-vertical (src_idx, tgt_idx)."""
    uids = pair.overlap_user_ids()
    src_map = src_vd.user_map.set_index("user_id")["user_idx"]
    tgt_map = tgt_vd.user_map.set_index("user_id")["user_idx"]
    df = pd.DataFrame({"user_id": uids})
    df["src_idx"] = df["user_id"].map(src_map)
    df["tgt_idx"] = df["user_id"].map(tgt_map)
    df = df.dropna().astype({"src_idx": "int64", "tgt_idx": "int64"}).reset_index(drop=True)
    return df
