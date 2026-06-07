"""B2 — build a 3-core Video_Games vertical (Video_Games_3core) + its 3 source bridges.

Faithful to notebooks/data_prep.ipynb Stage A/B, but:
  * k=3 for Video Games (vs 5-core elsewhere) — the R1 contingency / B2 lever.
  * written as a NEW vertical `Video_Games_3core` so the cached 5-core recommenders and
    all prior results are untouched (data/processed/Video_Games is never modified).
  * sampled negatives are NOT materialized (the modeling layer samples its own; unused) and
    the shared embed cache is read-only here (new items encoded in-memory) to dodge the
    iCloud large-write TimeoutError noted in progress.md.
  * Stage B writes only what the modeling layer reads: roles.parquet, id_maps/user.parquet, meta.json.

  PYTHONPATH=src python data/build_vg_3core.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
from scipy.sparse import csr_matrix

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PAIRS = PROC / "pairs"
CACHE = ROOT / "data" / "embed_cache" / "sentence-transformers_all-MiniLM-L6-v2.npz"
SIDE_CACHE = ROOT / "data" / "embed_cache" / "vg3core_new.npz"  # newly-encoded vectors (resumable)

SRC_VERT = "Video_Games"
OUT_VERT = "Video_Games_3core"
K = 3
SEED = 42
POS_TH = 4
DROP = (0, 3)
EXPL_NEG = (1, 2)
RATIOS = (0.8, 0.1, 0.1)
EDIM = 384
MAXSEQ = 256
EMODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCES = ["Movies_and_TV", "Toys_and_Games", "Books"]
FLOOR = 10_000


# ---------- Stage A helpers (copied from data_prep.ipynb) ----------
def load_ratings(vertical: str) -> pd.DataFrame:
    p = RAW / "reviews" / f"{vertical}.csv"
    tbl = pacsv.read_csv(p, convert_options=pacsv.ConvertOptions(column_types={
        "user_id": "string", "parent_asin": "string", "rating": "float32", "timestamp": "int64"}))
    df = tbl.to_pandas()
    df["rating"] = df["rating"].astype("int8")
    return df


def kcore_filter(df: pd.DataFrame, k: int, max_iters: int = 30) -> pd.DataFrame:
    prev = -1
    for _ in range(max_iters):
        u = df.groupby("user_id").size()
        df = df[df["user_id"].isin(u[u >= k].index)]
        i = df.groupby("parent_asin").size()
        df = df[df["parent_asin"].isin(i[i >= k].index)]
        if len(df) == prev:
            return df.reset_index(drop=True)
        prev = len(df)
    return df.reset_index(drop=True)


def _normalize(s):
    if not isinstance(s, str):
        return None
    s = " ".join(s.split())
    return s if s else None


def _join_list(xs) -> str:
    if not isinstance(xs, list):
        return ""
    return "\n".join(x.strip() for x in xs if isinstance(x, str) and x.strip())


def _details_to_text(d) -> str:
    if not isinstance(d, dict):
        return ""
    parts = []
    for k, v in d.items():
        if isinstance(v, list):
            v_str = ", ".join(x for x in v if isinstance(x, str))
        elif isinstance(v, str):
            v_str = v
        else:
            v_str = str(v)
        if v_str:
            parts.append(f"{k}: {v_str}")
    return ". ".join(parts)


def build_text(rec) -> tuple[str, str]:
    chunks = []
    if rec.get("title"):
        chunks.append(_normalize(rec["title"]))
    d = _join_list(rec.get("description"))
    if d:
        chunks.append(d)
    f = _join_list(rec.get("features"))
    if f:
        chunks.append(f)
    primary = "\n".join(c for c in chunks if c)
    if primary:
        return primary, "primary"
    chunks = []
    if rec.get("title"):
        chunks.append(_normalize(rec["title"]))
    if _normalize(rec.get("store")):
        chunks.append(_normalize(rec["store"]))
    dd = _details_to_text(rec.get("details"))
    if dd:
        chunks.append(dd)
    fb = ". ".join(c for c in chunks if c)
    if fb:
        return fb, "fallback"
    return "", "none"


def stream_meta_for_kept(vertical: str, kept: set) -> pd.DataFrame:
    p = RAW / "meta" / f"meta_{vertical}.jsonl"
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            a = r.get("parent_asin")
            if a not in kept:
                continue
            text, src = build_text(r)
            cats = r.get("categories") or []
            try:
                avg = float(r["average_rating"]) if r.get("average_rating") is not None else np.nan
            except (TypeError, ValueError):
                avg = np.nan
            try:
                rn = int(r["rating_number"]) if r.get("rating_number") is not None else 0
            except (TypeError, ValueError):
                rn = 0
            rows.append({
                "parent_asin": a, "item_text": text, "has_text": src != "none",
                "text_source": src,
                "categories": [_normalize(c) for c in cats if isinstance(c, str)],
                "average_rating": avg, "rating_number": rn,
            })
    return pd.DataFrame(rows)


def items_to_multihot(items: pd.DataFrame, vocab: dict) -> csr_matrix:
    rows, cols = [], []
    for i, cats in enumerate(items["categories"]):
        for c in cats:
            j = vocab.get(c)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return csr_matrix((np.ones(len(rows), dtype="float32"), (rows, cols)),
                      shape=(len(items), len(vocab)))


def temporal_split(ts: np.ndarray, ratios) -> np.ndarray:
    n = len(ts)
    train_end = int(ratios[0] * n)
    val_end = int((ratios[0] + ratios[1]) * n)
    order = np.argsort(ts, kind="mergesort")
    out = np.array(["train"] * n, dtype=object)
    out[order[train_end:val_end]] = "val"
    out[order[val_end:]] = "test"
    return out


def _pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def encode_inmem(items: pd.DataFrame) -> np.ndarray:
    """Embeddings for items.item_text. Hits come (lazily) from the shared cache and a
    resumable side cache; misses are encoded on GPU/MPS and persisted to the side cache
    so a re-run is instant. has_text==False -> zeros."""
    z = np.load(CACHE)
    keyset = set(z.files)                                  # ~843k names; cheap to build
    side = {}
    if SIDE_CACHE.exists():
        zs = np.load(SIDE_CACHE)
        side = {k: zs[k] for k in zs.files}
    print(f"  shared cache keys={len(keyset):,}  side cache={len(side):,}")

    texts = items["item_text"].tolist()
    has = items["has_text"].values
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() if t else "" for t in texts]
    out = np.zeros((len(items), EDIM), dtype="float32")
    need_idx, need_text = [], []
    hits = 0
    for i, (t, h) in enumerate(zip(texts, hashes)):
        if not t or not has[i]:
            continue
        if h in keyset:                                   # lazy extract only what we need
            out[i] = z[h]; hits += 1
        elif h in side:
            out[i] = side[h]; hits += 1
        else:
            need_idx.append(i); need_text.append(t)
    print(f"  cache_hits={hits:,}  to_encode={len(need_text):,}", flush=True)

    if need_text:
        import os
        from sentence_transformers import SentenceTransformer
        dev = _pick_device()
        print(f"  loading {EMODEL} on {dev} ...", flush=True)
        model = SentenceTransformer(EMODEL, device=dev)
        model.max_seq_length = MAXSEQ
        emb = model.encode(need_text, batch_size=256, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
        for k, i in enumerate(need_idx):
            out[i] = emb[k]
            side[hashes[i]] = emb[k]
        tmp = f"/tmp/vg3core_new_{os.getpid()}.npz"       # atomic write (iCloud-safe)
        np.savez(tmp, **side)
        os.replace(tmp, SIDE_CACHE)
        print(f"  persisted {len(side):,} vectors -> {SIDE_CACHE.name}", flush=True)

    ph = int((out == 0).all(axis=1).sum())
    print(f"  embedded={len(items) - ph:,}  placeholder={ph:,}")
    return out


def write_vertical(items, signal, seen_df, emb, cat):
    out = PROC / OUT_VERT
    (out / "id_maps").mkdir(parents=True, exist_ok=True)
    asins = items["parent_asin"].tolist()
    item2idx = {a: i for i, a in enumerate(asins)}
    users = sorted(set(signal["user_id"]))
    user2idx = {u: i for i, u in enumerate(users)}

    sig = signal.copy()
    sig["user_idx"] = sig["user_id"].map(user2idx).astype("int64")
    sig["item_idx"] = sig["parent_asin"].map(item2idx)
    sig = sig.dropna(subset=["item_idx"])
    sig["item_idx"] = sig["item_idx"].astype("int64")
    sig["split"] = temporal_split(sig["timestamp"].values, RATIOS)
    sig[["user_idx", "item_idx", "label", "weight", "rating", "timestamp", "split"]] \
        .to_parquet(out / "interactions.parquet", index=False)

    np.savez(out / "item_features.npz",
             text_emb=emb,
             has_text=items["has_text"].values.astype("bool"),
             text_source=np.array(items["text_source"].tolist(), dtype=object),
             cat_indptr=cat.indptr, cat_indices=cat.indices, cat_shape=np.array(cat.shape),
             avg_rating=items["avg_rating_filled"].values.astype("float32"),
             rating_number_z=items["rating_number_z"].values.astype("float32"),
             item_idx_to_asin=np.array(asins, dtype=object))

    # seen.parquet — all post-drop interactions of kept users/items (vectorized)
    sd = seen_df[seen_df["user_id"].isin(user2idx) & seen_df["parent_asin"].isin(item2idx)].copy()
    sd["user_idx"] = sd["user_id"].map(user2idx).astype("int64")
    sd["item_idx"] = sd["parent_asin"].map(item2idx).astype("int64")
    sd[["user_idx", "item_idx"]].drop_duplicates().to_parquet(out / "seen.parquet", index=False)

    pd.DataFrame({"user_id": list(user2idx), "user_idx": list(user2idx.values())}) \
        .to_parquet(out / "id_maps" / "user.parquet", index=False)
    pd.DataFrame({"parent_asin": asins, "item_idx": range(len(asins))}) \
        .to_parquet(out / "id_maps" / "item.parquet", index=False)
    sc = {k: int(v) for k, v in pd.Series(sig["split"]).value_counts().items()}
    print(f"  wrote {OUT_VERT}: users={len(user2idx):,} items={len(asins):,} "
          f"pos={int((sig['label']==1).sum()):,} splits={sc}")
    return user2idx


def write_pair_min(src: str):
    key = f"{src}__{OUT_VERT}"
    out = PAIRS / key
    (out / "id_maps").mkdir(parents=True, exist_ok=True)
    s_users = set(pd.read_parquet(PROC / src / "id_maps" / "user.parquet")["user_id"])
    t_users = set(pd.read_parquet(PROC / OUT_VERT / "id_maps" / "user.parquet")["user_id"])
    shared = s_users & t_users
    union = s_users | t_users
    jacc = len(shared) / max(1, len(union))
    users = sorted(union)
    u2i = {u: i for i, u in enumerate(users)}
    src_only = sorted(s_users - t_users)
    tgt_only = sorted(t_users - s_users)
    roles = pd.DataFrame({
        "user_idx": [u2i[u] for u in src_only] + [u2i[u] for u in tgt_only] + [u2i[u] for u in sorted(shared)],
        "role": ["source_only"] * len(src_only) + ["target_only"] * len(tgt_only) + ["overlap"] * len(shared),
    })
    roles.to_parquet(out / "roles.parquet", index=False)
    pd.DataFrame({"user_id": list(u2i), "user_idx": list(u2i.values())}) \
        .to_parquet(out / "id_maps" / "user.parquet", index=False)
    meta = {"source": src, "target": OUT_VERT, "overlap": len(shared), "jaccard": round(jacc, 6),
            "overlap_floor": FLOOR, "borderline": len(shared) < FLOOR, "k_core_target": K}
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  {src:>15s} -> {OUT_VERT}: overlap={len(shared):,} jaccard={jacc:.4f}"
          f"{'  [BORDERLINE]' if len(shared) < FLOOR else ''}")


def main():
    t = time.time()
    print(f"=== Stage A: {OUT_VERT} (k={K}) ===")
    df = load_ratings(SRC_VERT)
    df = df[~df["rating"].isin(DROP)].reset_index(drop=True)
    post_drop = df[["user_id", "parent_asin"]]                       # for seen
    df3 = kcore_filter(df, K)
    # dedup (user,item): latest ts wins
    df3 = df3.sort_values("timestamp").groupby(["user_id", "parent_asin"], as_index=False).tail(1).reset_index(drop=True)
    print(f"  0-core(post-drop) {len(df):,} -> {K}-core {len(df3):,}  "
          f"users={df3['user_id'].nunique():,} items={df3['parent_asin'].nunique():,} "
          f"retention={len(df3)/len(df)*100:.2f}%")

    kept = set(df3["parent_asin"].unique())
    items = stream_meta_for_kept(SRC_VERT, kept)
    items = items[items["parent_asin"].isin(kept)].drop_duplicates("parent_asin").reset_index(drop=True)
    # keep only interactions whose item has metadata (orphans -> drop, like assert_quality cov==1)
    df3 = df3[df3["parent_asin"].isin(set(items["parent_asin"]))].reset_index(drop=True)
    print(f"  items with meta={len(items):,}")

    # labels (pos + explicit neg; sampled negs skipped — unused by modeling)
    df3 = df3.copy()
    df3["label"] = -1
    df3.loc[df3["rating"] >= POS_TH, "label"] = 1
    df3.loc[df3["rating"].isin(EXPL_NEG), "label"] = 0
    df3 = df3[df3["label"] != -1].reset_index(drop=True)
    df3["weight"] = 1.0

    emb = encode_inmem(items)
    vocab = {}
    for cats in items["categories"]:
        for c in cats:
            if c and c not in vocab:
                vocab[c] = len(vocab)
    cat = items_to_multihot(items, vocab)
    print(f"  category vocab (VG-only)={len(vocab):,}  multihot nnz={cat.nnz:,}")

    rn = items["rating_number"].fillna(0).astype("float32")
    rnlog = np.log1p(rn)
    mu, sd = float(rnlog.mean()), float(rnlog.std() or 1.0)
    items["rating_number_z"] = ((rnlog - mu) / sd).astype("float32")
    items["avg_rating_filled"] = items["average_rating"].fillna(items["average_rating"].median()).astype("float32")

    write_vertical(items, df3, post_drop, emb, cat)

    print(f"=== Stage B: bridges -> {OUT_VERT} ===")
    for src in SOURCES:
        write_pair_min(src)
    print(f"done in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
