"""Full-catalog embedding pass for the per-vertical Stage A layout.

Companion to notebooks/data_prep.ipynb. The notebook ships with
`embed_sample_size = 2000` (smoke mode) so re-runs are fast; this script
fills in the remaining real embeddings without re-running any other Stage
A step.

For each vertical:
  - read data/processed/{vertical}/item_features.npz (smoke-mode artifact)
  - one-pass stream data/raw/meta/meta_{vertical}.jsonl to recover text for
    items currently holding the placeholder embedding
  - encode via sentence-transformers/all-MiniLM-L6-v2 against the shared
    SHA1 cache at data/embed_cache/sentence-transformers_all-MiniLM-L6-v2.npz
  - write the updated text_emb back into the npz (every other npz field
    preserved verbatim) via atomic /tmp staging + os.replace — direct
    writes to the iCloud-watched Desktop path time out on multi-hundred-MB
    npzs (Errno 60 ETIMEDOUT)

Idempotent: re-running after the cache is full is a near-no-op.

Run from project root:
    .venv/bin/python data/embed_all.py
"""
from __future__ import annotations
import json, hashlib, os, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def atomic_savez(target: Path, **arrays) -> None:
    """np.savez to a /tmp staging file, then os.replace into place.

    The Desktop path is iCloud-synced and long direct writes time out
    (Errno 60). /tmp is on the same root volume, so os.replace is atomic
    and iCloud sees a single rename rather than a multi-part write."""
    tmp = Path("/tmp") / f"_stage_{target.parent.name}_{target.name}.{os.getpid()}.npz"
    if tmp.exists():
        tmp.unlink()
    np.savez(tmp, **arrays)
    if not tmp.exists() and tmp.with_suffix(".npz.npz").exists():
        tmp = tmp.with_suffix(".npz.npz")
    os.replace(tmp, target)

ROOT       = Path("/Users/orimood/Desktop/homework/Amazon_CR")
PROCESSED  = ROOT / "data" / "processed"
EMBED_CACHE = ROOT / "data" / "embed_cache"
META_DIR   = ROOT / "data" / "raw" / "meta"

MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH     = 256
DEVICE    = "mps"

VERTICALS = ["Books", "Movies_and_TV", "CDs_and_Vinyl", "Video_Games", "Toys_and_Games"]


# ---------- text builder — identical to notebook + data/embed_full.py ----------
def _normalize(s):
    if not isinstance(s, str): return None
    s = " ".join(s.split())
    return s if s else None

def _join_list(xs):
    if not isinstance(xs, list): return ""
    return "\n".join(x.strip() for x in xs if isinstance(x, str) and x.strip())

def _details_to_text(d):
    if not isinstance(d, dict): return ""
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

def build_text(rec):
    # primary: title + description + features
    chunks = []
    if rec.get("title"): chunks.append(_normalize(rec["title"]))
    d = _join_list(rec.get("description"))
    if d: chunks.append(d)
    f = _join_list(rec.get("features"))
    if f: chunks.append(f)
    primary = "\n".join(c for c in chunks if c)
    if primary: return primary

    # fallback: title + store + details
    chunks = []
    if rec.get("title"): chunks.append(_normalize(rec["title"]))
    if _normalize(rec.get("store")): chunks.append(_normalize(rec["store"]))
    d = _details_to_text(rec.get("details"))
    if d: chunks.append(d)
    return ". ".join(c for c in chunks if c)


def text_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stream_text_for(vertical: str, wanted: set[str]) -> dict[str, str]:
    """One pass over meta_<Vertical>.jsonl -> {asin: item_text} for wanted asins."""
    out: dict[str, str] = {}
    path = META_DIR / f"meta_{vertical}.jsonl"
    with open(path) as fh:
        for line in tqdm(fh, desc=f"meta {vertical}", unit=" rec"):
            r = json.loads(line)
            a = r.get("parent_asin")
            if a in wanted:
                out[a] = build_text(r)
                if len(out) == len(wanted):
                    break
    return out


def main():
    cache_path = EMBED_CACHE / f"{MODEL.replace('/', '_')}.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cache = {k: z[k] for k in z.files}
        print(f"loaded {len(cache):,} cached vectors from {cache_path.name}")

    model = None  # lazy-load: skip if everything is cached

    grand_total = time.time()
    for vertical in VERTICALS:
        t0 = time.time()
        npz_path = PROCESSED / vertical / "item_features.npz"
        print(f"\n=== {vertical} ===")
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found")
            continue
        data = dict(np.load(npz_path, allow_pickle=True))
        asins      = data["item_idx_to_asin"]
        has_text   = data["has_text"].astype(bool)
        emb_in     = data["text_emb"]
        # placeholder = currently all-zero row; encode those that have_text
        is_placeholder = (emb_in != 0).any(axis=1) == False
        wanted = {a for a, ph, ht in zip(asins, is_placeholder, has_text) if ph and ht}
        print(f"  {len(asins):,} items, has_text={int(has_text.sum()):,}, "
              f"currently placeholder={int(is_placeholder.sum()):,}, "
              f"to recover-text-for={len(wanted):,}")

        if not wanted:
            print(f"  nothing to do ({time.time()-t0:.1f}s)")
            continue

        text_map = stream_text_for(vertical, wanted)
        # build (asin -> text) for ALL has_text items so we can cache-hit too
        texts  = [text_map.get(a, "") if ph and ht else "" for a, ph, ht in zip(asins, is_placeholder, has_text)]
        hashes = [text_hash(t) if t else "" for t in texts]

        need_idx = [i for i, (t, h) in enumerate(zip(texts, hashes)) if t and h not in cache]
        print(f"  need to encode: {len(need_idx):,}  (cache has {len(cache):,})")

        if need_idx:
            if model is None:
                model = SentenceTransformer(MODEL, device=DEVICE)
                model.max_seq_length = 256
            need_text = [texts[i] for i in need_idx]
            # encode in one big call; ST handles batching internally
            new_emb = model.encode(
                need_text, batch_size=BATCH, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            for j, i in enumerate(need_idx):
                cache[hashes[i]] = new_emb[j]
            atomic_savez(cache_path, **cache)
            print(f"  cache now: {len(cache):,}")

        # write back: rebuild text_emb from cache (preserves cells already filled)
        emb_out = emb_in.copy()
        for i, h in enumerate(hashes):
            if h and h in cache:
                emb_out[i] = cache[h]
        data["text_emb"] = emb_out
        atomic_savez(npz_path, **data)
        nonzero = int((emb_out != 0).any(axis=1).sum())
        print(f"  wrote {npz_path.relative_to(ROOT)}: nonzero rows {nonzero:,}/{len(emb_out):,}  "
              f"({time.time()-t0:.0f}s)")

    print(f"\nDone in {time.time()-grand_total:.0f}s. cache has {len(cache):,} vectors.")


if __name__ == "__main__":
    main()
