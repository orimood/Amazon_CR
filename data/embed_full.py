"""Full-catalog embedding job.

Targeted re-encode: reads the existing source/target_item_features.npz,
streams the matching meta JSONL to recover item_text via the notebook's
fallback chain, encodes with sha1-cache, and writes the real embeddings back
into the .npz. No other pipeline stage is re-run.

Idempotent: items already in the cache (e.g. the 5000-sample run) are skipped.
"""
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path("/Users/orimood/Desktop/homework/Amazon_CR")
PAIR_DIR = ROOT / "data" / "processed" / "Books__Movies_and_TV"
EMBED_CACHE = ROOT / "data" / "embed_cache"
META_DIR = ROOT / "data" / "raw" / "meta"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH = 256
DEVICE = "mps"

VERTS = [("Books", "source"), ("Movies_and_TV", "target")]

# --- text builder — IDENTICAL to the notebook ---------------------------------
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
    if primary: return primary, True

    # fallback: title + store + details (pseudo-title from Directors/etc.)
    chunks = []
    if rec.get("title"): chunks.append(_normalize(rec["title"]))
    if _normalize(rec.get("store")): chunks.append(_normalize(rec["store"]))
    d = _details_to_text(rec.get("details"))
    if d: chunks.append(d)
    fb = ". ".join(c for c in chunks if c)
    return (fb, True) if fb else ("", False)


def text_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stream_text_for(vertical: str, kept: set[str]) -> dict[str, str]:
    """One pass over meta_<Vertical>.jsonl → {asin: item_text} for kept asins."""
    out: dict[str, str] = {}
    path = META_DIR / f"meta_{vertical}.jsonl"
    with open(path) as fh:
        for line in tqdm(fh, desc=f"meta {vertical}", unit=" rec"):
            r = json.loads(line)
            a = r.get("parent_asin")
            if a in kept:
                txt, _ = build_text(r)
                out[a] = txt
                if len(out) == len(kept):
                    break  # all needed asins found, can stop
    return out


def main():
    cache_path = EMBED_CACHE / f"{MODEL.replace('/', '_')}.npz"
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cache = {k: z[k] for k in z.files}
        print(f"loaded {len(cache):,} cached vectors from {cache_path.name}")

    model = None  # lazy-load: skip if everything is cached

    for vertical, side in VERTS:
        t0 = time.time()
        npz_path = PAIR_DIR / f"{side}_item_features.npz"
        print(f"\n=== {side}: {vertical} ===")
        data = dict(np.load(npz_path, allow_pickle=True))
        asins = data["item_idx_to_asin"]
        print(f"  {len(asins):,} items in {npz_path.name}")

        text_map = stream_text_for(vertical, set(asins.tolist()))
        texts = [text_map.get(a, "") for a in asins]
        hashes = [text_hash(t) if t else "" for t in texts]

        need_idx = [i for i, (t, h) in enumerate(zip(texts, hashes))
                    if t and h not in cache]
        print(f"  need to encode: {len(need_idx):,}  (cache has {len(cache):,})")

        if need_idx:
            if model is None:
                model = SentenceTransformer(MODEL, device=DEVICE)
            need_text = [texts[i] for i in need_idx]
            new_emb = model.encode(
                need_text, batch_size=BATCH, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            for j, i in enumerate(need_idx):
                cache[hashes[i]] = new_emb[j]
            np.savez(cache_path, **cache)
            print(f"  cache now: {len(cache):,}")

        emb = np.zeros((len(texts), EMBED_DIM), dtype="float32")
        for i, h in enumerate(hashes):
            if h and h in cache:
                emb[i] = cache[h]

        data["text_emb"] = emb
        np.savez(npz_path, **data)
        nonzero = int((emb != 0).any(axis=1).sum())
        print(f"  wrote {npz_path.name}: nonzero rows {nonzero:,}/{len(emb):,}  "
              f"({time.time()-t0:.0f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
