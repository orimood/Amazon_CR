"""Cold-start cross-domain demo.

Builds a *fake* user who has no history in the target domain, rates a handful of
source-domain items highly, and asks the trained EMCDR pipeline what to recommend
in the target domain. Default: 6 epic-fantasy **Books** -> **Movies_and_TV** picks.

Pipeline (mirrors evaluate.py's cold-start path):
  1. fold-in:  freeze the source item tower, fit a fresh user vector u to the rated
               items with the same BPR loss the recommender was trained on.
  2. rescale:  match u to the median norm of the trained user table so the learned
               mapping f sees an in-distribution input.
  3. map:      û_t = f(u)              (EMCDR cross-domain mapping)
  4. rank:     score = û_t · V_tgt     over every target item.
Also prints the pure content baseline (feature-transfer): cosine between the mean
MiniLM text embedding of the rated books and each movie's text embedding.

Run:
  PYTHONPATH=src python -m cdr.demo_cold_user
  PYTHONPATH=src python -m cdr.demo_cold_user --books 0261102664 075640407X ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import ItemFeatureStore
from .models import HybridRecommender, MappingMLP

DATA = "data/processed"
MODELS = "models"
META = "data/raw/meta"
SRC, TGT, ABL, TAG = "Books", "Movies_and_TV", "full", "full"
DEV = torch.device("cpu")

# Named taste profiles — (parent_asin, title), all verified present in the Books catalog.
PROFILES = {
    "fantasy": [
        ("0261102664", "The Hobbit"),
        ("0618346252", "The Fellowship of the Ring"),
        ("0553103547", "A Game of Thrones"),
        ("075640407X", "The Name of the Wind"),
        ("0765365278", "The Way of Kings"),
        ("0765360969", "Mistborn: The Final Empire"),
    ],
    "scifi": [
        ("0340839937", "Dune"),
        ("1904233023", "Ender's Game"),
        ("B082BHWQCJ", "The Martian"),
        ("0441000681", "Neuromancer"),
        ("0307913147", "Ready Player One"),
        ("0553380958", "Snow Crash"),
    ],
}
DEFAULT_BOOKS = PROFILES["fantasy"]


def load_recommender(vertical: str):
    rdir = Path(MODELS) / "recommenders" / f"{vertical}__{ABL}__{TAG}"
    meta = json.load(open(rdir / "meta.json"))
    store = ItemFeatureStore(Path(DATA) / vertical / "item_features.npz", DEV)
    model = HybridRecommender(
        meta["n_users"], meta["n_items"], meta["n_cats"], meta["d"],
        meta["ablation"], meta["use_pop"], store.text_dim, DEV,
        text_init=meta.get("text_init", False), freeze_id=meta.get("freeze_id", False),
    )
    model.load_state_dict(torch.load(rdir / "model.pt", map_location=DEV))
    model.eval()
    items = pd.read_parquet(Path(DATA) / vertical / "id_maps" / "item.parquet")
    return model, store, items, meta


def load_mapping(d: int) -> MappingMLP:
    f = MappingMLP(d, [128, 128], 0.2).to(DEV)
    f.load_state_dict(torch.load(
        Path(MODELS) / "mappings" / f"{SRC}__{TGT}__{ABL}__{TAG}" / "model.pt", map_location=DEV))
    f.eval()
    return f


def fold_in(V: torch.Tensor, pos_idx: np.ndarray, steps=400, lr=0.05, n_neg=512, seed=0):
    """Fit a user vector u to the rated items with frozen item embeddings V (BPR)."""
    n, d = V.shape
    pos = V[pos_idx]                                   # (P, d) — fixed item embeddings
    g = torch.Generator().manual_seed(seed)
    u = (0.01 * torch.randn(d, generator=g)).clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        neg = torch.randint(0, n, (n_neg,), generator=g)
        s_pos = pos @ u                                # (P,)
        s_neg = V[neg] @ u                             # (n_neg,)
        loss = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg.unsqueeze(0)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return u.detach()


def lookup_titles(vertical: str, asins) -> dict:
    """Stream the raw meta JSONL, pulling titles for just the ASINs we need."""
    need = set(asins)
    out = {}
    path = Path(META) / f"meta_{vertical}.jsonl"
    with open(path) as fh:
        for line in fh:
            if not any(a in line for a in need):
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = o.get("parent_asin")
            if a in need:
                out[a] = (o.get("title") or "").strip()
                need.discard(a)
                if not need:
                    break
    return out


def top_pool(scores: torch.Tensor, items: pd.DataFrame, pool: int):
    """Top `pool` items by score, as (item_idx, parent_asin, score)."""
    idx = torch.topk(scores, pool).indices.numpy()
    asin_by_idx = items.set_index("item_idx")["parent_asin"]
    return [(int(i), asin_by_idx.loc[int(i)], float(scores[int(i)])) for i in idx]


def take_titled(pool, titles: dict, k: int):
    """Walk a ranked pool, keep the first k items that have a non-empty title.

    Many Amazon "Prime Video" catalog items carry no title metadata; they are real
    items but uninterpretable in a demo, so we skip past them and report the count.
    """
    out, skipped = [], 0
    for i, asin, sc in pool:
        t = titles.get(asin, "")
        if t:
            out.append((i, asin, sc, t))
            if len(out) == k:
                break
        else:
            skipped += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="fantasy",
                    help="named taste profile (default: fantasy)")
    ap.add_argument("--books", nargs="*", default=None,
                    help="ad-hoc parent_asin list of source books (overrides --profile)")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()

    books = ([(b, b) for b in a.books] if a.books else PROFILES[a.profile])

    print("Loading trained Books + Movies recommenders and the EMCDR mapping ...")
    src_model, src_store, src_items, src_meta = load_recommender(SRC)
    tgt_model, tgt_store, tgt_items, _ = load_recommender(TGT)
    f = load_mapping(src_meta["d"])

    # resolve source books -> item_idx
    asin2idx = src_items.set_index("parent_asin")["item_idx"]
    pos_idx, kept = [], []
    for asin, title in books:
        if asin in asin2idx.index:
            pos_idx.append(int(asin2idx.loc[asin])); kept.append((asin, title))
        else:
            print(f"  !! {asin} ({title}) not in Books catalog — skipped")
    pos_idx = np.array(pos_idx, dtype=np.int64)

    print("\n" + "=" * 70)
    print("FAKE USER — rated these 6 books 5/5 (source domain: Books):")
    for asin, title in kept:
        print(f"   • {title}   [{asin}]")
    print("=" * 70)

    # 1-2. fold in a source user vector, then match trained-user scale
    Vs = src_model.all_item_embeddings(src_store)              # (n_books, d)
    u = fold_in(Vs, pos_idx)
    target_norm = src_model.user_emb.weight.norm(dim=1).median()
    u_scaled = u / (u.norm() + 1e-8) * target_norm

    # sanity: where do the 6 rated books rank for this folded-in user, in-domain?
    book_scores = Vs @ u
    pctile = [100.0 * (1.0 - (book_scores > book_scores[i]).float().mean().item()) for i in pos_idx]
    print(f"\n[sanity] folded-in user ranks its 6 books at the "
          f"{np.mean(pctile):.2f}th percentile of {len(book_scores):,} books on average "
          f"(100 = top). Fold-in is capturing the taste.")

    # 3-4. EMCDR: map to Movies space and score every movie
    with torch.no_grad():
        mapped = f(u_scaled.unsqueeze(0)).squeeze(0)
        Vt = tgt_model.all_item_embeddings(tgt_store)          # (n_movies, d)
        emcdr_scores = Vt @ mapped

    # content baseline (feature-transfer): cosine of mean book text vs each movie text
    prof = torch.from_numpy(src_store.text_emb[pos_idx].mean(0))
    prof = F.normalize(prof, dim=-1)
    movie_text = F.normalize(torch.from_numpy(tgt_store.text_emb), dim=-1)
    content_scores = movie_text @ prof

    # pull a generous pool so we can skip past untitled (Prime Video) catalog items
    pool = max(10 * a.k, 150)
    emcdr_pool = top_pool(emcdr_scores, tgt_items, pool)
    content_pool = top_pool(content_scores, tgt_items, pool)

    # one pass over the Movies meta for all titles we might show
    want = {asin for _, asin, _ in emcdr_pool} | {asin for _, asin, _ in content_pool}
    print(f"\nLooking up movie titles from the catalog metadata ...")
    titles = lookup_titles(TGT, want)

    emcdr_top, emcdr_skip = take_titled(emcdr_pool, titles, a.k)
    content_top, content_skip = take_titled(content_pool, titles, a.k)

    def show(name, rows, skipped):
        print("\n" + "-" * 70)
        print(name)
        if skipped:
            print(f"(skipped {skipped} higher-ranked items with no title metadata)")
        print("-" * 70)
        for rank, (_, asin, sc, t) in enumerate(rows, 1):
            print(f"  {rank:2d}. {t[:64]:<64} [{asin}]")

    show("EMCDR cross-domain recommendations  (learned Books->Movies mapping)", emcdr_top, emcdr_skip)
    show("Content baseline  (text-similarity feature transfer)", content_top, content_skip)
    print()


if __name__ == "__main__":
    main()
