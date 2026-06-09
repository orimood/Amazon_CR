"""Real held-out cold-start user demo (faithful, ground-truth-backed).

Unlike demo_cold_user.py (a synthetic fold-in, illustrative only), this fabricates
nothing. It picks a REAL user from the Books -> Movies_and_TV evaluation cohort --
someone with a genuinely trained source embedding AND a known movie they later
watched -- runs the actual trained EMCDR pipeline, and shows:

  * the books they rated highly in the source domain,
  * EMCDR's top movie recommendations,
  * the movie they ACTUALLY watched, and where each model ranked it under the exact
    leave-one-out protocol from evaluate.py: the held movie + 100 sampled negatives,
    averaged over the same 3 seeds. This is the protocol behind the headline metric.

Run:
  PYTHONPATH=src python -m cdr.demo_real_user                  # default showcase user
  PYTHONPATH=src python -m cdr.demo_real_user --user-id <ID>
  PYTHONPATH=src python -m cdr.demo_real_user --src-idx 596747
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .data import VerticalData, PairData, align_overlap, ItemFeatureStore
from .models import HybridRecommender, MappingMLP

SRC, TGT, ABL = "Books", "Movies_and_TV", "full"
# A real cold-start reader of military/space sci-fi (Lost Fleet, Star Carrier, Kris
# Longknife, ...). EMCDR ranks the movie they actually watched #1 of 101, beating
# popularity and content. Override with --user-id / --src-idx to inspect others.
DEFAULT_USER_ID = "AEE75VYSOPYGNXYIHXZZD2XOTU3Q"


def load_recommender(cfg: Config, vertical: str):
    rdir = Path(cfg.models_root) / "recommenders" / f"{vertical}__{ABL}__{cfg.tag}"
    meta = json.load(open(rdir / "meta.json"))
    store = ItemFeatureStore(Path(cfg.data_root) / vertical / "item_features.npz", torch.device("cpu"))
    model = HybridRecommender(
        meta["n_users"], meta["n_items"], meta["n_cats"], meta["d"], meta["ablation"],
        meta["use_pop"], store.text_dim, torch.device("cpu"),
        text_init=meta.get("text_init", False), freeze_id=meta.get("freeze_id", False))
    model.load_state_dict(torch.load(rdir / "model.pt", map_location="cpu"))
    model.eval()
    return model, store


def held_out_movie(tgt_vd: VerticalData, tgt_users: np.ndarray) -> dict:
    """Latest test positive per user — the movie we hide and try to predict."""
    tp = tgt_vd.test_positives(restrict_users=tgt_users)
    last = tp.sort_values("timestamp").groupby("user_idx", as_index=False).tail(1)
    return dict(zip(last["user_idx"].to_numpy(), last["item_idx"].to_numpy()))


def loo_ranks(score_vecs: dict, held: int, seen: set, n_items: int,
              n_neg: int, seeds) -> dict:
    """Leave-one-out rank of `held` against n_neg sampled negatives, per evaluate.py.

    Same candidate set is shared by every model within a seed (held at index 0).
    """
    ranks = {m: [] for m in list(score_vecs) + ["random"]}
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        negs, need = [], n_neg
        while need > 0:
            for it in rng.integers(0, n_items, size=need * 2 + 8):
                if it not in seen:
                    negs.append(int(it))
                    if len(negs) == n_neg:
                        break
            need = n_neg - len(negs)
        cand = np.array([held] + negs[:n_neg])
        for m, sv in score_vecs.items():
            cs = sv[cand]
            ranks[m].append(int(1 + (cs > cs[0]).sum()))
        rs = rng.random(len(cand))                      # random baseline (held also random)
        ranks["random"].append(int(1 + (rs > rs[0]).sum()))
    return ranks


def lookup_titles(meta_path: Path, asins) -> dict:
    """One pass over a meta JSONL; regex-prefilter to just the ASINs we need."""
    need = set(asins)
    if not need:
        return {}
    pat = re.compile("|".join(re.escape(a) for a in need))
    out = {}
    with open(meta_path) as fh:
        for line in fh:
            if not pat.search(line):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--user-id", default=DEFAULT_USER_ID)
    ap.add_argument("--src-idx", type=int, default=None, help="select by Books user_idx instead")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    cfg = Config.from_yaml(a.config)
    dev = torch.device("cpu")

    print("Loading trained recommenders + EMCDR mapping ...")
    src_model, src_store = load_recommender(cfg, SRC)
    tgt_model, tgt_store = load_recommender(cfg, TGT)
    f = MappingMLP(cfg.d, cfg.map_hidden, cfg.map_dropout).to(dev)
    f.load_state_dict(torch.load(
        Path(cfg.models_root) / "mappings" / f"{SRC}__{TGT}__{ABL}__{cfg.tag}" / "model.pt",
        map_location="cpu"))
    f.eval()

    src_vd = VerticalData(cfg.data_root, SRC, dev, store=src_store)
    tgt_vd = VerticalData(cfg.data_root, TGT, dev, store=tgt_store)

    # Reproduce the exact cold-eval cohort: overlap users, seed-42 80/20 split
    # (train_mapping.py), restricted to those with a held-out target test positive.
    align = align_overlap(PairData(cfg.data_root, SRC, TGT), src_vd, tgt_vd)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(align))
    eval_df = align.iloc[perm[int(cfg.map_train_frac * len(align)):]].reset_index(drop=True)
    held = held_out_movie(tgt_vd, eval_df["tgt_idx"].to_numpy())
    cohort = eval_df[eval_df["tgt_idx"].isin(set(held))].reset_index(drop=True)

    if a.src_idx is not None:
        row = cohort[cohort["src_idx"] == a.src_idx]
        sel = f"src_idx={a.src_idx}"
    else:
        row = cohort[cohort["user_id"] == a.user_id]
        sel = f"user_id={a.user_id}"
    if len(row) == 0:
        raise SystemExit(f"{sel} is not in the {SRC}->{TGT} cold-start cohort "
                         f"(size {len(cohort)}). Pick another, or use --src-idx.")
    row = row.iloc[0]
    s_idx, t_idx, h_item = int(row["src_idx"]), int(row["tgt_idx"]), int(held[int(row["tgt_idx"])])

    # the user's source library (Books they rated >=4 -> positive, train split)
    pos = src_vd.train_positives(restrict_users=[s_idx])
    book_items = pos[:, 1]
    bmap = pd.read_parquet(Path(cfg.data_root) / SRC / "id_maps" / "item.parquet").set_index("item_idx")["parent_asin"]
    tmap = pd.read_parquet(Path(cfg.data_root) / TGT / "id_maps" / "item.parquet").set_index("item_idx")["parent_asin"]
    book_asins = [bmap.loc[i] for i in book_items]
    held_asin = tmap.loc[h_item]

    # --- run the real trained EMCDR + the same baselines evaluate.py compares ---
    V = tgt_model.all_item_embeddings(tgt_store)
    with torch.no_grad():
        mapped = f(src_model.user_emb(torch.tensor([s_idx]))).squeeze(0)
        emcdr = (mapped @ V.T).numpy()
        mean_user = tgt_model.user_emb.weight.mean(0)
        target_mf = (mean_user @ V.T).numpy()
    pop = tgt_vd.item_popularity()
    prof = F.normalize(torch.from_numpy(src_store.text_emb[book_items].mean(0)), dim=-1)
    content = (F.normalize(torch.from_numpy(tgt_store.text_emb), dim=-1) @ prof).numpy()

    score_vecs = {"emcdr": emcdr, "baseline_target_mf": target_mf,
                  "baseline_mostpop": pop, "baseline_feature_transfer": content}
    seen = tgt_vd.user_seen_sets(restrict_users=[t_idx]).get(t_idx, set())
    ranks = loo_ranks(score_vecs, h_item, seen, tgt_vd.n_items, cfg.eval_n_neg, cfg.eval_seeds)
    full_rank = int(1 + (emcdr > emcdr[h_item]).sum())

    # --- titles (one pass each over the relevant meta files) ---
    pool = max(6 * a.k, 60)
    rec_idx = torch.topk(torch.from_numpy(emcdr), pool).indices.numpy()
    rec_asins = [tmap.loc[int(i)] for i in rec_idx]
    print("Looking up titles from catalog metadata (movies, then books) ...")
    mtitles = lookup_titles(Path("data/raw/meta") / f"meta_{TGT}.jsonl", set(rec_asins) | {held_asin})
    btitles = lookup_titles(Path("data/raw/meta") / f"meta_{SRC}.jsonl", set(book_asins))

    # --- report ---
    print("\n" + "=" * 74)
    print(f"REAL cold-start user  ({sel})")
    print(f"  source library: {len(book_asins)} books rated highly in {SRC}")
    print("=" * 74)
    shown = [btitles.get(b, "") for b in book_asins if btitles.get(b, "")]
    for t in shown[:22]:
        print(f"   • {t[:66]}")
    if len(shown) > 22:
        print(f"   ... and {len(shown) - 22} more")

    print("\n" + "-" * 74)
    print(f"EMCDR top {a.k} movie recommendations  (trained Books->Movies mapping)")
    print("-" * 74)
    n, skip = 0, 0
    for i, asin in zip(rec_idx, rec_asins):
        t = mtitles.get(asin, "")
        if not t:
            skip += 1
            continue
        n += 1
        print(f"  {n:2d}. {t[:64]:<64} [{asin}]")
        if n == a.k:
            break
    if skip:
        print(f"(skipped {skip} higher-ranked items with no title metadata)")

    print("\n" + "-" * 74)
    print("THE MOVIE THEY ACTUALLY WATCHED (held out, hidden from every model):")
    print(f"  >> {mtitles.get(held_asin, '(untitled)')}  [{held_asin}]")
    print("-" * 74)
    print(f"  Leave-one-out rank of that movie vs {cfg.eval_n_neg} random movies "
          f"(seeds {cfg.eval_seeds}, lower=better):\n")
    print(f"    {'model':<28}{'per-seed':<16}{'mean':<8}")
    order = ["emcdr", "baseline_target_mf", "baseline_mostpop",
             "baseline_feature_transfer", "random"]
    for m in order:
        rs = ranks[m]
        print(f"    {m:<28}{str(rs):<16}{np.mean(rs):<8.1f}")
    print(f"\n  (EMCDR also ranks it #{full_rank:,} of {tgt_vd.n_items:,} across the whole catalog.)")

    # aggregate context so the single user is not mistaken for the whole result
    try:
        summ = json.load(open(Path(cfg.results_root) / "SUMMARY.json"))
        pj = next(p for p in summ["pairs"] if p["src"] == SRC and p["tgt"] == TGT)
        r10 = pj["metrics"]["emcdr"]["Recall@10"]["mean"]
        ft10 = pj["metrics"]["baseline_feature_transfer"]["Recall@10"]["mean"]
        print(f"\n  Context: over all {pj['n_eval_users']:,} cold-start users on this pair, "
              f"EMCDR Recall@10={r10:.3f} vs content {ft10:.3f} (deployed config).")
    except (FileNotFoundError, KeyError, StopIteration):
        pass
    print("\n  Note: this is one selected, illustrative real user; the line above is the\n"
          "  aggregate. Use --src-idx to inspect others (596747, 653973 are also strong).\n")


if __name__ == "__main__":
    main()
