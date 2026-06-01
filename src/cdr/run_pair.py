"""Orchestrate one (source, target, ablation): recommenders -> mapping -> evaluation.

Usage:
  PYTHONPATH=src python -m cdr.run_pair --config configs/smoke.yaml
  PYTHONPATH=src python -m cdr.run_pair --config configs/full.yaml --src Books --tgt Movies_and_TV --ablation full
"""
from __future__ import annotations

import argparse

from .config import Config
from .data import VerticalData, PairData, align_overlap
from .evaluate import evaluate_pair
from .train_mapping import train_mapping
from .train_recommender import train_recommender
from .utils import log, save_json, select_device, set_seed


def run_pair(cfg: Config, src: str, tgt: str, ablation: str,
             src_model=None, tgt_model=None, src_vd=None, tgt_vd=None):
    device = select_device(cfg.device)
    set_seed(cfg.seed)

    if src_vd is None:
        src_vd = VerticalData(cfg.data_root, src, device)
    if tgt_vd is None:
        tgt_vd = VerticalData(cfg.data_root, tgt, device)

    pair = PairData(cfg.data_root, src, tgt)
    align = align_overlap(pair, src_vd, tgt_vd)
    if cfg.subsample_users:
        align = align.head(cfg.subsample_users).reset_index(drop=True)
    log(f"pair {src}->{tgt} aligned overlap users={len(align)} "
        f"(meta overlap={pair.meta.get('overlap')})")

    if src_model is None:
        src_model = train_recommender(src_vd, cfg, ablation, include_users=align["src_idx"].unique())
    if tgt_model is None:
        tgt_model = train_recommender(tgt_vd, cfg, ablation, include_users=align["tgt_idx"].unique())

    f, eval_df = train_mapping(src_model, tgt_model, align, cfg, src, tgt, ablation)
    res = evaluate_pair(f, src_model, tgt_model, src_vd, tgt_vd, eval_df, cfg, ablation)

    out = {"src": src, "tgt": tgt, "ablation": ablation,
           "overlap_aligned": int(len(align)), "config": cfg.to_dict(), **res}
    save_json(out, cfg.result_path(src, tgt, ablation))
    log(f"wrote {cfg.result_path(src, tgt, ablation)}")
    return out, src_model, tgt_model


def print_summary(out: dict) -> None:
    metrics = out["metrics"]
    names = list(next(iter(metrics.values())).keys())
    print(f"\n=== {out['src']} -> {out['tgt']}  [{out['ablation']}]  "
          f"cohort={out['n_eval_users']} ===")
    hdr = "model".ljust(28) + "".join(n.ljust(16) for n in names)
    print(hdr)
    print("-" * len(hdr))
    for m, md in metrics.items():
        row = m.ljust(28) + "".join(f"{md[n]['mean']:.4f}±{md[n]['std']:.3f}".ljust(16) for n in names)
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--tgt", default=None)
    ap.add_argument("--ablation", default=None)
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config)
    src = a.src or cfg.pairs[0][0]
    tgt = a.tgt or cfg.pairs[0][1]
    ablation = a.ablation or cfg.ablation
    out, *_ = run_pair(cfg, src, tgt, ablation)
    print_summary(out)


if __name__ == "__main__":
    main()
