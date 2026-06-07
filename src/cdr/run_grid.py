"""Run the full deliverable grid: all configured pairs × {full, cats_only, text_only}.

Per-vertical recommenders are trained once per (vertical, ablation) and reused across
every pair that touches them — the Stage A/B reuse pattern in model form. Intended for
``configs/full.yaml`` (subsample_users=null). Heavy; deferred from the build+smoke pass.

  PYTHONPATH=src python -m cdr.run_grid --config configs/full.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .data import VerticalData
from .run_pair import print_summary, run_pair
from .utils import log, save_json, select_device, set_seed

ABLATIONS = ["full", "cats_only", "text_only"]


def run_grid(cfg: Config, ablations=None):
    ablations = ablations or cfg.ablations or ABLATIONS
    device = select_device(cfg.device)
    set_seed(cfg.seed)
    reuse = cfg.subsample_users is None        # only reuse recommenders across pairs in full mode

    vds: dict[str, VerticalData] = {}
    rec_cache: dict[tuple, object] = {}
    results = []

    for src, tgt in cfg.pairs:
        for ablation in ablations:
            for v in (src, tgt):
                if v not in vds:
                    vds[v] = VerticalData(cfg.data_root, v, device)
            sm = rec_cache.get((src, ablation)) if reuse else None
            tm = rec_cache.get((tgt, ablation)) if reuse else None
            out, sm, tm = run_pair(cfg, src, tgt, ablation,
                                   src_model=sm, tgt_model=tm,
                                   src_vd=vds[src], tgt_vd=vds[tgt])
            if reuse:
                rec_cache[(src, ablation)] = sm
                rec_cache[(tgt, ablation)] = tm
            print_summary(out)
            results.append({"src": src, "tgt": tgt, "ablation": ablation,
                            "n_eval_users": out["n_eval_users"], "metrics": out["metrics"]})

    grid_path = Path(cfg.results_root) / f"GRID__{cfg.tag}.json"
    save_json({"pairs": cfg.pairs, "ablations": ablations, "results": results}, grid_path)
    log(f"wrote {grid_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run_grid(Config.from_yaml(a.config))


if __name__ == "__main__":
    main()
