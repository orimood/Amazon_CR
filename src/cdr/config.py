"""Experiment configuration: a flat dataclass with defaults, loadable from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # identity / runtime
    name: str = "run"
    tag: str = "full"                 # checkpoint namespace: keeps smoke != full caches
    rec_tag: Optional[str] = None     # if set, recommenders load from THIS tag (reuse a
                                      # cached per-vertical layer) while mappings/results use `tag`
    rec_tag_overrides: dict = field(default_factory=dict)  # {vertical: tag} — per-vertical recommender
                                      # tag override (e.g. point only the Video_Games target at a B1
                                      # recommender while sources stay on the cached `full` layer)
    device: str = "auto"              # auto | cpu | mps | cuda
    seed: int = 42

    # model dims
    d: int = 64
    use_pop: bool = True
    # B1 (IMPROVEMENTS §B1): content-ground the per-item id factors for sparse targets.
    text_init: bool = False           # warm-start id_factor from a fixed projection of MiniLM text
    freeze_id: bool = False           # freeze id_factor (zeroed unless text_init) -> content-only item tower

    # per-vertical recommender (BPR)
    rec_epochs: int = 15
    rec_lr: float = 0.01
    rec_weight_decay: float = 1e-6
    rec_batch_size: int = 4096
    rec_steps_per_epoch: Optional[int] = None   # None = one pass over train positives
    subsample_users: Optional[int] = None       # None = all users (reusable embeddings)

    # EMCDR mapping (MLP)
    map_hidden: list = field(default_factory=lambda: [128, 128])
    map_epochs: int = 200
    map_lr: float = 1e-3
    map_weight_decay: float = 1e-5
    map_batch_size: int = 1024
    map_dropout: float = 0.2
    map_cos_weight: float = 0.5                  # loss = mse + cos_weight * (1 - cos)
    map_train_frac: float = 0.8                  # overlap split: mapping-train vs cold-eval

    # evaluation (leave-one-out, sampled)
    eval_ks: list = field(default_factory=lambda: [10, 20])
    eval_n_neg: int = 100
    eval_seeds: list = field(default_factory=lambda: [0, 1, 2])
    max_eval_users: Optional[int] = None

    # content-collaborative hybrid scorer (IMPROVEMENTS §C1): per-user z-scored blend of the
    # EMCDR score and the content-cosine (feature-transfer) score, swept over alpha.
    # alpha=1.0 -> pure EMCDR, alpha=0.0 -> pure content (both reproduce their base model exactly).
    run_hybrid: bool = False
    hybrid_alphas: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # what to run
    ablation: str = "full"                       # full | cats_only | text_only | id_only
    ablations: list = field(default_factory=lambda: ["full", "cats_only", "text_only"])  # grid sweep
    run_baselines: bool = True
    pairs: list = field(default_factory=list)    # list of [src, tgt]

    # paths (relative to project root / CWD)
    data_root: str = "data/processed"
    models_root: str = "models"
    results_root: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}        # noqa: E1133
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    # which recommender tag a given vertical resolves to: per-vertical override first, then
    # the run-wide rec_tag, then the results tag. Lets one run mix a B1 target recommender
    # with the cached `full` source recommenders.
    def rec_tag_for(self, vertical: str) -> str:
        return (self.rec_tag_overrides or {}).get(vertical) or self.rec_tag or self.tag

    # convenience: per-vertical recommender checkpoint dir.
    # Uses rec_tag when set so a results run can reuse a previously-cached recommender layer
    # (e.g. the heavy `full` recommenders) without retraining or clobbering the original results.
    def rec_dir(self, vertical: str, ablation: str) -> Path:
        return Path(self.models_root) / "recommenders" / f"{vertical}__{ablation}__{self.rec_tag_for(vertical)}"

    # convenience: per-pair mapping checkpoint dir
    def map_dir(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.models_root) / "mappings" / f"{src}__{tgt}__{ablation}__{self.tag}"

    def result_path(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.results_root) / f"{src}__{tgt}__{ablation}__{self.tag}.json"
