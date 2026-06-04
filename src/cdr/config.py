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
    device: str = "auto"              # auto | cpu | mps | cuda
    seed: int = 42

    # model dims
    d: int = 64
    use_pop: bool = True

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

    # history-grounded bridge (the redesign — see bridge.py)
    pool: str = "mean"                           # mean | attention (source-history pooling)
    aug_k: int = 4                               # sub-history augmentation samples per user per epoch (0 = off)
    aug_min_frac: float = 0.4                    # sub-history keeps a random frac in [aug_min_frac, 1.0]
    bridge_hidden: list = field(default_factory=lambda: [128, 128])
    bridge_bottleneck: Optional[int] = None      # low-dim bottleneck before the MLP (decouples mapping dim from d)
    bridge_lr: float = 1e-3
    bridge_weight_decay: float = 1e-5
    bridge_dropout: float = 0.2
    bridge_epochs: int = 40
    bridge_batch: int = 1024
    bridge_neg: int = 1                          # target negatives per positive (BPR)
    early_stop_patience: int = 5                 # epochs without val improvement (0 = fixed epochs)
    semisup_weight: float = 0.0                  # CORAL alignment weight on non-overlap users (0 = off)
    semisup_batch: int = 512                     # users per alignment step (source-only / target-only)
    semisup_pool_cap: int = 20000                # cap on non-overlap users whose histories are cached

    # evaluation (leave-one-out, sampled)
    eval_ks: list = field(default_factory=lambda: [10, 20])
    eval_n_neg: int = 100
    eval_seeds: list = field(default_factory=lambda: [0, 1, 2])
    max_eval_users: Optional[int] = None
    holdout: str = "temporal"                    # held-out target positive: temporal (latest = predict
                                                 # the future, realistic) | random (seeded)
    neg_sampling: str = "uniform"                # negative sampling for eval candidates:
                                                 # uniform | popularity (draw negs ~ popularity to
                                                 # neutralize the MostPop baseline's free edge)
    compare_original: bool = False               # also train+eval the ORIGINAL free-embedding EMCDR
                                                 # mapping (f: U_src->U_tgt) for a head-to-head
    neg_train: str = "uniform"                   # TRAINING negatives (recommender + bridge):
                                                 # uniform | popularity (hard negatives — forces the
                                                 # model to learn taste, not popularity)
    neg_train_power: float = 1.0                 # popularity exponent for hard training negatives
                                                 # (1.0 = match eval distribution; 0.75 = softer)

    # what to run
    ablation: str = "full"                       # full | cats_only | text_only | id_only
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

    # convenience: per-vertical recommender checkpoint dir
    def rec_dir(self, vertical: str, ablation: str) -> Path:
        return Path(self.models_root) / "recommenders" / f"{vertical}__{ablation}__{self.tag}"

    # convenience: per-pair mapping checkpoint dir
    def map_dir(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.models_root) / "mappings" / f"{src}__{tgt}__{ablation}__{self.tag}"

    def result_path(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.results_root) / f"{src}__{tgt}__{ablation}__{self.tag}.json"

    # convenience: per-pair bridge checkpoint dir + result path (the redesign)
    def bridge_dir(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.models_root) / "bridges" / f"{src}__{tgt}__{ablation}__{self.tag}"

    def bridge_result_path(self, src: str, tgt: str, ablation: str) -> Path:
        return Path(self.results_root) / f"bridge__{src}__{tgt}__{ablation}__{self.tag}.json"
