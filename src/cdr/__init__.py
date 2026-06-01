"""Cross-Domain Recommendation (EMCDR) modeling package.

Reuse-first architecture mirroring the Stage A / Stage B data layout:
  - 5 per-vertical hybrid recommenders, trained once each and reused across pairs
    (``train_recommender``).
  - 8 per-pair mapping MLPs that reuse the cached per-vertical embeddings
    (``train_mapping``).
  - Leave-one-out sampled evaluation of EMCDR vs. baselines (``evaluate``).

See the approved plan and ``data/processed/DATASET_DESCRIPTION.md`` for schemas.
"""
