# Model Card — EMCDR Cross-Domain Recommender (CRISP-DM §4)

_Status: pipeline built + smoke-validated (2026-06-01). Full-scale training over all 8 pairs is deferred._
Code: `src/cdr/`. Configs: `configs/`. Results: `results/`. Plan: the approved modeling plan.

## What this is

A reuse-first EMCDR pipeline that recommends items in a vertical a user has no history in,
using their history in another vertical. The cross-vertical bridge is **item-side metadata**
(MiniLM text embeddings + hierarchical categories), consistent with the project thesis that
"metadata carries the signal." Deliverable scope is **all 8 viable vertical pairs**, not one.

## Architecture

**Per-vertical hybrid recommender** (trained once per vertical, reused across every pair):
- User tower: learned embedding table `U[user] ∈ R^d`.
- Item tower (hybrid): `item_emb = id_factor + W_text·text_emb(384) + cat_bag(categories) [+ pop_proj]`.
  - `cat_bag` is a learned category table aggregated with `index_add_` (not `nn.EmbeddingBag`,
    which is unimplemented on Apple MPS) — so the code runs identically on CPU and MPS.
- Score: `s(u,i) = U[u] · item_emb_i`. Trained with **BPR** (pairwise, sampled negatives
  excluding each user's seen set).
- **Ablations** toggle the content branches (`id` always present):
  `full = id+text+cat(+pop)`, `cats_only = id+cat`, `text_only = id+text`, `id_only = pure MF`.

**EMCDR mapping** (trained once per directed pair, reuses cached per-vertical embeddings):
- Overlap users aligned across verticals by `user_id` (the pair-global and per-vertical
  `user_idx` spaces differ; `align_overlap` bridges them via the id-maps).
- `f_θ`: MLP `R^d → R^d` trained on overlap-**train** users to match `f(U_src) ≈ U_tgt`
  (loss = MSE + `cos_weight`·(1−cos)). Overlap users are split into mapping-train / cold-eval
  so the evaluation cohort is unseen by the mapping.

**Cold-start inference:** `û_t = f(U_src[u])`; rank target items by `û_t · V_tgt`.

## Evaluation (HW1 §1.3 protocol)

Leave-one-out, sampled: per cold-eval user, 1 held-out target test positive (latest by
timestamp) + 100 sampled unseen negatives, ranked together. Metrics: Recall@10/20,
NDCG@10/20, HR@10 — mean ± std across negative-sampling seeds. The temporal split (Stage A)
is the time-based split. Baselines on the identical candidate sets:
`baseline_random` (floor), `baseline_mostpop` (deployed "popular this week" fallback),
`baseline_target_mf` (single-vertical MF applied to a cold user via the mean user vector),
`baseline_feature_transfer` (Model B — cosine of mean source liked-item text vs. target item
text, in the shared MiniLM space).

## Smoke validation (not headline numbers)

Tiny run on Toys_and_Games→Video_Games (4,000 overlap users, d=32, 2 BPR epochs, CPU;
cohort = 257 cold-eval users, 2 seeds). Purpose: prove the pipeline trains end-to-end and the
metrics plumbing is correct. Observations that sanity-check the design:
- Content-bearing EMCDR sits above the chance baselines; `id_only` (pure MF, no content)
  falls **below** chance — i.e. the metadata branches are what enable cold-start transfer.
- `feature_transfer` is strong and identical across ablations (independent of the recommender),
  confirming real transferable signal in the shared MiniLM space.
- Re-runs are deterministic under the seed.

These are 2-epoch / d=32 / 4k-user numbers and must not be read as model quality.

## How to run

```bash
# smoke (CPU, minutes) — what has been validated
PYTHONPATH=src python -m cdr.run_pair --config configs/smoke.yaml
PYTHONPATH=src python -m cdr.run_pair --config configs/smoke.yaml --ablation cats_only

# deferred: full grid, all 8 pairs × {full, cats_only, text_only}, MPS
PYTHONPATH=src python -m cdr.run_grid --config configs/full.yaml
```

## Deferred (follow-up sessions)
- Full-scale training on `configs/full.yaml` (5 per-vertical recommenders × 3 ablations + 8
  mappings; Books is heaviest at 7.5M positives), tuned d / epochs.
- The full 8-pair × ablation results grid with per-pair lift-over-baseline tables.
- Random-split robustness (Stage-A re-run with `split="random"`, ≥2 seeds) for HW1 §1.3 (2).
- Qualitative N=50 face-validity spot-check (HW1 §1.3 subjective criterion).
