# Modeling Experiments — Cross-Domain Recommendation (EMCDR & beyond)

_Branch: `emcdr-modeling`. Code: `src/cdr/`. Results: `results/`. This document logs every modeling
experiment, the setup, the numbers, **why** they came out the way they did, and what to do next._

---

## TL;DR

We built an EMCDR-style cross-domain recommender, then redesigned its weak component (the mapping),
then evaluated everything honestly. The honest evaluation produced the most important result of the
project:

> **Under a properly debiased evaluation (temporal hold-out + popularity-weighted negatives), both
> learned cross-domain mappings — the original EMCDR *and* our redesigned history-grounded bridge —
> collapse to ~random. Their apparent performance under the standard (uniform-negative) protocol was
> largely *popularity exploitation*, not genuine taste transfer. The one method that keeps real
> signal is the simplest: a content-cosine over MiniLM text embeddings (`feature_transfer`).**

This was only visible because we changed the evaluation. Under the common uniform-negative protocol,
the learned models looked strong (Recall@10 ≈ 0.33–0.50) and we would have shipped a popularity
recommender in disguise.

---

## 1. Problem & setup

**Task.** Recommend items in a target vertical to a user who is *cold* there (no target history),
using their history in a source vertical. Cross-domain, cold-start, implicit feedback, top-K ranking.
The bridge across verticals is **item metadata** (MiniLM text embedding of title+description, and a
hierarchical-category multi-hot), since item IDs don't overlap across verticals but **users do**.

**Data.** Amazon Reviews 2023 (McAuley Lab), 5 verticals: Books, Movies & TV, CDs & Vinyl,
Video Games, Toys & Games. Per-vertical 5-core; rating ≥4 → positive, ≤2 → explicit negative, =3
dropped; +4:1 sampled negatives. MiniLM-L6-v2 (384-dim) item embeddings, 3,092-term category vocab,
popularity priors (avg rating + log-standardized rating count). 8 vertical pairs clear the ~10k
shared-user floor. Full data prep verified correct (see `progress.md`).

**Per-vertical recommender (shared by all methods).** Hybrid two-tower:
`item_emb = id_factor + W_text·text + cat_bag(categories) + pop_proj(popularity)`, user embedding
table, scored by dot product, trained with **BPR** using **uniform** negatives. `d=64` (full runs).
Its job is to produce item embeddings (and, for the original mapping, user factors).

**Compute.** CPU throughout — Apple MPS produces NaN BPR loss for this model (`nn.EmbeddingBag`
gaps; worked around with `index_add_`) and is slower for this embedding-lookup workload anyway.

**Evaluation.** Leave-one-out, sampled: hold out one target positive per cold user, mix with 100
negatives, rank the 101. Report Recall@10/20, NDCG@10/20, HR@10, mean±std over seeds. **Three knobs
turned out to matter enormously** (see §6): held-out item (temporal=latest vs random), and negative
sampling (uniform vs popularity-weighted).

---

## 2. Experiment 1 — Original EMCDR baseline (8 pairs × 3 ablations)

**Method.** Classic EMCDR: train per-vertical recommenders, then an MLP mapping `f: U_src → U_tgt`
fit on overlap users (MSE + cosine). Cold-start: `û_t = f(U_src) · V_tgt`. Ablations toggle the item
tower's content branches (`full` / `cats_only` / `text_only`).

**Protocol.** Global temporal split test positives + **uniform** negatives (the `evaluate.py`
harness). Recall@10, `full` ablation:

| pair | cohort | EMCDR | target-MF | feat-transfer | MostPop | Random | EMCDR beats all |
|---|--:|--:|--:|--:|--:|--:|:--:|
| Books → Movies&TV | 5918 | **0.308** | 0.293 | 0.231 | 0.191 | 0.097 | ✅ |
| Books → Toys&Games | 2416 | **0.327** | 0.318 | 0.299 | 0.238 | 0.102 | ✅ |
| Movies → CDs&Vinyl | 2009 | **0.351** | 0.316 | 0.262 | 0.238 | 0.096 | ✅ |
| Books → CDs&Vinyl | 1524 | **0.369** | 0.321 | 0.207 | 0.230 | 0.102 | ✅ |
| Movies → Toys&Games | 1325 | 0.286 | **0.303** | 0.273 | 0.226 | 0.096 | ❌ |
| Movies → Video Games | 647 | 0.170 | 0.177 | **0.246** | 0.131 | 0.100 | ❌ |
| Toys → Video Games | 943 | 0.164 | 0.122 | **0.289** | 0.095 | 0.094 | ❌ |
| Books → Video Games | 435 | 0.163 | 0.164 | **0.211** | 0.142 | 0.093 | ❌ |

**Ablation (text vs categories).** Best variant per pair: **text_only ×5, cats_only ×2, full ×1.**
Text is the stronger signal; stacking categories on top rarely helps. (Answers the project's §1.1
question.)

**Reading at the time.** EMCDR beat every baseline on the 4 high-overlap pairs and lost on the thin
ones — and where it lost, the *no-learning* content cosine (`feature_transfer`) won. Diagnosis:
**the mapping is data-starved at thin overlap** (~11–16k shared users), not too weak.

---

## 2b. Design discussions — ideas considered (the reasoning trail)

Recording the *why* behind the redesign, including paths we deliberately did **not** take.

**Fuse the content baseline into EMCDR — considered, rejected.** Since the content cosine (Model B)
beat EMCDR at thin overlap, an obvious move was to fuse them. Three options were sketched:
(A) score blend `α·s_emcdr + (1−α)·s_content`; (B) input fusion — feed the source content profile
into the mapping; (C) content prior + learned residual `û_t = g(content) + f(U_src)`, which degrades
gracefully to content when overlap is thin. **Rejected** as a crutch — content is *unlearned*, so
fusing it props the mapping up instead of fixing it; decision was to improve the *learned* mapping
itself. (Irony: §6 later showed content is the *only* honest signal, so a content-anchored design is
back on the table — now for the right reason, see Future plans.)

**Why the mapping fails at thin overlap — it's data scarcity, not capacity.**
- A *more complex* mapping is the wrong fix: in a low-data regime more parameters overfit worse.
- The ~10k overlap floor is a *viability* bar, not *sufficiency*. The mapping is a 64→64-dim function
  fit on ~10k **noisy** point estimates (`U_src`,`U_tgt` are themselves learned) — sparse coverage of
  a 64-dim space. Empirically EMCDR only became reliably best above ~27k overlap.
- Relaxing 5-core → 4/3-core (the project's R1 contingency) adds overlap, but the *new* users are the
  thinnest-history, noisiest ones, and it's bounded by how much overlap actually exists.

**The fixes we adopted (all keep everything learned):**
- *History-grounded source encoder* (pool over source-item embeddings) — a function of history, so it
  enables augmentation, is robust for cold/thin users, and removes the fragile untrained free `U_src`.
- *Don't regress onto the noisy `U_tgt`* — train the bridge **end-to-end on the actual target ranking**
  (BPR) so the noisy learned embedding never serves as a label. (This + history-grounding is, by our
  own derivation, the PTUPCDR family.)
- *Sub-history augmentation* — turn each overlap user into many (partial-history → target) examples,
  matching the cold-start test condition; multiplies effective supervision without inventing data.
- *Semi-supervised CORAL* — exploit the source-only/target-only majority we'd otherwise discard.
- *`d` is a tunable knob* — smaller `d` eases the mapping's data-density problem (curse of
  dimensionality) at the cost of recommender expressiveness; can decouple via a bottleneck.

**Data-prep verification.** Before modeling we audited the preprocessing (statically — PyTorch was
temporarily iCloud-evicted): label encoding; the **drop-{0,3}-then-5-core order** (so the ≥5 guarantee
applies to *labeled* interactions — the order matters and is correct); zero orphans/duplicates;
negative-sampling integrity; the split. One real issue surfaced — a *global* temporal split leaves
some users with no training rows — which the redesign fixes with a **per-user** temporal hold-out.

**The `pop` feature.** The item tower includes a `pop` term (catalog avg-rating + log-standardized
rating count). It was **never ablated** — and §6 makes it the prime suspect for the popularity bias,
hence the `pop`-off ablation in Future plans.

---

## 3. Experiment 2 — The redesign: history-grounded bridge

Decision (reached through discussion): don't bolt content onto EMCDR as a crutch; make the *learned*
mapping itself better. `src/cdr/bridge.py`:

- **History-grounded source encoder.** Source-user rep = learned pool (mean; attention available)
  over the embeddings of the user's source items → a *function of history*. Enables augmentation,
  is robust for thin/cold users, removes the fragile free `U_src`.
- **End-to-end BPR-on-target.** The bridge is trained to *rank the user's real target items*, not to
  regress onto the noisy learned `U_tgt`.
- **Sub-history augmentation.** Each overlap user → many examples per epoch (resampled source subsets).
- **Semi-supervised CORAL alignment.** Source-only / target-only users (the non-overlap majority)
  pull the bridge's output distribution toward the real target-user distribution.
- **Validation early-stopping + best-on-val selection**, per-user temporal hold-out.

All paths validated end-to-end on a tiny smoke pair (Toys→Video Games).

---

## 4. Experiment 3 — Confirmation runs, and the evaluation-protocol journey

Two pairs: **Books → Video Games** (thin, ~12k overlap — where old EMCDR lost) and
**Books → Movies&TV** (rich, ~109k). Same cached recommenders; only the eval protocol changed.
This is where the story turned. Recall@10:

### (A) Random hold-out + uniform negatives
| | bridge | MostPop | feat-transfer | Random |
|---|--:|--:|--:|--:|
| Books→Video Games | **0.332** | 0.549 | 0.220 | 0.095 |
| Books→Movies | **0.497** | 0.563 | 0.211 | 0.097 |
Bridge beats content by +51% / +135% — and **reverses** old EMCDR's thin-pair loss (was 0.163 < 0.211).
But MostPop is suspiciously huge (~0.55).

### (B) Temporal hold-out + uniform negatives
| | bridge | MostPop | feat-transfer | Random |
|---|--:|--:|--:|--:|
| Books→Video Games | **0.326** | 0.465 | 0.187 | 0.095 |
| Books→Movies | **0.445** | 0.468 | 0.199 | 0.097 |
MostPop drops (0.55→0.47). **Why:** holding out a *random* positive tends to pick a user's *popular
core* item (easy); holding out the *latest* item is the genuinely hard "predict the future" case.
Random hold-out was inflating everything. Bridge still beats content decisively (+74% / +123%).

### (C) Temporal hold-out + **popularity-weighted negatives** (the debias) + original head-to-head
| | bridge | emcdr_original | MostPop | feat-transfer | Random |
|---|--:|--:|--:|--:|--:|
| Books→Video Games | 0.074 | 0.096 | 0.080 | **0.165** | 0.098 |
| Books→Movies | 0.102 | 0.100 | 0.059 | **0.194** | 0.099 |
Drawing the 100 negatives ∝ popularity removes MostPop's free edge — and **everything that relied on
popularity collapses to ~random.** Only `feature_transfer` (pure semantic cosine) stays clearly above
random.

---

## 5. Experiment 4 — Redesign vs. Original (apples-to-apples, debiased)

Run in one harness (same recommenders, cohort, hold-out, negatives), the original free-embedding
EMCDR mapping (`emcdr_original`, MLP `U_src→U_tgt`) vs. the new bridge, under protocol (C):

| pair | bridge | emcdr_original | verdict |
|---|--:|--:|---|
| Books→Video Games | 0.074 | 0.096 | indistinguishable (both ≈ random 0.098) |
| Books→Movies | 0.102 | 0.100 | indistinguishable (both ≈ random 0.099) |

**The redesign and the original are a wash under fair evaluation** — both ≈ random. The mapping
architecture was *not* the binding constraint. The redesign's earlier apparent advantage (protocols
A/B) was the same popularity signal the original also had.

---

## 6. The central finding — why it happened

**The learned cross-domain models were trained against *uniform* negatives, so they learned to rank
popular items high.** A held-out positive is itself popularity-skewed; against 100 *random* (mostly
unpopular) negatives, "rank popular high" wins easily → Recall@10 ≈ 0.33–0.50. Replace the negatives
with *equally popular* items and that signal is worthless → Recall@10 ≈ random.

Root cause is **upstream**: the recommenders' item embeddings (`id_factor` + `pop`, trained with
uniform BPR) encode popularity, and every downstream method that scores via those embeddings inherits
it — the bridge, the original mapping, and (by construction) MostPop. The content cosine survives
because it scores by *raw MiniLM semantic similarity* and never touches popularity: it ranks the
held-out item above popular-but-semantically-unrelated negatives.

Two evaluation lessons, in order of impact:
1. **Uniform negatives conflate popularity with quality.** Popularity-weighted (or hard) negatives
   are the honest test for cold-start ranking. This single change flipped the conclusion.
2. **Random hold-out is easier than temporal.** A random positive favors a user's popular core;
   the latest positive is the realistic predict-the-future case.

A note on irony: the content baseline, earlier dismissed as "just averaging, no learning," is the
**only** method that genuinely transfers cross-domain taste under honest evaluation.

---

## 7. What's solid vs. what's open

**Solid.** Data prep (verified). The recommender + bridge code (validated end-to-end, deterministic).
The redesign's *engineering* (history pooling, augmentation, semi-sup, early-stopping all work). The
ablation result (text > categories). The evaluation methodology (debias is the right call).

**Open / negative.** No learned cross-domain method yet beats a simple content cosine under honest
evaluation. The bridge-vs-original question is settled (a wash) but for an unsatisfying reason
(shared popularity bottleneck). MostPop dominance under uniform negatives was an artifact.

---

## 8. Future plans

In priority order, all directly motivated by §6:

1. **Hard-negative / popularity-weighted *training*** (highest leverage). Match training to the
   honest objective: sample training negatives ∝ popularity (or mine hard negatives) for *both* the
   recommenders and the bridge, forcing them to learn semantic discrimination instead of popularity.
   Hypothesis: this is what closes the gap to (and past) `feature_transfer`.
2. **Ablate the `pop` feature / content-anchor the item tower.** `pop` was never ablated; it (and the
   `id_factor`) is the likely popularity crutch. Try `pop`-off, down-weighted `id_factor`, or
   initializing/anchoring item embeddings on the raw MiniLM vectors that `feature_transfer` uses so
   effectively.
3. **Re-run the full 8-pair grid under the debiased harness** — the 8-pair conclusions in §2 were
   under uniform negatives and need re-checking honestly.
4. **Architecture sweeps** once training is fixed: attention pooling vs mean, bridge bottleneck / `d`,
   stronger semi-supervision, and a PTUPCDR-style per-user meta-network.
5. **Robustness & reporting**: ≥2 random data splits + the time split; the N=50 qualitative
   face-validity check; W&B/export for the write-up.
6. **Throughput**: vectorize the per-element Python negative-sampling loop before larger sweeps.

The benchmark to beat is now explicit: **`feature_transfer` (content cosine), ~0.165–0.194 Recall@10
under the debiased protocol.**

---

## 9. Reproducibility

```bash
# original EMCDR full grid (8 pairs × 3 ablations), uniform negatives
PYTHONPATH=src python -m cdr.run_grid --config configs/full.yaml

# history-grounded bridge + original head-to-head, debiased eval (thin + rich pair)
PYTHONPATH=src python -m cdr.bridge --config configs/bridge_full.yaml
#   knobs: holdout (temporal|random), neg_sampling (uniform|popularity), compare_original (bool)
```

- Per-experiment metrics (per-seed mean±std): `results/*.json`; grids: `results/GRID__full.json`,
  `results/BRIDGE_GRID__bridge_full.json`. Original-grid lift tables: `RESULTS.md`. Architecture +
  decisions: `MODEL_CARD.md`, `progress.md`. Model weights are gitignored.
- All runs are CPU + seed 42. The `results/bridge__*` JSONs hold the **final debiased** numbers;
  the uniform-negative numbers (protocols A/B) are preserved in this document and `progress.md`,
  since the JSONs were overwritten by the debiased re-run.

**Operational notes.** Apple MPS yields NaN BPR loss for this model → all runs on CPU. One
`bridge_full` run logged ~10h wall-clock — the machine slept overnight; real compute is minutes once
the recommenders are cached. Mid-project, iCloud Drive evicted `.venv` (PyTorch `libtorch_cpu.dylib`
showed 0 on-disk blocks), hanging any torch import; verified code statically (`py_compile` + review)
until the re-download completed. Full iCloud/operational history is in `progress.md`.
```
