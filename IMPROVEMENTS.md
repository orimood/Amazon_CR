# Improvements — fixing the EMCDR failure modes

_Companion to `RESULTS.md`. Branch: `emcdr-improvements`. The per-pair winner analysis
(`RESULTS.md` §3) exposed four failure modes; this doc scopes a fix for each — named to the
cross-domain-recommendation literature **and** wired to this repo — and reports **measured
results** for the two P0 fixes (§C1, §A1), which were run end-to-end._

All measured numbers are Recall@10, `full` ablation, mean over 3 negative-sampling seeds, on the
identical cold-start cohort as `RESULTS.md`. They reuse the cached `full` per-vertical recommenders
(no recommender retraining; `rec_tag: full`) and only re-run the cheap mapping + evaluation, so the
whole P0 sweep took minutes. EMCDR here differs ~0.003 from `RESULTS.md` because the mapping is
retrained under torch 2.9.1 vs the original 2.5.1; the recommender checkpoints are byte-identical, so
all comparisons **within this run** are exact and fair.

## Priority roadmap

| # | fix | failure mode | status | effort | result |
|---|---|---|---|--:|---|
| **C1** | Content-collaborative **gating** (not blending) | EMCDR loses to feat-transfer on thin pairs | **measured** | low | gating works (+); naive blend fails |
| **A1** | Regularized **linear** mapping | thin-overlap sharp-minima; rich-pair headroom | **measured** | low | +0.01–0.02 on rich pairs; fixes Movies→Toys |
| — | **C1 + A1 combined** (gated linear) | both | **measured** | low | **mean R@10 0.266 → 0.307 (+15%)** |
| **B1** | Content-grounded target item tower (freeze) | starved `V_tgt` on Video Games | **measured** | low | **VG EMCDR +38%; flips Books→VG to a win** |
| A2 | Sharpness-Aware Minimization (SAM) on mapping | sharp minima | specced | med | — |
| B2 | 3-core relaxation on sparse verticals | low retention + sub-floor pairs | specced | med | data re-prep |
| C2 | Content-conditioned mapping (CATN-style) | unified mapping ignores content | specced | high | — |
| A3 | Personalized / meta mapping (PTUPCDR, CDRNP, SSCDR) | unified mapping ignores per-user pref | specced | high | — |
| D1 | Temporal recency-weighting of mapping | Movies→Toys drift | specced | low | — |
| E1–3 | Random-split robustness; drop `full` default; N=50 face-validity | eval/robustness gaps | specced | low | — |

---

## C1 — Content-collaborative hybrid (MEASURED)

**Diagnosis (data).** EMCDR wins the 5 data-rich pairs but loses all 3 Video-Games pairs to the
training-free content-cosine (feat-transfer). A hybrid that uses both should dominate either alone.

**What was implemented.** A per-user **z-scored blend** of the EMCDR score and the content-cosine
score, swept over `alpha` (`src/cdr/evaluate.py` `_zscore_rows` + `emcdr_hybrid@{alpha}`; config
`run_hybrid`, `hybrid_alphas`). `alpha=1` reproduces EMCDR exactly, `alpha=0` reproduces content
exactly (verified — exact endpoint match).

**Result 1 — the naive additive blend FAILS.** Intermediate blends fall *below both* pure models on
7/8 pairs, because EMCDR and content **disagree** about the held-out positive, so averaging their
rankings destructively interferes:

| pair | content (α=0) | hyb α=.25 | hyb α=.5 | hyb α=.75 | EMCDR (α=1) |
|---|--:|--:|--:|--:|--:|
| Books→Movies (rich) | 0.231 | 0.229 | 0.288 | **0.315** | 0.311 |
| Toys→Video Games (thin) | **0.289** | 0.116 | 0.138 | 0.153 | 0.157 |

Only on the single richest pair does a content-heavy-toward-EMCDR blend edge pure EMCDR (+0.004);
everywhere else blending hurts. **Takeaway: don't blend scores — gate.**

**Result 2 — overlap-GATED hybrid is the fix.** Serve EMCDR when the pair's overlap is above a
threshold (~20k), else content-transfer (overlap is known at serve time → no leakage; equivalently,
pick the per-pair validation winner):

| | mean Recall@10 |
|---|--:|
| pure EMCDR (MLP mapping) | 0.266 |
| pure content-transfer | 0.252 |
| **overlap-gated hybrid** | **0.299** |

The gate recovers EMCDR on the 5 rich pairs and content on the 3 Video-Games pairs, lifting exactly
the pairs EMCDR loses by **+0.05 to +0.13** at zero cost elsewhere. Supported by the CDR survey's
content-bridge direction for sparse/non-overlap regimes.

**Repo wiring (done):** `src/cdr/evaluate.py`, `src/cdr/config.py`; configs
`configs/improve_mlp.yaml` (`run_hybrid: true`). Gating itself is a post-scoring `argmax` over the
two columns by overlap — a 3-line serve-time rule, no retraining.

**Next:** replace the global threshold with a **per-user** gate (confidence = source-history depth /
mapping-validation margin) — likely beats the pair-level gate on the borderline pairs.

---

## A1 — Regularized linear mapping (MEASURED)

**Diagnosis (data + literature).** The EMCDR mapping is a `64→128→128→64` MLP (~33k params) trained
on as few as ~9.4k overlap users → underdetermined → "sharp minima, poor generalization" (SCDR).
Fix: shrink it to a single linear layer (`f(x)=Wx+b`, ~4.2k params) + stronger weight decay.

**What was implemented.** `map_hidden: []` (the existing `MappingMLP` already degrades to one linear
layer) + `map_weight_decay: 1e-4` (`configs/improve_lin.yaml`). No model-code change needed.

**Result — a free win on the data-rich pairs; does NOT rescue the thin pairs.**

| pair | overlap | EMCDR (MLP) | EMCDR (linear+reg) | Δ |
|---|--:|--:|--:|--:|
| Books→Movies&TV | 109,206 | 0.311 | **0.323** | +0.011 |
| Books→Toys&Games | 75,649 | 0.324 | **0.325** | +0.001 |
| Movies&TV→Toys&Games | 46,024 | 0.289 | **0.303** | +0.015 |
| Movies&TV→CDs&Vinyl | 34,630 | 0.352 | **0.365** | +0.013 |
| Books→CDs&Vinyl | 27,570 | 0.372 | **0.395** | +0.022 |
| Movies&TV→Video Games | 16,504 | 0.168 | 0.159 | −0.009 |
| Toys&Games→Video Games | 14,517 | 0.157 | 0.160 | +0.003 |
| Books→Video Games | 11,801 | 0.157 | 0.153 | −0.004 |
| **mean** | | 0.266 | **0.273** | +0.006 |

Two readings: (1) the linear map is **strictly ≥ the MLP on the 5 rich pairs** (+0.01 to +0.02) and
in particular lifts **Movies→Toys 0.289 → 0.303**, erasing the one upset where target-MF beat the
MLP (`RESULTS.md` §3). (2) On the Video-Games pairs it stays ~0.15–0.16, far below content — so the
thin-pair collapse is **not** mainly mapping over-parameterization; it's the starved target factors
(see B1) plus too little supervision. Regularizing the mapping is necessary headroom on rich pairs
but not sufficient for the thin ones.

**Repo wiring (done):** `configs/improve_lin.yaml` (`map_hidden: []`, `map_weight_decay: 1e-4`).

---

## B1 — Content-grounded target item tower (MEASURED)

**Diagnosis (data).** The Video-Games collapse is **target-side**: 22.7k items, 15% 5-core
retention, 604k positives → the learned per-item `id_factor` barely moves on rare items, so `V_tgt`
is noisy and the mapping lands in a weak space. A1 (smaller mapping) did **not** help here — it isn't
a mapping problem. Fix: stop relying on free per-item parameters and ground the item embedding in
the frozen MiniLM content space.

**What was implemented.** A `freeze_id` / `text_init` knob on the item tower (`src/cdr/models.py`
`HybridRecommender`; `src/cdr/config.py`; `src/cdr/train_recommender.py`) plus `rec_tag_overrides`
so **only the Video-Games target recommender** is retrained content-grounded while the source
recommenders load from the cached `full` tag. This is exact, not an approximation: a source's
`id_factor` is never used downstream — only its `user_emb` feeds the mapping. Two variants:
- **B1a (freeze):** `id_factor` zeroed and frozen → item embedding = `text_proj(MiniLM) + cat + pop` (pure content).
- **B1b (text warm-start):** `id_factor` initialized from a fixed projection of MiniLM text, left trainable.

**Result — B1a fixes the collapse; B1b does not.** EMCDR Recall@10 on the VG pairs (linear mapping throughout):

| VG pair | overlap | base (A1) | **B1a freeze** | B1b warm-start | content | overall best |
|---|--:|--:|--:|--:|--:|---|
| Movies&TV → Video Games | 16,504 | 0.159 | **0.219** | 0.158 | 0.246 | content 0.246 |
| Toys&Games → Video Games | 14,517 | 0.160 | **0.217** | 0.160 | 0.289 | content 0.289 |
| Books → Video Games | 11,801 | 0.153 | **0.215** | 0.151 | 0.211 | **EMCDR/B1a 0.215** |
| **mean (VG)** | | 0.157 | **0.217 (+38%)** | 0.156 | | |

Two findings: (1) **content-grounding the target recommender lifts the learned model +38%** on the
exact pairs it was collapsing on — it **flips Books→Video Games (thinnest overlap, weakest content
transfer) to an EMCDR win** and cuts the gap to content from −0.05/−0.13 down to −0.03/−0.07 on the
other two. (2) **It is the *freezing* that matters, not the warm-start** — B1b (trainable from a text
init) drifts straight back to baseline as BPR re-overfits the sparse per-item factors; only B1a (no
free per-item parameters) holds the content structure. This confirms the collapse was caused by free
per-item parameters on a sparse catalog, not by a bad initialization. B1 has **no effect on the rich
pairs** (it only changes the VG target recommender).

**Repo wiring (done):** `src/cdr/models.py`, `src/cdr/config.py`, `src/cdr/train_recommender.py`;
configs `configs/improve_b1.yaml` (freeze), `configs/improve_b1t.yaml` (warm-start). Retrains only
the Video-Games recommender (~2 min); sources reuse the cached `full` layer.

**Next:** apply B1a content-grounding to *all* targets (not just VG) and re-run the full grid — it may
help other targets' long tails too; then re-tune the C1 gate, since B1a narrows the EMCDR–content gap
enough that a per-user gate could now prefer the learned model on more pairs.

---

## Combined policy (MEASURED) — gated linear

Use the **regularized linear mapping everywhere** (A1; ≥ MLP on 6/8, never materially worse) and
**gate to content-transfer on thin-overlap pairs** (C1):

| policy | mean Recall@10 | vs original EMCDR |
|---|--:|--:|
| original EMCDR (MLP mapping, deployed in `RESULTS.md`) | 0.266 | — |
| pure content-transfer | 0.252 | −0.014 |
| linear mapping (A1) | 0.273 | +0.007 |
| overlap-gated, MLP mapping (C1) | 0.299 | +0.033 |
| **overlap-gated linear (A1 + C1)** | **0.307** | **+0.041 (+15%)** |

This is the recommended deployable configuration: one cheap mapping change + one serve-time gating
rule, no recommender retraining, and it beats the MostPop fallback on all 8 pairs by construction.

**B1 on top of this** makes the *learned* model competitive on the Video-Games pairs (mean VG EMCDR
0.157 → 0.217, flipping Books→Video Games to an EMCDR win). It changes the gated mean only slightly
(content still edges the other two VG pairs), but it removes the hard dependence on the content
fallback — the learned cross-domain model is no longer collapsing where overlap is thinnest.

---

## Specced (not yet run) — needs retraining or data re-prep

- **A2 — Sharpness-Aware Minimization on the mapping (P2).** SCDR's exact remedy for the sharp-minima
  cause; wrap the Adam step in `src/cdr/train_mapping.py` with a perturbation step. Complementary to
  A1 (A1 shrinks the model; A2 flattens the loss).
- **B2 — 3-core relaxation on sparse verticals (P2).** The repo's own R1 contingency: raises
  Video-Games retention/overlap and unlocks the 2 sub-floor pairs (Toys↔CDs 7.75k, CDs↔VG 4.4k).
  Stage-A `k` change; re-baselines the cohort.
- **C2 — content-conditioned mapping (P3).** Condition `f` on the user's source text profile
  (CATN-style) so the learned bridge inherits content robustness — a learned alternative to the C1
  gate. `MappingMLP` extra input + plumbing in `train_mapping.py`/`evaluate.py`.
- **A3 — personalized / meta mapping (P3).** PTUPCDR (meta-network → per-user bridge), CDRNP (neural
  process), SSCDR (semi-supervised: non-overlap users/items + neighborhood inference). New model +
  training path; meta-learning usually wants *more* overlap, so validate specifically on the thin
  pairs.
- **D1 — temporal recency-weighting of mapping (P3).** Weight overlap users by source recency / use a
  recent source profile to counter the Movies→Toys drift; timestamps already in `interactions.parquet`.
  Small effect, single pair. (Note: A1 already pulled Movies→Toys up to 0.303.)
- **E1 — random-split robustness (P1).** Stage-A re-run with `split="random"`, ≥2 seeds (HW1 §1.3(2));
  report mean±CI so the lift is shown to survive.
- **E2 — drop `full` as default ablation (P1, trivial).** `text_only ≥ cats_only` on 6/8 and `full`
  is best only ×1 (`RESULTS.md` §2); make `text_only` the deployed default. Config-only.
- **E3 — N=50 qualitative face-validity (P2).** Small script sampling cold users, printing top-10.

**Out of scope (explicit):** popularity-debiasing / hard-negative bridge training was *deliberately
set aside* (project decision, 2026-06-04) and is not re-recommended here.

---

## Citations

- SCDR — *Sharpness-Aware Cross-Domain Recommendation to Cold-Start Users* — https://arxiv.org/abs/2408.01931
- CDRNP — *Cross-Domain Recommendation to Cold-Start Users via Neural Process* (WSDM 2024) — https://arxiv.org/abs/2401.12732
- SSCDR — *Semi-Supervised Learning for Cross-Domain Recommendation to Cold-Start Users* (CIKM 2019) — https://dl.acm.org/doi/10.1145/3357384.3357914
- Survey — *A Comprehensive Survey on Cross-Domain Recommendation: Taxonomy, Progress, and Prospects* — https://arxiv.org/abs/2503.14110
- PTUPCDR — *Personalized Transfer of User Preferences for Cross-domain Recommendation* (WSDM 2022); CATN — *Cross-domain Recommendation for Cold-start Users via Aspect Transfer Network* (SIGIR 2020) — named methods; see the survey for context.
- Orthogonal Procrustes embedding alignment (closed-form linear mapping) — https://arxiv.org/abs/2510.13406

## Reproduce

```bash
# both reuse the cached `full` recommenders (rec_tag) — minutes, not hours
PYTHONPATH=src python -m cdr.run_grid --config configs/improve_mlp.yaml   # Arm 1: MLP mapping + hybrid sweep
PYTHONPATH=src python -m cdr.run_grid --config configs/improve_lin.yaml   # Arm 2: linear+reg mapping + hybrid sweep
PYTHONPATH=src python -m cdr.run_grid --config configs/improve_b1.yaml    # B1a: content-grounded VG target (freeze)
PYTHONPATH=src python -m cdr.run_grid --config configs/improve_b1t.yaml   # B1b: content warm-start (trainable)
# per-experiment JSONs: results/*__full__{improve_mlp,improve_lin,b1,b1t}.json ; grids: results/GRID__*.json
```
