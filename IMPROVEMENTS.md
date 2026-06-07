# Improvements — fixing the EMCDR failure modes

_Companion to `RESULTS.md`. Branch: `emcdr-improvements`. The per-pair winner analysis
(`RESULTS.md` §3) exposed the failure modes; this doc scopes a fix for each — named to the
cross-domain-recommendation literature **and** wired to this repo — and reports **measured
results** for five fixes (A1, C1, B1, C2, B2), all run end-to-end. See the Summary below for the
final outcome._

All measured numbers are Recall@10, `full` ablation, mean over 3 negative-sampling seeds, on the
identical cold-start cohort as `RESULTS.md`. They reuse the cached `full` per-vertical recommenders
(no recommender retraining; `rec_tag: full`) and only re-run the cheap mapping + evaluation, so the
whole P0 sweep took minutes. EMCDR here differs ~0.003 from `RESULTS.md` because the mapping is
retrained under torch 2.9.1 vs the original 2.5.1; the recommender checkpoints are byte-identical, so
all comparisons **within this run** are exact and fair.

## Summary — what changed and what it bought

**Starting point** (`RESULTS.md` §3): the 8-pair EMCDR beat the deployed MostPop fallback
everywhere, but lost to a **training-free content-cosine** baseline on the 3 thin Video-Games pairs,
and to **target-MF** on Movies→Toys. Five fixes were implemented and measured end-to-end:

| fix | what it is | measured outcome |
|---|---|---|
| **A1** | regularized **linear** mapping (vs the ~33k-param MLP) | +0.01–0.02 R@10 on the 5 data-rich pairs; **fixes the Movies→Toys upset** (0.289→0.303). No help on the thin VG pairs. |
| **C1** | content-collaborative **gating** (not score-blending) | the additive blend **fails** (destructive interference); the overlap-**gated** best-of lifts mean R@10 **0.266 → 0.299**. |
| **B1** | content-grounded target item tower (freeze `id_factor`) | fixes the VG collapse: VG EMCDR **+38%**, flips **Books→VG** to a win. It's the *freeze* that matters, not the warm-start. |
| **C2** | residual content-conditioned bridge | **negative result** — does not beat content; the learned residual doesn't generalize from thin overlap. The *wrong* lever. |
| **B2** | 3-core Video Games (denser target + ~2× overlap) | flips **Movies→VG** to a win (0.339 vs 0.303) and widens Books→VG. The *right* lever for the thin pairs. |

**Combined deployable policy:** regularized **linear mapping** everywhere (A1) + **content-grounded
target tower** for sparse targets (B1) + **3-core** for the sparse VG target (B2) + a serve-time
**overlap gate to content** where it still wins (C1). On the 5-core 8-pair set, mean R@10 rose from
**0.266 → 0.307** (A1 + C1); with B1/B2 the learned model went from losing 3 pairs to losing 1.

**Final scorecard — best model per pair** (✅ = our learned EMCDR is best; ❌ = content-transfer wins):

| pair | best model | beats content? | beats MostPop? |
|---|---|:--:|:--:|
| Books → Movies&TV | EMCDR (linear) | ✅ | ✅ |
| Books → Toys&Games | EMCDR (linear) | ✅ | ✅ |
| Movies&TV → Toys&Games | EMCDR (linear) | ✅ | ✅ |
| Movies&TV → CDs&Vinyl | EMCDR (linear) | ✅ | ✅ |
| Books → CDs&Vinyl | EMCDR (linear) | ✅ | ✅ |
| Movies&TV → Video Games | EMCDR (B1 + B2 3-core) | ✅ | ✅ |
| Books → Video Games | EMCDR (B1 + B2 3-core) | ✅ | ✅ |
| Toys&Games → Video Games | content-transfer (gated) | ❌ | ✅ |

**Net: the learned EMCDR is the best model on 7 of 8 pairs (up from 5/8), and every pair beats the
deployed MostPop fallback.** The lone holdout — Toys→VG — has the strongest content affinity of any
pair, so the C1 gate ships content there. *Caveat:* the two Video-Games rows use the 3-core setup, so
their absolute R@10 is **not** comparable to the 5-core rows; the ✅/❌ is the per-pair, like-for-like
signal.

**Also produced:** a reusable diagnostic — the **warm-oracle** (rank cold users by their real target
embedding) — which cheaply indicates, *before* building any bridge, whether a pair's collaborative
signal can rival content. (B2 showed it is a strong reference, but not a strict ceiling.)

## Priority roadmap

| # | fix | failure mode | status | effort | result |
|---|---|---|---|--:|---|
| **C1** | Content-collaborative **gating** (not blending) | EMCDR loses to feat-transfer on thin pairs | **measured** | low | gating works (+); naive blend fails |
| **A1** | Regularized **linear** mapping | thin-overlap sharp-minima; rich-pair headroom | **measured** | low | +0.01–0.02 on rich pairs; fixes Movies→Toys |
| — | **C1 + A1 combined** (gated linear) | both | **measured** | low | **mean R@10 0.266 → 0.307 (+15%)** |
| **B1** | Content-grounded target item tower (freeze) | starved `V_tgt` on Video Games | **measured** | low | **VG EMCDR +38%; flips Books→VG to a win** |
| **C2** | Residual content-conditioned bridge | beat content on the last 2 VG pairs | **measured** | low | does NOT beat content (wrong lever); content's signal dominates the thin VG pairs |
| **B2** | 3-core Video Games (denser target + 2× overlap) | thin/sparse VG target | **measured** | med | **EMCDR beats content on Movies→VG & Books→VG; only Toys→VG still content** |
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

## C2 — residual content-conditioned bridge, and the content ceiling (MEASURED)

**Goal.** Beat content-transfer on the last two pairs it still wins (Movies/Toys → Video Games).

**What was implemented (`src/cdr/run_c2.py`).** `score = content_scale·cos(profile_src, text_tgt) +
(Linear[U_src; profile_src])·V_tgt`, trained with BPR on overlap-train users' target positives
(reuses the cached `full` sources + the B1 VG target). Idea: keep content as a floor, learn a
collaborative residual on top.

**Result — C2 does NOT beat content; it underperforms even EMCDR(B1).**

| pair | content | C2 | EMCDR(B1) | **warm-oracle (ceiling)** |
|---|--:|--:|--:|--:|
| Movies&TV → Video Games | 0.246 | 0.197 | 0.219 | 0.248 |
| Toys&Games → Video Games | 0.289 | 0.176 | 0.217 | 0.189 |
| Books → Video Games | 0.211 | 0.194 | 0.215 | 0.224 |

The residual term's magnitude grew during BPR training and dominated the content floor (`content_scale`
stayed ≈1), so C2 collapsed toward the learned term — the same destructive interference as the C1
blend, now learned. The learned collaborative signal does not generalize from the ~10k overlap users
to cold users, however it is conditioned.

**The decisive diagnostic — the warm-oracle.** We then ranked the cold-eval users with their *real*
target embedding `U_tgt` — an oracle that already knows their actual Video-Games taste. This is the
**ceiling for any cold-start collaborative method**: you cannot map a cold user to anything better
than their own true embedding. It settles whether "beat content" is even possible:

- **Toys→VG: ceiling 0.189 ≪ content 0.289.** Even perfect knowledge of the user's VG embedding loses
  to content by 0.10. **No mapping, bridge, or supervision augmentation can beat content here** — Toys
  and Video Games share so much catalog semantics (franchises/hobby items) that text similarity is a
  strictly better ranking signal than the sparse VG collaborative space. The only way to win is to
  **raise the ceiling** (denser/richer target representation), not to build a cleverer bridge.
- **Movies→VG: ceiling 0.248 ≈ content 0.246**, with EMCDR(B1) at 0.219 — i.e. ~0.03 of cold→warm
  headroom is still on the table. Collaborative *can* tie content here, so a better-*supervised*
  mapping is the route.
- **Books→VG: ceiling 0.224 > content 0.211**, and EMCDR(B1) 0.215 already wins.

**Revised "beat everything" verdict (then settled by B2 below):**
- **Books→VG — won** (B1; B2 widens it).
- **Movies→VG — has headroom → B2 wins it** (denser target + ~2× overlap; see below).
- **Toys→VG — the lone holdout;** content's signal there is the strongest of any pair.

**Takeaway.** On the last two pairs the binding constraint is the **target-domain collaborative
signal**, not the bridge. The intuitive next step (a content-aware bridge, C2) is measurably the
**wrong lever**; **B2 (more/denser target data) is the right one** — it is what finally beats content
on Movies→VG. *Caveat on the warm-oracle:* B2 shows it is a **reference, not a strict ceiling** — at
3-core EMCDR exceeds the warm-oracle on all three VG pairs, because a regularized, content-grounded
mapped vector generalizes to the held-out TEST positive better than the user's own train-fit
embedding. So the 5-core reading "Toys→VG unwinnable by any mapping" was too strong; denser data
lifts the collaborative side substantially (Toys→VG EMCDR 0.217 → 0.279) — content simply wins there
by signal strength, not an absolute ceiling.

---

## B2 — 3-core Video Games: denser target + 2× overlap (MEASURED)

**What was built.** A 3-core Video Games vertical `Video_Games_3core` (a NEW vertical — the 5-core
data and cached recommenders are untouched). 3-core recovers far more of the catalogue and audience,
and roughly doubles every pair's overlap:

| | 5-core | 3-core |
|---|--:|--:|
| users | 80,886 | 278,971 |
| items | 22,746 | 48,139 |
| retention vs 0-core | 15.0% | 33.3% |
| overlap Movies→VG | 16,504 | 34,967 |
| overlap Toys→VG | 14,517 | 32,930 |
| overlap Books→VG | 11,801 | 26,206 |

Code: `data/build_vg_3core.py` (Stage A + minimal bridges); configs `improve_b1_3core.yaml`
(content-grounded recommender + EMCDR/content/hybrid) and `improve_c2_3core.yaml` (warm-oracle).

**Result — at 3-core the cold-start EMCDR beats content on 2 of the 3 VG pairs.**

| pair | overlap | EMCDR | content | EMCDR wins? |
|---|--:|--:|--:|:--:|
| Movies&TV → Video Games | 34,967 | **0.339** | 0.303 | ✅ (was ❌ at 5-core) |
| Books → Video Games | 26,206 | **0.308** | 0.216 | ✅ (decisive) |
| Toys&Games → Video Games | 32,930 | 0.279 | **0.364** | ❌ |

Denser target factors (B1 on the 3-core catalogue) plus ~2× overlap (a better-supervised mapping)
push the learned model above content on **Movies→VG** — exactly the pair the 5-core oracle flagged as
having headroom — and widen the **Books→VG** win. **Toys→VG stays with content**, whose 0.364 is the
strongest content score of any pair (Toys and Video Games share extreme catalogue affinity —
franchises, hobby items). The hybrid blend again underperforms both (hyb@0.5 ≈ 0.21–0.24) — gating,
not blending, remains the right combine.

**Caveat — eval setups differ across k.** 3-core changes the cohort, the candidate pool (48k vs 22k
items) and the held-out positives, so 3-core absolute numbers are **not** comparable to the 5-core
ones; the valid, like-for-like claim is the **within-3-core EMCDR-vs-content** comparison (identical
cohort/candidates per pair).

**Final scorecard (best learned config per pair, A1 + B1 + B2).** The learned EMCDR is now the best
model on **7 of 8 pairs**; only **Toys→VG** is still best served by training-free content-transfer,
and the C1 gate deploys content there. Every pair beats the deployed MostPop fallback.

---

## Specced (not yet run) — needs retraining or data re-prep

- **Toys→VG (only remaining loss).** Content's affinity here (0.364) is exceptionally strong; the
  collaborative side improved with 3-core (0.217 → 0.279) but not enough. Options if pursued: even more
  overlap/target density (2-core, or a richer item encoder), or accept the content fallback the C1 gate
  already deploys. SSCDR-style semi-supervised augmentation (pseudo-label non-overlap users) is the
  mapping-side lever, but the gap to 0.364 is large.
- **Extend B2 / 3-core to unlock sub-floor pairs.** The same 3-core build makes Toys↔CDs (7.75k) and
  CDs↔Video Games (4.4k) viable; rebuild those bridges to widen the deliverable from 8 to 10 pairs.
- **A2 — Sharpness-Aware Minimization on the mapping (P2).** SCDR's exact remedy for the sharp-minima
  cause; wrap the Adam step in `src/cdr/train_mapping.py` with a perturbation step. Complementary to
  A1 (A1 shrinks the model; A2 flattens the loss).
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
PYTHONPATH=src python -m cdr.run_c2   --config configs/improve_c2.yaml    # C2 residual bridge + warm-oracle ceiling
# B2 — 3-core Video Games (denser target + 2x overlap):
PYTHONPATH=src python data/build_vg_3core.py                              # build Video_Games_3core + bridges (~7 min, MPS)
PYTHONPATH=src python -m cdr.run_grid --config configs/improve_b1_3core.yaml  # B2: EMCDR + content + hybrid on 3-core
PYTHONPATH=src python -m cdr.run_c2   --config configs/improve_c2_3core.yaml  # B2: warm-oracle on 3-core
# per-experiment JSONs: results/*__full__{improve_mlp,improve_lin,b1,b1t,c2,b1_3core,c2_3core}.json
```
