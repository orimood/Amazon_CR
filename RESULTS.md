# Results — EMCDR Cross-Domain Recommender (full run)

_Run: 2026-06-01, `configs/full.yaml`, CPU, ~4h38m wall. Source data: `results/GRID__full.json` + `results/{src}__{tgt}__{ablation}__full.json`._

All 8 viable vertical pairs × {full, cats_only, text_only} = **24 experiments**, plus 5 per-vertical
recommenders (trained once per ablation, reused across pairs) and 8 per-pair EMCDR mappings.

**Setup.** d=64, BPR (15 epochs), mapping MLP [128,128] (200 epochs). Evaluation: leave-one-out,
1 held-out target test positive + 100 sampled unseen negatives; metrics averaged over 3
negative-sampling seeds. Cohort: cold-start cross-vertical users (held-out overlap users, target
history hidden, scored only through the mapping). Single **temporal** split (the time-based split).

## 1. EMCDR vs. baselines (deliverable, `full` ablation)

R@10 = Recall@10, N@10 = NDCG@10. "beats all" = EMCDR ≥ every baseline on R@10.

| pair | cohort | EMCDR R@10 | EMCDR N@10 | target-MF | feat-transfer | MostPop | Random | lift vs MostPop | beats all |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Books → Movies&TV | 5918 | **0.308** | 0.158 | 0.293 | 0.231 | 0.191 | 0.097 | +61% | ✅ |
| Books → Toys&Games | 2416 | **0.327** | 0.191 | 0.318 | 0.299 | 0.238 | 0.102 | +37% | ✅ |
| Movies&TV → Toys&Games | 1325 | **0.286** | 0.167 | 0.303 | 0.273 | 0.226 | 0.096 | +27% | ❌ |
| Movies&TV → CDs&Vinyl | 2009 | **0.351** | 0.191 | 0.316 | 0.262 | 0.238 | 0.096 | +47% | ✅ |
| Books → CDs&Vinyl | 1524 | **0.369** | 0.200 | 0.321 | 0.207 | 0.230 | 0.102 | +60% | ✅ |
| Movies&TV → Video Games | 647 | **0.170** | 0.082 | 0.177 | 0.246 | 0.131 | 0.100 | +29% | ❌ |
| Toys&Games → Video Games | 943 | **0.164** | 0.085 | 0.122 | 0.289 | 0.095 | 0.094 | +73% | ❌ |
| Books → Video Games | 435 | **0.163** | 0.082 | 0.164 | 0.211 | 0.142 | 0.093 | +15% | ❌ |

Baselines: **target-MF** = single-vertical MF applied to a cold user via the mean user vector
(HW1's literal single-vertical baseline); **feat-transfer** = Model B, cosine of mean source
liked-item text vs. target item text in the shared MiniLM space; **MostPop** = target popularity
(the deployed "popular this week" fallback); **Random** = floor (≈10/101 ≈ 0.099, confirming
calibration).

**Findings.**
- **EMCDR beats the deployed MostPop fallback on 8/8 pairs** (+15% to +73%) — it clears HW1's
  business baseline everywhere.
- **EMCDR beats *every* baseline on the 4 high-overlap pairs** (Books→Movies, Books→Toys,
  Movies→CDs, Books→CDs).
- On the **thin-overlap pairs (the 3 Video Games targets, just above the 10k floor) the simple
  feature-transfer (Model B) wins.** This is the **HW1 R1 risk** in evidence: with little shared-user
  supervision the learned mapping is undertrained, and naive content-cosine transfer is the better
  tool. Cohorts there are also small (435–943), so noisier.

## 2. Ablation — is text worth keeping on top of categories? (HW1 §1.1)

EMCDR R@10 by content variant; **bold** = best of the three.

| pair | full | cats_only | text_only | best |
|---|--:|--:|--:|:--:|
| Books → Movies&TV | 0.308 | 0.294 | **0.313** | text_only |
| Books → Toys&Games | 0.327 | **0.338** | 0.333 | cats_only |
| Movies&TV → Toys&Games | 0.286 | **0.315** | 0.302 | cats_only |
| Movies&TV → CDs&Vinyl | 0.351 | 0.316 | **0.371** | text_only |
| Books → CDs&Vinyl | 0.369 | 0.336 | **0.380** | text_only |
| Movies&TV → Video Games | 0.170 | 0.128 | **0.177** | text_only |
| Toys&Games → Video Games | 0.164 | 0.147 | **0.167** | text_only |
| Books → Video Games | **0.163** | 0.139 | 0.159 | full |

Best-variant tally: **text_only ×5, cats_only ×2, full ×1.**

**Finding.** Text is the stronger signal — `text_only ≥ cats_only` on 6/8 pairs — and **`full`
(text + categories) is almost never best**: stacking the category multi-hot on top of text usually
does not help and sometimes hurts. Practical read for §1.1: *keep the text embeddings; categories
add little once text is present.*

## 3. Caveats / next steps
- **Single temporal split.** HW1 §1.3(2) requires the lift to survive ≥2 random splits + the time
  split. Not yet run (deferred: Stage-A re-run with `split="random"`, ≥2 seeds).
- **Thin-overlap underperformance** (the Video Games pairs) is the R1/R3 diagnostic target: grow
  overlap (3-core contingency) and/or raise mapping regularization.
- Per-experiment numbers (all metrics, per-seed, mean±std) are in `results/*.json`; weights are gitignored.
