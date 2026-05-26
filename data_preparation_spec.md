# Data Preparation — Pre-processing Implementation Spec

**Audience:** an implementation agent (Claude Code) building the data-preparation
pipeline for this project.
**Goal:** turn the raw *Amazon Reviews 2023* snapshot into model-ready inputs for an
EMCDR-style cross-domain recommender, following the CRISP-DM Data Preparation phase
(§3.1–§3.5).
**Provenance:** every decision below traces to the Data Understanding write-up
(`HW2 - Template.docx`, §2.1–§2.4) and the EDA notebook (`notebooks/eda.ipynb`). Where a
number is quoted, the notebook reproduces it.

This document is a *spec*, not code. It says **what** to build, **why** (tied to a §2
finding), and the **shape** of each output. It does not pick the modeling algorithm or
train anything.

---

## 0. Scope

**In scope** — the five CRISP-DM Data Preparation tasks, producing the model-ready
"Data Set":

1. **Select** the data that enters modeling (§3.1).
2. **Clean** it to the quality the model needs (§3.2).
3. **Construct** labels, text embeddings, and category features (§3.3).
4. **Integrate** ratings with metadata and join users across the two domains (§3.4).
5. **Format** everything into the exact files/shapes the training loop consumes (§3.5).

**Out of scope** — training the per-domain base models (MF/NCF), training the
source→target mapping function, hyper-parameter search, and evaluation. The pipeline
stops at *model-ready artifacts*; it defines their schemas so the modeling code has a
stable contract.

**Principles**

- **Config-driven.** Every choice that the team has not locked (below) is a parameter,
  not a hard-coded constant.
- **Reproducible.** One global `SEED`; a pinned input snapshot; each stage idempotent
  (re-running with the same config + input yields identical output).
- **Defensive.** The §2.4 quality results held on the current snapshot but must be
  re-asserted at runtime, because the pipeline may be re-run on a fresh pull.
- **Compute-aware.** Inputs are large (see §1); stream the metadata, and embed text
  **only for items that survive selection** — embedding is the heaviest job (HW1 risk R2).

---

## 1. Inputs

Raw data lives under `data/raw/` (gitignored; produced by `data/download.py`).

**Ratings** — one CSV per vertical at `data/raw/reviews/<Vertical>.csv`, header
`user_id,parent_asin,rating,timestamp`:

| column        | type            | notes |
|---------------|-----------------|-------|
| `user_id`     | string token    | shared across verticals by construction |
| `parent_asin` | string (10-char)| join key to metadata |
| `rating`      | float in `{1..5}`| stored as `X.0` |
| `timestamp`   | int64           | ms since Unix epoch |

**Metadata** — one JSON-Lines file per vertical at `data/raw/meta/meta_<Vertical>.jsonl`,
one item per line. Relevant fields: `parent_asin`, `title`, `description` (list),
`features` (list), `categories` (list = hierarchical path), `main_category`, `store`,
`details` (dict), `price`, `average_rating`, `rating_number`, `images`, `videos`,
`bought_together`, and `author` (Books only).

**Scale** (drives memory/perf choices — stream, don't fully load Books meta):

| vertical       | interactions | items     | reviews CSV | meta JSONL |
|----------------|-------------:|----------:|------------:|-----------:|
| Books          | 29,139,329   | 4,446,065 | 1.69 GB     | 14.70 GB   |
| Movies_and_TV  | 17,158,519   | 747,764   | 1.00 GB     | 1.29 GB    |
| Toys_and_Games | 16,052,440   | 890,667   | 0.93 GB     | 2.64 GB    |
| CDs_and_Vinyl  | 4,772,071    | 701,673   | 0.28 GB     | 0.95 GB    |
| Video_Games    | 4,555,500    | 137,249   | 0.26 GB     | 0.44 GB    |

Read CSVs with a columnar/streaming reader (the EDA used `pyarrow.csv`); stream the
metadata line-by-line. Working budget is ~200 GB on disk.

---

## 2. Configuration

Expose a single config (e.g. `config.yaml` or a dataclass). Defaults shown; **flagged
items are not locked** — see §8.

```yaml
# --- domain pair (NOT LOCKED — §2.3) -------------------------------------
source: Books              # recommended default
target: Movies_and_TV      # recommended default; alt candidate: Toys_and_Games
# the other three verticals are retained only for ablations, not this run

# --- density filter ------------------------------------------------------
k_core: 5                  # iterative k-core, applied per vertical
k_core_overrides: {}       # R1 contingency: e.g. {target: 3} if 5-core overlap < ~10k

# --- rating -> training signal (CONTEMPLATED scheme — §2.1) --------------
positive_threshold: 4      # rating >= 4  -> positive
drop_ratings: [0, 3]       # 3 = neutral/ambiguous (dropped); 0 = invalid (4 Books rows)
explicit_negative: [1, 2]  # kept as known negatives
neg_sample_ratio: 4        # sampled unobserved negatives per positive
explicit_neg_weight: 1.0   # default 1.0; ablation raises it (>1)

# --- text features -------------------------------------------------------
embed_model: sentence-transformers/all-MiniLM-L6-v2
embed_dim: 384
text_fields: [title, description, features]   # concatenated, newline-joined
text_fallback: [title, store, details]        # used when text_fields are empty

# --- splits --------------------------------------------------------------
split: temporal            # temporal (chronological) is primary; 'random' is the seeded alt
split_ratios: [0.8, 0.1, 0.1]   # train / val / test
seed: 42
```

**Why the pair is a parameter.** §2.3 deliberately did **not** lock the source→target
pair. The pipeline must run end-to-end for any `(source, target)` so the team can compare
Books→Movies (highest overlap, 127k shared users) against Books→Toys (87k, richer target
metadata) before committing. Build for the parameter, not the default.

---

## 3. Pipeline stages

### 3.1 Select Data → *Rationale for Inclusion / Exclusion*

**Verticals.** Use only the configured `source` and `target`. (All five were collected so
this pick is data-driven; the other three are ablation reserves.)

**Fields.** Keep what the model consumes; drop the rest at read time.

| keep (ratings)               | keep (metadata)                                   | drop |
|------------------------------|---------------------------------------------------|------|
| `user_id`, `parent_asin`, `rating`, `timestamp` | `title`, `description`, `features`, `categories`, `main_category`, `store`, `details`, `author` (Books) | `images`, `videos`, `price`, `subtitle`, `bought_together`, `average_rating`, `rating_number`* |

\* `average_rating` / `rating_number` are low-priority popularity priors — keep them only
if the baseline wants a popularity feature; otherwise drop. `bought_together` is dropped
because it is **100% null** across all 6.9M items (§2.4).

**Row selection.**

1. Drop invalid ratings: `rating in drop_ratings` removes the four Books `0` rows and all
   `3` ("neutral") rows (§2.4 / §2.1). *(3s are dropped from the training signal but stay
   in the per-user "seen set" — see §3.3.)*
2. Apply the **iterative k-core filter** per vertical: repeatedly drop users with `< k`
   interactions and items with `< k` interactions until the table is stable. This is the
   density filter motivated by the pervasive cold-start in §2.2 (median user rated one
   item). Honor `k_core_overrides` for the R1 contingency.

> **Important ordering.** Run k-core on the **interaction set used for training**, i.e.
> after deciding which interactions count. Decide explicitly whether 3-star rows are
> present during k-core counting; recommended: drop `0`/`3` *before* k-core so the core is
> computed on the interactions the model actually trains on, and recompute the per-user
> seen set from the *unfiltered* interactions separately (§3.3).

**Output of this stage:** the selected raw subset for the pair, plus a short written
rationale (counts kept/dropped at each step) — this *is* the CRISP-DM "Rationale for
Inclusion/Exclusion" and should be logged to `meta.json`.

### 3.2 Clean Data → *Data Cleaning Report*

The snapshot is already clean (§2.4); cleaning is mostly *guarding* and a few field fixes.

- **Drop unused/empty fields** at ingest (see table above), especially `bought_together`.
- **Missing rich text → fallback, never drop the item.** When `text_fields` are all empty
  for an item, fall back to `text_fallback` (`title` + `store` + `details`). If *still*
  empty, the item gets a placeholder embedding in §3.3 — it is **not** removed.
  *Why not remove:* deleting a text-poor item also deletes its interactions, which can
  cascade into dropping its users (re-introducing the sparsity k-core just removed), and
  biases the catalog against exactly the cold, long-tail items cross-domain transfer is
  meant to help (§2.4). Missing text is represented as empty list / null — no sentinel
  codes.
- **`main_category` is not the partition.** It holds the live store label (a Movies record
  may read "Comedy"). The vertical of an item is the **file it came from**, never
  `main_category` (§2.2/§2.4). Tag every item with its file-of-origin vertical.
- **Normalize categorical strings.** `store` and `categories` tokens carry casing and
  near-duplicate variants (§2.4): lowercase + strip + collapse whitespace before using as
  identifiers; keep a raw copy.
- **Quality guards (assert, don't assume — §2.4):**
  - every ratings row parses into exactly 4 fields;
  - `rating in {1..5}` after the zero-drop;
  - no null `user_id` / `parent_asin`;
  - timestamps positive and within `[1996-01-01, 2024-01-01)`;
  - **0 orphan interactions** (every `parent_asin` in ratings has a metadata record) — fail
    loudly if any appear on a re-pull;
  - **0 duplicate** `(user_id, parent_asin)` pairs (dedupe defensively, keeping the latest
    timestamp).

### 3.3 Construct Data → *Derived Attributes + Generated Records*

**(a) Labels — the rating→signal scheme (CONTEMPLATED, §2.1; keep it swappable).**

- `rating >= positive_threshold (4)` → **positive** (label 1).
- `rating in {1,2}` → **explicit negative** (label 0), kept and **always included** in the
  negative set (do not rely on random sampling to surface these scarce known dislikes).
- `rating == 3` → **dropped** from labels (too ambiguous), but retained in the seen set.
- **Sampled negatives:** for each positive, draw `neg_sample_ratio` items the user never
  interacted with — uniform over the *unobserved* catalog, excluding the user's full **seen
  set**, and never treating a 1–2-star item as "unobserved". Decide and state whether
  negatives are **precomputed and frozen** into the interaction tables (seed with `SEED`
  for reproducibility) or **sampled at train time** (then the prep must ship the per-user
  seen set and the item-id space so the loop samples correctly). Default: precompute, for a
  fixed, inspectable dataset.
- **Weights:** explicit negatives carry `explicit_neg_weight` (default `1.0`; the ablation
  raises it to test whether emphasizing known dislikes helps). Sampled negatives weight 1.
- **Same encoding for the single-vertical baseline and the cross-domain model**, so the
  proof-of-value comparison is like-for-like.

> Because Amazon ratings are heavily positive-skewed (≥4 is 74.5–87.3%; §2.2), explicit
> negatives are scarce — most negative signal still comes from sampled unobserved items.

**(b) Seen set vs positive set (two distinct per-user sets).**

- **positive set** (≥4) → trains the model / forms the user representation.
- **seen set** (every interaction the user had, *any* rating incl. 3s and 1–2s) → used
  only at inference to exclude already-seen items from recommendations. Persist both; do
  not conflate them.

**(c) Item text embedding (derived attribute).**

- Build item text = newline-join of `text_fields` (fallback chain per §3.2).
- **Truncate** item text to the model's max sequence length (all-MiniLM-L6-v2 = 256
  word-pieces) before encoding.
- Encode with `embed_model` → `embed_dim`-vector per item, in batches (e.g. 256–1024) on
  GPU if available, else CPU. Record whether outputs are L2-normalized in `meta.json`.
- **Cache to disk** keyed by `parent_asin` + a hash of the (truncated) text, so re-runs and
  ablations skip re-encoding.
- **Embed only items that survive §3.1 selection for the chosen pair** (not all 6.9M) —
  this is the main compute lever (R2).
- **Generated record — the placeholder.** Items with no usable text get a single shared
  "no-content" vector (e.g. a fixed zero/learned sentinel embedding). Carry a boolean
  `has_text` so the model can treat them differently if desired.

**(d) Category feature.** Encode the hierarchical `categories` path as categorical
id(s) / multi-hot. Keep it as a separate feature so the **categories-only ablation**
(§2.1) can run without the text vector.

**(e) Single-attribute transforms.** Timestamps → int seconds (or datetime) for splitting;
optionally L2-normalize embeddings if the base model expects unit vectors.

### 3.4 Integrate Data → *Merged Data*

- **Within-vertical join:** ratings ⨝ metadata on `parent_asin` to attach item features to
  interactions. Expected to be complete (0 orphans, §2.4) — assert it.
- **Cross-vertical user join (the heart of EMCDR):** intersect `user_id` sets of `source`
  and `target` *after* k-core to produce the **shared-user index**. These overlapping users
  are the supervised training set for the source→target mapping — each contributes one
  (source-embedding, target-embedding) training pair. Record the count and fail-soft if it
  is below ~10k (the planning-phase floor for fitting the mapping; triggers the R1
  `k_core_overrides` contingency).
- **Aggregations:** per-user interaction histories (for sequence/history features and the
  seen-set filter) and, if needed, per-item popularity counts.

### 3.5 Format Data → *Reformatted Data*

- **ID remapping:** map `user_id` and `parent_asin` to contiguous integer indices (embedding
  layers need dense ids). Maintain **per-domain item maps** and a **user map** that is
  consistent across both domains (so a shared user has one `user_idx`). Persist the maps.
- **Splits:** assign each interaction to train/val/test.
  - *temporal (primary):* sort by `timestamp`, cut chronologically by `split_ratios` so the
    model trains on earlier interactions and is evaluated on later ones (§2.2). Use a global
    cut (or per-user last-out by time — state which; global is the default).
  - *random (seeded alt):* reproducible shuffle with `seed`.
- **Column order / shapes:** emit tidy tables with a stable schema (§4).
- Serialize as Parquet (tables) + NPZ/`.npy` (dense matrices). Write a `meta.json`
  describing the run (this is the CRISP-DM "Data Set Description").

---

## 4. Output artifacts (the "Data Set")

Write under `data/processed/<source>__<target>/`:

| file | schema / contents |
|------|-------------------|
| `source_interactions.parquet` | `user_idx, item_idx, label (0/1), weight, rating, timestamp, split` |
| `target_interactions.parquet` | same schema as source |
| `source_item_features.npz` | `item_idx → text_emb[384]`, `category_ids`, `has_text` |
| `target_item_features.npz` | same |
| `shared_users.parquet` | `user_idx` present in both domains (mapping supervision) + per-domain activity counts |
| `seen_sets.parquet` | `user_idx → [item_idx,...]` per domain (inference-time exclusion) |
| `id_maps.json` (or parquet) | `user_id↔user_idx`, `parent_asin↔item_idx` per domain |
| `meta.json` | full config, embed-cache hash, seed, snapshot date, and the L2-normalize choice |
| `stats.json` (+ `STATS.md`) | the **§3 reporting numbers** (see §8): per-step counts, label distribution, fallback/placeholder counts, shared-user overlap, split sizes — machine-readable, with a human-readable mirror the report is written from |

`meta.json` + `stats.json` together are the CRISP-DM **Data Set Description**, **Rationale
for Inclusion/Exclusion**, and **Data Cleaning Report** for this phase.

---

## 5. Quality guards (from §2.4 — assert at runtime)

These passed on the current snapshot; encode them as assertions so a re-pull can't silently
break the pipeline:

- ratings rows all 4-field; `rating ∈ {1..5}` post zero-drop; no null keys.
- timestamps positive, within `[1996, 2024)`.
- 0 orphan interactions after the within-vertical join.
- 0 duplicate `(user_id, parent_asin)` pairs (dedupe → latest timestamp).
- `bought_together` fully null → confirm before dropping.
- shared-user overlap ≥ ~10k after k-core, else log a warning and apply `k_core_overrides`.

---

## 6. Suggested module layout & order

```
src/preprocess/
  config.py        # dataclass / yaml loader
  io.py            # streaming readers for CSV + JSONL
  select.py        # §3.1  field/row selection + k-core
  clean.py         # §3.2  guards, fallback text, normalization
  construct.py     # §3.3  labels, negatives, embeddings, categories
  integrate.py     # §3.4  joins + shared-user index + aggregations
  format.py        # §3.5  id remap, splits, serialization
  run.py           # orchestrates 3.1→3.5, writes meta.json
tests/             # assert the §5 guards on a small sample
```

Implement in order 3.1 → 3.5. Make each stage read the previous stage's artifact so stages
are independently runnable and cacheable. Put the §5 guards in `tests/` against a small
sampled subset for fast CI, and as runtime asserts in the real run.

---

## 7. Performance notes

- Books ratings = 29M rows; use columnar/streaming reads and avoid materializing all of
  Books metadata (14.7 GB) — stream it once, keep only kept fields for selected items.
- Text embedding is the dominant cost (R2): embed **only selected, surviving items**, batch
  on GPU, and cache by content hash so re-runs and ablations are cheap.
- k-core and the cross-vertical intersection are cheap relative to embedding — they run on
  two columns of the ratings tables in seconds to a minute even for Books.

---

## 8. Reporting outputs the run must emit (for HW2 §3)

The HW2 §3 write-up is produced **from these numbers**, so the run must compute and persist
them to `stats.json` (mirrored to `STATS.md`), **for the configured pair**, logging counts at
every transition so nothing is claimed without evidence. They map one-to-one to the gaps the
EDA notebook does not cover.

**§3.1 Select**
- per domain: 0-core interactions / users / items;
- rows dropped for `rating == 0` and for `rating == 3`;
- after the post-drop 5-core: interactions / users / items, and retention % vs 0-core.

**§3.2 Clean**
- orphan-interaction count (expect 0) and duplicate-pair count (expect 0);
- per-field missing-value rate for the kept fields, for the two selected verticals.

**§3.3 Construct**
- label distribution per domain: # positives (≥4), # explicit negatives (1–2), # dropped (3);
- # items using the text fallback, and # items with **no usable text** (→ placeholder
  embeddings), per domain;
- # items embedded, embedding dimension, cache hits; # distinct category paths.

**§3.4 Integrate**
- shared-user count for the pair after 5-core, plus each domain's user count;
- join coverage (share of interactions whose item has metadata — expect 100%).

**§3.5 Format**
- id-space sizes (`n_users`, `n_items` per domain);
- temporal split: train / val / test interaction counts and the two cutoff timestamps;
  seeded-split sizes.

Once `stats.json` exists, §3.1–§3.5 of the report can be written directly from it (and the
§2.4-style quality numbers — duplicates, rating sanity, flat-file structure — should be
emitted here too, since the prep run is the natural place to reconfirm them on the snapshot).

---

## 9. Run & environment

- **Environment:** the project virtualenv (`.venv`) — it has `pyarrow`, `pandas`,
  `sentence-transformers`, and `torch`. Do **not** run on a small-memory box: Books is 29M
  interactions and 14.7 GB of metadata, and the MiniLM embedding is the dominant cost
  (HW1 risk R2). Use the project workstation (≈200 GB disk budget) and a GPU for the
  embedding step if available.
- **Command (after implementing per §6):** `python -m src.preprocess.run --config config.yaml`.
- **Smoke test first:** run end-to-end on a small pair (e.g. CDs_and_Vinyl → Video_Games) or a
  row-capped sample to validate the §5 guards and the §4 output schemas before the full run.
- **Pairs to run:** the default `Books → Movies_and_TV`, then re-run with `source/target` set
  to `Books → Toys_and_Games`, so §2.3's overlap-vs-richness decision can be made on real
  prepared-data sizes.
- **Outputs:** `data/processed/<source>__<target>/` with the §4 artifacts plus `stats.json` /
  `STATS.md` (§8) and `meta.json`.
- **Idempotent & cached:** re-running with the same config + snapshot reproduces identical
  outputs; the text-embedding cache makes pair re-runs and ablations cheap.

---

## 10. Open decisions to confirm before locking (carry as parameters)

1. **Source→target pair** — default Books→Movies_and_TV; alternative Books→Toys_and_Games
   (§2.3). Run both, compare, then lock.
2. **Rating→signal scheme** — the ≥4-positive / 3-drop / 1–2-explicit-negative scheme is the
   team's *current best guess*, not final (§2.1). Keep label construction swappable; the
   `explicit_neg_weight` upweight is an ablation, not the default.
3. **k-core threshold** — 5 by default; relax the *secondary* vertical to 3 only if its
   5-core shared-user overlap falls below ~10k (HW1 R1 contingency).
4. **Split variant** — temporal (global chronological cut) is primary; confirm whether a
   per-user last-out-by-time split is also wanted.
