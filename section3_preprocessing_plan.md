# Section 3 — Data Preparation: Preprocessing Plan

**Project:** Amazon Reviews 2023 — Cross-Domain Recommendation (EMCDR)
**Phase:** CRISP-DM §3 Data Preparation
**Purpose:** Implementation spec for Claude Code. Every preprocessing step needed to fill §3.1–§3.5 of the HW2 template, derived directly from the §2 (Data Understanding) findings. The last section, *Final Preprocess*, is the single consolidated pipeline to build.

---

## Architecture: per-vertical core + parameterized pair bridge

The pipeline is split into two layers, because most preprocessing is **not** pair-specific:

- **Per-vertical layer (runs once for all 5 verticals, cached):** load, clean, 5-core filter, label encoding, item-text embedding, category + popularity features. None of this depends on the chosen pair — Books' item embeddings are identical regardless of what Books is paired with, and the expensive MiniLM encoding of ~6.9M items should never be redone per pair.
- **Pair bridge layer (parameterized by `(SOURCE, TARGET)`, run for every viable pair):** the §3.4 cross-vertical work — user-set intersection, matched source→target training pairs, per-user source profiles.

The project goal (per the rewritten §2.3) is cross-domain recommendation across **every viable vertical pair**, not a single locked pair. "Viable" = clears the ~10k shared-user floor after 5-core: **8 of the 10 pairs qualify**. The bridge therefore runs once per qualifying pair, all reusing the same cached per-vertical artifacts — no pair re-runs the expensive per-vertical layer. The two sub-floor pairs (Toys ↔ CDs 7.75k, CDs ↔ Video Games 4.4k) are excluded by default but recoverable via the **R1 contingency** (relax to 3-core).

---

## 0. Context carried in from Section 2

These are the §2 facts the whole of §3 must respect. Nothing in §3 should re-litigate them; it should act on them.

- **Experiment set:** **all 8 vertical pairs that clear the ~10k shared-user floor** after 5-core, run source→target with the larger vertical as source. Ranked by 5-core overlap: Books→Movies&TV (127k), Books→Toys&Games (87k), Movies&TV→Toys&Games (53k), Movies&TV→CDs&Vinyl (39k), Books→CDs&Vinyl (32k), Movies&TV→VideoGames (20k), Toys&Games→VideoGames (17k), Books→VideoGames (14k). The last three are borderline (just above floor) → lower-confidence. Excluded (sub-floor, R1 candidates): Toys&Games↔CDs&Vinyl (7.75k), CDs&Vinyl↔VideoGames (4.4k). All 5 verticals are processed at the per-vertical layer so every pair is available without reprocessing. Per-pair caveat: catalog richness varies — pairs targeting Movies & TV (title 58%, features 5%) lean on the source side + title fallback.
- **Filtering policy:** 5-core, applied **per vertical independently**, iterated to stability. 0-core was retained at ingest specifically so we can run (and ablate) the core filter ourselves.
- **Join key:** `parent_asin` (10-char parent-level ASIN), ratings ⨝ metadata, within a vertical. §2.2.8 found **zero orphan interactions** — every rated item has a metadata record — so no interactions are dropped at the within-vertical join.
- **Cross-vertical key:** `user_id` is the same person across verticals by construction; shared users = intersection of user columns. This set is the supervised training set for the mapping function.
- **Label encoding (from §2.1, confirmed by the §2.2 rating skew, 75–87% positive):** rating ≥4 → positive; rating = 3 → dropped (neutral); rating ≤2 → explicit negative; plus sampled negatives from non-interacted items. Same encoding for the single-vertical baseline and the cross-domain model.
- **Known data-quality actions owed to §3 (from §2.4):**
  1. Drop 4 Books rows with rating = 0 (out of range).
  2. Drop the `bought_together` field — 100% null across all 6.9M items.
  3. Text fallback for items with empty rich text: fall back to `title` + `store` + `details`; if still empty, assign a shared "no-content" placeholder embedding (do **not** drop the item).
  4. Normalize free-text categoricals (`categories`, `store`, `main_category`) — casing / near-duplicate variants.
  5. `main_category` is the live store label, **not** the partition; source of truth for an item's vertical is the file it came from.
- **Text encoder:** `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vector per item. Multi-line `description` / `features` joined with newlines before encoding.
- **Split:** time-based (chronological) train/validation/test on `timestamp` (ms since epoch); feasible because each vertical spans 1996–Sep 2023 with a dense recent decade.

---

## §3.1 Select Data

**CRISP-DM output:** *Rationale for Inclusion / Exclusion.*

This section narrows the collected 22.6 GB corpus down to the modeling input, restating §2.1 selection criteria in light of §2.2–§2.4 evidence. Selection is two-tiered: **which verticals enter the per-vertical layer** (all 5) and **which pairs the experiment includes** (the 8 that clear the ~10k floor).

### Steps

1. **Select verticals — keep all 5.** Books, Movies & TV, CDs & Vinyl, Video Games, Toys & Games all enter the per-vertical layer. Rationale: §2.3 found 8 of 10 pairs clear the overlap floor and all 8 are in the experiment set; processing all 5 once makes every pair available with no rework.
2. **Select tables.** Per vertical: the 0-core ratings CSV and the raw metadata JSONL. Reject (already at ingest, restated here): full review text, pre-built splits, the global item→vertical map.
3. **Select rows — apply 5-core.** Run iterative 5-core on each vertical's ratings table independently. This is a *selection* decision (which interactions enter modeling), so it lives here even though it executes as code in the Final Preprocess.
4. **Select columns.**
   - Ratings: `user_id`, `parent_asin`, `rating`, `timestamp` (all four).
   - Metadata kept: `parent_asin`, `title`, `description`, `features`, `categories`, `main_category`, `store`, `details`, `average_rating`, `rating_number`, and `author` (Books only).
   - Metadata excluded: `images`, `videos` (URLs only), `price` (sparse, irrelevant to taste transfer), `subtitle` (cosmetic), `bought_together` (100% null).
5. **Select the pair set.** Include the 8 pairs above the ~10k floor (listed in §0), each directed larger→smaller as source→target. Exclude the 2 sub-floor pairs, flagged as R1 candidates (relax to 3-core). Optionally add reverse directions as a symmetric ablation (16 directed experiments).
6. **Document the rationale.** For each inclusion/exclusion above, one line tying it to a §2 finding (volume, richness, join integrity, overlap floor, or relevance to the cross-domain goal). This *is* the §3.1 deliverable text.

**Inputs:** `data/raw/reviews/{vertical}.csv`, `data/raw/meta/meta_{vertical}.jsonl` for all 5 verticals.
**Outputs:** selection manifest (verticals, tables, columns, row-filter policy, the 8-pair experiment set) + the written rationale.

---

## §3.2 Clean Data

**CRISP-DM output:** *Data Cleaning Report* — decisions/actions addressing every problem the §2.4 Data Quality Report raised.

**Layer:** per-vertical — runs for all 5 verticals. Each §2.4 finding maps to exactly one action here.

### Steps

1. **Out-of-range ratings.** Drop the 4 Books rows with `rating = 0`. (Verify count post-load; report actual number cleaned.)
2. **Universally-null field.** Drop `bought_together` entirely (no signal).
3. **Missing rich text — fallback chain, not deletion.** For each item, build the text to encode as:
   `description + features` → if empty, fall back to `title + store + details` → if still empty, mark for the shared "no-content" placeholder embedding.
   Record per-vertical counts at each fallback tier (the §3.2 report needs them).
4. **Categorical normalization.** Lowercase + strip + collapse whitespace on `store` and on each `categories` path element; map known synonym/casing variants to a canonical form. Keep an original→canonical map for traceability.
5. **`main_category` correctness.** Do not trust `main_category` as the vertical label. Stamp each record with a `source_vertical` field taken from the file it came from; treat `main_category` as a descriptive feature only.
6. **Confirm clean axes (no action, just assert + log).** From §2.4: zero duplicate (user, item) pairs, zero null keys, all timestamps positive and within 1996–Sep 2023. Add assertions so the pipeline fails loudly if a future data refresh breaks these.

**Inputs:** selected tables from §3.1.
**Outputs:** cleaned ratings + cleaned metadata (per vertical), plus the cleaning log (counts per action) feeding the §3.2 report.

---

## §3.3 Construct Data

**CRISP-DM output:** *Derived Attributes* + *Generated Records*.

**Layer:** per-vertical — runs for all 5 verticals; embeddings cached by `parent_asin` so no pair re-encodes.

### Derived attributes

1. **Interaction label `y`** (the core derived attribute). From `rating`:
   `rating ≥ 4 → 1` (positive); `rating == 3 → drop`; `rating ≤ 2 → 0` (explicit negative).
2. **Item text embedding (384-dim).** Encode the §3.2 fallback text with `all-MiniLM-L6-v2`. Items flagged "no-content" receive the single shared placeholder vector. Cache embeddings keyed by `parent_asin` (reuse `data/embed_cache/`).
3. **Category feature.** Encode the normalized `categories` path as a categorical/multi-hot feature (and retain the raw path for the categories-only ablation).
4. **Popularity priors.** Pass through `average_rating`, `rating_number` as candidate item features (normalize — see below).
5. **Normalization (CRISP-DM §3.3 explicitly calls for this).** Scale `rating_number` (and any other heavy-tailed numeric) — log1p then standardize — so high-cardinality counts don't dominate. Embeddings are already unit-comparable; document whether L2-normalized.
6. **Datetime parse.** Derive a parsed datetime / year from `timestamp` (ms→datetime) for the split and for any temporal feature.

### Generated records

7. **Sampled negatives.** For each user, sample negatives from items they never interacted with (e.g. uniform or popularity-corrected, ratio configurable, default ~4:1 neg:pos). Combine with the explicit ≤2 negatives; expose an ablation flag to up-weight explicit negatives. These generated (user, item, y=0) rows are new records — document the sampling scheme, ratio, and seed.

**Inputs:** cleaned tables from §3.2.
**Outputs:** labeled interaction table with negatives; per-item feature matrix (embedding + category + popularity); embedding cache.

---

## §3.4 Integrate Data

**CRISP-DM output:** *Merged Data*.

**Layer:** split in two — the within-vertical join is **per-vertical (all 5)**; the cross-vertical bridge is **pair-specific, parameterized by `(SOURCE, TARGET)`** and run for each of the 8 viable pairs (looped) without touching the per-vertical artifacts.

### Steps

1. **Within-vertical join (per-vertical, all 5).** For each vertical, join cleaned ratings ⨝ item-feature table on `parent_asin` (left join from ratings; §2.2.8 guarantees zero orphans, but assert it). Produces per-vertical enriched interactions: (user, item, label, timestamp, item features). Done once for all 5.
2. **Build the cross-vertical user bridge (pair-specific, loop over the 8 pairs).** For each `(SOURCE, TARGET)`, intersect `user_id` sets of the two 5-core tables → the shared-user set. For each shared user, assemble the (source-side, target-side) training pair the EMCDR mapping is supervised on. Report the realized overlap per pair (sanity-check against the §2.3 numbers).
3. **Aggregate per-user source profile (pair-specific).** For each shared user, aggregate their SOURCE interactions into the source-user representation the mapping consumes (e.g. mean/weighted item embeddings, or the user-factor from the source recommender — note the dependency on the modeling choice; expose as a function so §4 can swap it).
4. **Assemble the three matched roles** (per pair): source-only users, target-only users, and overlap users — kept as explicit splits, since the mapping trains on overlap and is evaluated on transfer to target.

**Inputs:** §3.3 labeled tables + item features (all 5 verticals); the 8-pair experiment set.
**Outputs:** per-vertical enriched interaction tables (all 5); per-pair shared-user index + matched (source, target) training pairs, one folder per pair (e.g. `data/processed/pairs/Books__Movies_and_TV/`).

---

## §3.5 Format Data

**CRISP-DM output:** *Reformatted Data* — syntactic-only changes for the modeling tool.

**Layer:** the per-vertical split + ID maps are **per-vertical (all 5)** and feed the single-vertical baselines directly; the pair assembly reuses them for each of the 8 pairs.

### Steps

1. **Time-based split.** Sort interactions chronologically; cut per user (or global timestamp quantiles) into train / validation / test (e.g. 80/10/10 by time). Earlier → train, later → val/test. Persist split assignment as a column or separate index. Confirm both sides of the cut are well-populated (dense recent decade — §2.2.3). Done per vertical for all 5.
2. **ID remapping.** Map `user_id` and `parent_asin` string tokens → contiguous integer indices (required by embedding-table models), per vertical. Persist the lookup dictionaries both directions. The bridge layer keeps a global `user_id`→index map so a shared user has one consistent index across SOURCE and TARGET.
3. **Column order / schema.** Standardize the output schema: identifier columns first, label (`y`) last, features in a fixed documented order. Same schema for baseline and cross-domain inputs so the comparison is like-for-like.
4. **Storage format.** Write to an efficient columnar format (Parquet for tables, `.npy`/memmap or `.pt` for embedding matrices) under `data/processed/`. Record dtypes.
5. **Record ordering** if the tool needs it (e.g. sort train by user then timestamp for sequence batching).

**Inputs:** §3.4 merged data.
**Outputs:** model-ready train/val/test files + ID maps + feature arrays, schema documented in the Data Set Description.

---

## Final Preprocess — consolidated pipeline

The single end-to-end procedure to implement. Build as one orchestrated script (e.g. `data/preprocess.py`) with two stages — a **per-vertical loop over all 5 verticals** and a **pair-bridge step parameterized by `(SOURCE, TARGET)`**. Every step is cacheable/idempotent so a rerun skips finished work; the per-vertical artifacts are never recomputed when a new pair is bridged. All paths relative to project root; reads `data/raw/`, writes `data/processed/`.

**Config (top of script):**
```
VERTICALS = ["Books", "Movies_and_TV", "CDs_and_Vinyl",
             "Video_Games", "Toys_and_Games"]   # per-vertical layer: all 5
OVERLAP_FLOOR = 10_000           # pair is viable if 5-core shared users >= floor
# 8 viable pairs (larger -> smaller as source -> target), ranked by 5-core overlap:
PAIRS = [("Books","Movies_and_TV"), ("Books","Toys_and_Games"),
         ("Movies_and_TV","Toys_and_Games"), ("Movies_and_TV","CDs_and_Vinyl"),
         ("Books","CDs_and_Vinyl"), ("Movies_and_TV","Video_Games"),
         ("Toys_and_Games","Video_Games"), ("Books","Video_Games")]
# excluded sub-floor (R1 candidates via 3-core): (Toys,CDs)=7.75k, (CDs,VideoGames)=4.4k
ADD_REVERSE = False              # set True for symmetric sparse->dense ablation (16 pairs)
KCORE = 5
POS_THRESHOLD = 4          # rating >= 4 -> positive
NEUTRAL = 3                # dropped
NEG_SAMPLE_RATIO = 4       # sampled neg : pos
EXPLICIT_NEG_WEIGHT = 1.0  # ablation knob
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
SPLIT = (0.8, 0.1, 0.1)    # time-based, chronological
SEED = 42
```

### Stage A — per-vertical layer (loop over all 5 in `VERTICALS`)

For each vertical, run and cache:

1. **Load & select** (§3.1) — read ratings CSV (pyarrow, typed) and stream the metadata JSONL; keep only the selected columns.
2. **Clean** (§3.2) — drop out-of-range ratings (the 4 `rating=0` rows in Books); drop `bought_together`; assert no dup keys / no null keys / valid timestamps; normalize `store` + `categories` casing; stamp `source_vertical` from the filename.
3. **5-core filter** (§3.1/§3.2) — iterate to stability; log surviving interactions/users/items.
4. **Construct labels** (§3.3) — map rating → `y` (≥4=1, ≤2=0, drop 3).
5. **Build item features** (§3.3) — fallback text chain → encode with MiniLM (cache by `parent_asin` in `data/embed_cache/`); placeholder vector for no-content items; category multi-hot; log1p+standardize `rating_number`.
6. **Sample negatives** (§3.3) — per-user negatives from non-interacted items at `NEG_SAMPLE_RATIO`; merge with explicit ≤2 negatives; seeded.
7. **Within-vertical join** (§3.4) — ratings ⨝ item features on `parent_asin` (assert zero orphans).
8. **Format** (§3.5) — time-based chronological split; remap user/item IDs to contiguous ints (persist maps); fix schema (ids first, `y` last); write Parquet + embedding arrays to `data/processed/{vertical}/`.

Output of Stage A: a self-contained, model-ready dataset per vertical — these are also the single-vertical baseline inputs.

### Stage B — pair bridge (loop over the 8 viable pairs in `PAIRS`)

Consumes Stage A artifacts; never recomputes them. Re-runnable for any pair; sub-floor pairs only enter via the R1 3-core contingency.

For each `(SOURCE, TARGET)` in `PAIRS`:

9. **Intersect users** (§3.4) — shared-user set of the two 5-core tables; report overlap and assert ≥ `OVERLAP_FLOOR` (sanity-check against §2.3 numbers).
10. **Source profiles + matched pairs** (§3.4) — aggregate each shared user's SOURCE history into the source representation; assemble (source, target) training pairs and the source-only / target-only / overlap role splits; keep a consistent global user index across the two verticals.
11. **Write pair artifacts** — to `data/processed/pairs/{SOURCE}__{TARGET}/`.

### Stage C — describe

12. **Emit Data Set Description** (§3 phase output) — write `data/processed/DATASET_DESCRIPTION.md`: per-vertical row/user/item counts per split, per-pair overlap size, feature schema, label balance, and the seed/config used — one source of truth for the §3 write-up and §4 modeling.

**Guardrails:** every drop/fallback/sample step logs a count; assertions enforce the §2.4 clean axes; the whole run is reproducible from `SEED` + config; Stage B for a new pair must run without re-executing Stage A.

**Deliverables this produces for the HW2 template §3:**
- §3.1 rationale text (selection manifest).
- §3.2 cleaning report (per-action counts).
- §3.3 derived-attributes + generated-records description (labels, embeddings, negatives).
- §3.4 merged-data description (join + bridge + overlap count).
- §3.5 reformatted-data description (split, ID maps, schema, storage).
- Phase output: the model-ready dataset + `DATASET_DESCRIPTION.md`.
