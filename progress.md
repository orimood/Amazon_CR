# Progress

Working notes for the Cross-Domain Recommendation (CDR) project. Written so someone joining cold can pick up the work without reading the back-and-forth.

## What the project is

Build an EMCDR-style cross-domain recommender on Amazon Reviews 2023 and show it beats a single-vertical baseline (Matrix Factorization / Neural Collaborative Filtering) on a chosen source→target vertical pair. One deliverable model, six-month engagement, four-person team. Full motivation, methods, success criteria, and risk register are in `submitted/HW1_CDR (2).docx` (CRISP-DM Business Understanding) and `submitted/cross_domain_recsys_project (1).docx` (project brief).

The cross-vertical bridge is item-side: hierarchical categories + a text embedding of title and description through `sentence-transformers/all-MiniLM-L6-v2` (384-dim). User IDs are shared across verticals by construction in the McAuley release, which is what makes EMCDR-style mapping training possible.

## Repository layout

```
Amazon_CR/
├── CRISP-DM User Guide.pdf       # methodology reference (1999 paper)
├── HW2 - Template.docx           # current homework — filling in §2.x and §3.x
├── HW2 - Template.bak.docx       # backup before automated edits
├── submitted/                    # previous-phase deliverables
│   ├── HW1_CDR (2).docx
│   └── cross_domain_recsys_project (1).docx
├── data/
│   ├── download.py               # reproducible HuggingFace pull
│   └── raw/                      # gitignored — not committed
│       ├── reviews/   *.csv      # 0-core ratings per vertical
│       └── meta/      *.jsonl    # raw item metadata per vertical
├── progress.md                   # this file
└── .gitignore
```

## What's been done

**Data acquisition (done 2026-05-26).**

Pulled the Amazon Reviews 2023 corpus from `McAuley-Lab/Amazon-Reviews-2023` for five verticals: **Books, Movies_and_TV, CDs_and_Vinyl, Video_Games, Toys_and_Games**. Two files per vertical:

- `benchmark/0core/rating_only/<Cat>.csv` — `user_id, parent_asin, rating, timestamp`. We took the **unfiltered (0-core) variant on purpose** — the project plan runs the 5-core filter ourselves on a pinned snapshot, and the pre-cored variants would lock in a filtering policy we want to be able to vary in ablations.
- `raw/meta_categories/meta_<Cat>.jsonl` — full item metadata.

Total: 10 files, ~22.6 GB. Download script at `data/download.py`. Items per vertical (JSONL line counts on the meta files):

| vertical          | items     | reviews size | meta size |
|-------------------|----------:|-------------:|----------:|
| Books             | 4,448,181 | 1.69 GB      | 14.70 GB  |
| Toys_and_Games    |   890,874 | 0.93 GB      |  2.64 GB  |
| Movies_and_TV     |   748,224 | 1.00 GB      |  1.29 GB  |
| CDs_and_Vinyl     |   701,959 | 0.28 GB      |  0.95 GB  |
| Video_Games       |   137,269 | 0.26 GB      |  0.44 GB  |

**Why these five.** Entertainment-media cluster — the cluster where the "taste transfers via shared categories and descriptions" hypothesis from HW1 §1.1 is most defensible. Books is the source/hub (richest text, deepest category tree). Movies_and_TV is the canonical CDR partner for Books in the literature. CDs_and_Vinyl, Video_Games, Toys_and_Games are the three candidate targets to evaluate in §2.3. The final source→target pair is locked there, not now.

**Why not Kindle_Store** (which was the obvious fifth candidate): it is functionally a re-format of Books and would conflate same-domain transfer with cross-domain transfer.

**Files explicitly NOT downloaded** (and why):

- `raw/review_categories/<Cat>.jsonl` (full review bodies, tens of GB per vertical) — the cross-vertical bridge uses **item-side** text (title + description), not the user-written review body.
- `5core_*` and `5core_last_out_w_his_*` pre-built splits — we define our own seeded splits plus a time-based split.
- `asin2category.json` (1.25 GB global ASIN → top-level-vertical map) — redundant with per-vertical meta files we already have.

## Schema findings

Metadata is JSON-Lines, one item per line. Schema is consistent across all 5 verticals — same field names, different population rates.

**Core fields (every vertical).** `parent_asin` (10-char ASIN, the join key to reviews), `title`, `categories` (hierarchical list), `description` (list of paragraphs), `features` (bullet lines), `details` (dict of structured attributes), `main_category`, `store`, `average_rating`, `rating_number`, `price`, `images`, `videos`, `bought_together`.

**Vertical-specific extras.** Books has `author` (dict) and `subtitle`. Movies_and_TV has `subtitle` plus Directors / Audio languages / Subtitles in `details`. CDs_and_Vinyl has Original Release Date / Run time in `details`. Video_Games has Rated / Manufacturer in `details`. Toys_and_Games has Package Dimensions / Material in `details`.

**Three caveats that bit us already or will bite later:**

1. **`bought_together` is universally null.** 100% null across all 6,926,507 items in our 5 verticals. The field exists in the schema but carries zero signal. Drop at ingest.
2. **`main_category` is the live store category, not the dataset partition.** Movies_and_TV records can read `"Prime Video"`. CDs_and_Vinyl records can read `"Digital Music"`. The source of truth for "which vertical is this item in" is **the file the record came from**, not this field.
3. **Long-tail items have empty rich-text fields.** First Books and Toys records both had `description == []`, `categories == []`. Feature pipeline must fall back to `title + store + details` rather than dropping the item, to preserve catalog coverage.

## Field importance ranking

For the EMCDR pipeline:

| tier | fields | role |
|---|---|---|
| Critical | `user_id`, `parent_asin`, `rating`, `timestamp` | identity + interaction signal; rating ≥4 → positive |
| High | `title`, `description`, `features`, `categories`, `main_category` | cross-vertical bridge inputs (text encoder + taxonomy) |
| Medium | `details`, `store`, `author` (Books only) | structured side features, ablation candidates |
| Low | `average_rating`, `rating_number` | popularity priors |
| Unused | `images`, `videos`, `price`, `subtitle`, `bought_together` | dropped at ingest |

## HW2 status

HW2 is the **Data Understanding + Data Preparation** phase write-up. Template at `HW2 - Template.docx`, methodology reference at `CRISP-DM User Guide.pdf`.

- §2.1 Collect Initial Data — **drafted, under revision.** Currently covers source, acquisition, data requirements planning (Interactions / Item metadata / Shared users), selection criteria (vertical pick + file-variant pick + field tier), insertion/extraction notes (formats, encoding plan, missing values, `bought_together`-empty, join key, `main_category` caveat), merge-quality cautions, and an inventory of the 10-file snapshot.
- §2.2 Describe Data — **TODO.** Gross properties: row counts on the ratings tables, attribute types and value ranges, free-text presence rates, basic distributions (ratings histogram, items per user, users per item).
- §2.3 Explore Data — **TODO.** This is where the source→target vertical pair gets locked. Run 5-core filter, compute pairwise shared-user overlap across all 10 vertical pairs, pick the pair with strongest overlap × strongest catalog text richness. Also: per-vertical sparsity, category-tree depth distributions, timestamp coverage for the time-based split.
- §2.4 Verify Data Quality — **TODO.** Orphan interactions (interactions whose ASIN has no metadata record), missing-value rates per field per vertical, duplicate detection, timestamp sanity (range, monotonicity, outliers), rating value sanity.
- §3.x Data Preparation — **TODO.** Selection (which subset enters modeling after §2.4), cleaning (handle orphans + missing fields), construction (derived attributes — user profile aggregation, text embeddings), integration (cross-vertical user join), formatting (final shape for the training loop).

**Style convention for the report.** Short labeled paragraphs in `Term: prose.` form. Bold lead-ins. No filesystem paths. Reader is assumed not to have access to the working tree. Reference cross-section labels (`§1.3`, `§S2`) rather than file names. The opening "Source and acquisition" subsection of §2.1 is the canonical style example.

## Open decisions / next checkpoints

1. **Source→target vertical pair.** Pending §2.3. Default working hypothesis is Books → Movies_and_TV (the literature-standard pair), but the actual pick depends on whichever pair clears ~10k shared users after 5-core filtering and has the richest catalog metadata. R1 in the HW1 risk register.
2. **5-core threshold.** Project plan commits to 5-core. If the chosen pair falls short of ~10k shared users at 5-core, contingency is relaxing to 3-core on the secondary vertical (HW1 §1.4 R1 contingency). Decision deferred until overlap numbers are in §2.3.
3. **Style consistency in §2.1.** The user-edited "Source: / Acquisition:" lines have no bold lead-in. The other labeled paragraphs ("Interactions: / Item metadata: / Shared users:") do. Pick one and apply consistently before submission.

## Reproducing the data pull

```
cd Amazon_CR
python3 data/download.py
```

Requires `huggingface_hub` (tested with 0.36.0). Pulls ~22.6 GB into `data/raw/`. Idempotent — re-running skips files already on disk at the expected size.

## §3 Data Preparation — done (2026-06-01)

The §3 spec was rewritten as `section3_preprocessing_plan.md` (replacing the old `data_preparation_spec.md`). The key shift: the pipeline is now two layers so a new pair never re-runs the expensive per-vertical work.

- **Stage A — per-vertical (all 5 verticals).** Load → 5-core → label → seen sets → sampled negatives (4:1) → MiniLM text embeddings (shared SHA1 cache) → category multi-hot (shared 3,092-token vocab) → popularity priors (log1p z) → within-vertical join → temporal split → ID-remap → write `data/processed/{vertical}/`.
- **Stage B — pair bridge (8 pairs above the 10k shared-user floor).** Intersect users, assemble matched (source, target) training pairs and the role splits, write `data/processed/pairs/{SRC}__{TGT}/` with a pair-global `user_idx` so a shared user has one index on both sides. Reads Stage A artifacts only; never touches `data/raw/`. Sub-floor pairs (Toys ↔ CDs ≈ 7.75k, CDs ↔ Video Games ≈ 4.4k) excluded; flagged as R1 candidates (3-core).
- **Stage C — describe.** Emits `data/processed/DATASET_DESCRIPTION.md` + the JSON mirror — the consolidated §3 deliverable.

Per-vertical 5-core counts and per-pair overlaps live in `DATASET_DESCRIPTION.md`. Regression check vs. the prior single-pair run: Books → Movies_and_TV shared users = 109,206 exactly matches the old pipeline.

### Embeddings

The notebook's default is **smoke mode** (`embed_sample_size = 2000`) so a plain Run All finishes in ~20 min and exercises the full Stage A + B + C. The dedicated **full pass** is in `data/embed_all.py` — reads each vertical's `item_features.npz`, recovers item text from the meta JSONLs only for items currently holding the zero placeholder, encodes against the shared cache, and writes the npz back atomically. After running it the description is regenerated with `data/regen_description.py`. Post-full-pass coverage:

| vertical        | items embedded | placeholder | total   |
|-----------------|---------------:|------------:|--------:|
| Books           |        440,134 |           0 | 440,134 |
| Movies_and_TV   |        181,528 |           4 | 181,532 |
| CDs_and_Vinyl   |         80,990 |           0 |  80,990 |
| Video_Games    |         22,746 |           0 |  22,746 |
| Toys_and_Games  |        148,572 |           0 | 148,572 |

The four Movies_and_TV placeholders are the `text_source == 'none'` items the §3.2 fallback policy kept on purpose. The shared SHA1 cache holds 843,669 vectors (~1.5 GB) and is gitignored.

### iCloud Drive gotcha

The project lives on `~/Desktop`, which iCloud Drive syncs. Two failure modes hit during the full embedding pass and are worth knowing about:

1. **Direct `np.savez()` to a multi-hundred-MB file on the iCloud path can return `TimeoutError [Errno 60]` mid-write.** `data/embed_all.py` works around this by writing the new npz to `/tmp` first and `os.replace`-ing into place — atomic rename from the script's POV, single inode swap from iCloud's.
2. **Force-reinstalling pip packages while iCloud is active can leave thousands of duplicate `* 2.py` / `__init__ 2.py` files in `.venv/lib/python3.12/site-packages/`.** Python imports that scan those files block on the syscall and the interpreter appears to hang at 0% CPU during `import sentence_transformers`. Fix is to delete every `* 2*` entry under site-packages and (if needed) reinstall whatever real submodule got displaced.

The proper long-term fix is to move the project off the Desktop path. Until then, these two workarounds are stable.

## HW2 status — §3 sections to fill

`HW2 - Template.docx` §3.1–§3.5 can be written directly from `data/processed/DATASET_DESCRIPTION.md`: each table there maps to one section of the template.

## §4 Modeling — pipeline built + smoke-validated (2026-06-01)

Branch `emcdr-modeling`. Code in `src/cdr/` (run via `PYTHONPATH=src python -m cdr.run_pair --config configs/smoke.yaml`). Full architecture + how-to-run in `MODEL_CARD.md`; deciding rationale in the approved plan.

**Deliverable scope is all 8 pairs**, not one — the HW1 "proof-of-value pair" wording is superseded by the §3 rewrite. The pipeline mirrors the Stage A/B reuse layout: 5 per-vertical hybrid recommenders trained once each + 8 per-pair mapping MLPs reusing them, aligned across verticals by `user_id` through the id-maps.

- **Model.** Hybrid two-tower per vertical: learned user table + item tower `id_factor + W_text·text_emb + cat_bag(categories) [+ pop]`, BPR loss. EMCDR mapping `f: U_src→U_tgt` (MLP) trained on overlap users; cold-start inference `û_t=f(U_src)·V_tgt`. Ablations toggle the content branches (`full`/`cats_only`/`text_only`/`id_only`).
- **Eval.** Leave-one-out, 1 held-out target test positive + 100 sampled unseen negatives; Recall@10/20, NDCG@10/20, HR@10, mean±std over seeds. Baselines: random, mostpop (deployed fallback), target-MF + mean-user-vector, feature-transfer (Model B, content cosine in shared MiniLM space).
- **MPS gotcha.** `nn.EmbeddingBag` is unimplemented on Apple MPS — the category bag uses a learned table + `index_add_` instead, so it runs on CPU and MPS alike.
- **Smoke result (Toys→Video_Games, d=32, 2 epochs, CPU; NOT headline numbers).** Pipeline trains end-to-end, deterministic on re-run, all 5 models produce finite metrics. Content-bearing EMCDR beats the chance baselines; `id_only` falls below chance (content is what enables transfer); `feature_transfer` is strong and ablation-invariant. Results in `results/*.json` (tracked; `models/` weights gitignored).

**Deferred:** full training on `configs/full.yaml` (all 8 pairs × 3 ablations, MPS; Books heaviest), the full lift-over-baseline grid, random-split robustness, and the N=50 qualitative check.

## §4 Modeling — mapping redesign: history-grounded bridge (2026-06-02)

Full-run diagnosis: the EMCDR mapping is strong at high overlap (won 4/8 pairs) but **loses to a no-learning content baseline at thin overlap** (the ~11–16k-overlap Video Games targets). Root cause worked out with the user: the mapping is *data-starved and trained poorly*, not too weak. Decision: **do not** bolt on the content crutch (Model B fusion); instead make the *learned* mapping itself better. Approved plan: `~/.claude/plans/sprightly-tinkering-dusk.md`.

New module `src/cdr/bridge.py` (config knobs in `config.py`; `configs/bridge_{smoke,full}.yaml`):
- **History-grounded source encoder** — source-user rep = learned pool (mean | attention) over the embeddings of the user's source items, so it's a function of history (enables augmentation, robust for thin/cold users, no fragile free `U_src`).
- **End-to-end BPR-on-target** — the bridge is trained to rank the user's *real* target interactions, NOT by regressing onto the noisy learned `U_tgt`.
- **Sub-history augmentation** — each overlap user → many examples per epoch (resampled source subsets).
- **Semi-supervised CORAL alignment** — source-only/target-only users pull the bridge output distribution toward the real target-user distribution.
- **Validation early-stopping + best-on-val model selection**, per-user temporal hold-out (also fixes the global-split untrained-user issue). Reuses the recommenders only for frozen item embeddings + the existing eval helpers.

**Status: written, NOT yet execution-tested.** PyTorch (and most of `.venv`) was iCloud-evicted mid-session (`libtorch_cpu.dylib` showed 0 on-disk blocks), so any python importing torch/yaml hangs on re-download. Verified what's possible without it: `py_compile` clean (no syntax errors), config keys cross-checked against the dataclass, full static logic review. **To validate once `.venv` is restored:** `PYTHONPATH=src python -m cdr.bridge --config configs/bridge_smoke.yaml` (tiny, CPU), then `configs/bridge_full.yaml` for the Phase-5 confirmation (thin Books→Video_Games must reach ≥ the content baseline; rich Books→Movies must not regress).
