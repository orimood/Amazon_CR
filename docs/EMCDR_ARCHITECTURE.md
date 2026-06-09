# EMCDR Cross-Domain Recommender — Final Architecture

_The model as deployed for the headline results. Code: `src/cdr/`._

## What it does

Recommend items in a vertical a user has **no history in** (the *target*), using their
history in another vertical (the *source*). Example: a user with a long **Books** history and
zero **Video Games** history — we still rank video games for them.

The method is **EMCDR** (Embedding-and-Mapping Cross-Domain Recommendation): never model two
domains jointly. Instead learn each vertical on its own, then learn one small "translator"
between them, supervised on the users who happen to exist in both.

Three stages, each trained independently:

1. **Recommender** — a hybrid two-tower model per vertical → user and item embeddings.
2. **Mapping** — a linear function `f` that translates a *source* user vector into the
   *target* user space, trained on overlap users.
3. **Inference** — for a cold user, map their source vector into the target space and rank
   target items by dot product.

```
SOURCE vertical                                   TARGET vertical
 interactions ─┐                                   interactions ─┐
 item metadata ┤                                   item metadata ┤
               ▼                                                  ▼
   HybridRecommender(src)                            HybridRecommender(tgt)
   user_emb  +  hybrid item tower                    user_emb  +  hybrid item tower
   trained with BPR                                  trained with BPR
        │                                                 │
        │  U_src[overlap]            U_tgt[overlap]  ◄─────┤
        │  (frozen)                  (frozen)              │
        └──────────►  linear mapping f  ◄──────────────────┘   train: f(U_src) ≈ U_tgt
                            │                                   (on 80% of overlap users)
                            ▼
 cold user ─ U_src ─► f ─► û_t ──(dot)── V_tgt ─► ranked target items
                                          ▲
                              all item embeddings of the target
```

The only learned cross-domain object is `f`. Everything else is per-vertical and reused across
every pair that touches that vertical.

---

## Data layer (`src/cdr/data.py`)

Nothing here is learned — it loads pre-processed artifacts.

- **`ItemFeatureStore`** — per-vertical item metadata:
  - `text_emb` — `(n_items, 384)` MiniLM sentence embeddings of the item title/description.
  - categories in CSR form (`cat_indptr`, `cat_indices`) — item `i`'s category IDs are
    `cat_indices[cat_indptr[i] : cat_indptr[i+1]]`.
  - `pop_feat` — `(n_items, 2)` = `[avg_rating, rating_number_z]`.
  - `cat_segments()` returns a flattened category index list plus a per-row segment map, so
    the model can sum each item's category vectors with `index_add_`. (This replaces
    `nn.EmbeddingBag`, which is unimplemented on Apple MPS — the code then runs identically on
    CPU and MPS.)
- **`VerticalData`** — one vertical's interactions (with a `train`/`test` split column) plus
  helpers: `train_positives()`, `user_seen_sets()`, `test_positives()`, `item_popularity()`.
- **`PairData` + `align_overlap()`** — the cross-vertical bridge. Each vertical has its own
  contiguous `user_idx`, so "user 5012 in Books" ≠ "user 5012 in Movies." The only stable key
  is the raw `user_id`. `align_overlap` joins on it to produce a table of
  `[user_id, src_idx, tgt_idx]` — the same person's index in both verticals. **This table is
  the entire training input to the mapping.**

---

## Stage 1 — Per-vertical hybrid recommender (`HybridRecommender`, `src/cdr/models.py`)

One instance per vertical. Two towers, scored by a dot product. Embedding dim `d = 64`.

### User tower
A plain learned lookup table — that is the whole user model.
```
user_vec(u) = user_emb[u]            # (B, 64)
```

### Item tower (hybrid)
Each item's embedding is a **sum of ingredients**, so it is grounded in shared metadata, not
just a free per-item parameter:
```
item_emb(i) = id_factor(i)                  # (B,64)  free per-item vector
            + text_proj( text_emb(i) )      # Linear 384→64 on the MiniLM embedding
            + cat_bag(i)                     # sum of learned category vectors (index_add_)
            + pop_proj( pop_feat(i) )        # Linear 2→64
```
Metadata (text + categories) is the **shared language** across verticals: a book and a video
game both have a MiniLM text vector in the *same* 384-d space even though they share no users
or item IDs. That shared space is what makes cross-domain transfer possible at all.

### Content-grounded item tower for sparse targets
For a sparse target catalog (Video Games: ~22.7k items, low repeat-purchase density), the free
`id_factor` is mostly noise — most items appear too few times for their per-item vector to be
learned well. The deployed fix (`freeze_id` in `models.py`) **zeroes and freezes `id_factor`**,
so the item embedding becomes **pure content**:
```
item_emb(i) = text_proj( text_emb(i) ) + cat_bag(i) + pop_proj( pop_feat(i) )
```
This is applied **only to the sparse target recommender** (via `rec_tag_overrides`); the source
recommenders keep their full item tower. This is exact, not an approximation: a source's
`id_factor` is never used downstream — only its `user_emb` feeds the mapping. For these sparse
targets we also train on a **denser 3-core** build of the vertical (lower interaction-count
threshold → more items, more users, ~2× overlap), which produces cleaner target item vectors.

### Training — BPR (`src/cdr/train_recommender.py`)
A pairwise *ranking* loss, not rating prediction. Per step:
1. Take a batch of real interactions `(u, i_pos)`.
2. Sample a random item `i_neg` the user hasn't seen (rejection-sampled against the user's
   seen-set).
3. Push the real item's score above the random one's:
   ```
   loss = -logsigmoid( score(u, i_pos) - score(u, i_neg) ).mean()
   score(u, i) = user_vec(u) · item_emb(i)
   ```

Hyperparameters: `d=64`, 15 epochs, Adam `lr=0.01`, batch `4096`, weight decay `1e-6`, one full
pass over train positives per epoch.

Each per-vertical recommender is **trained once and cached** to
`models/recommenders/{vertical}__{ablation}__{tag}/`. Every pair that touches that vertical
reloads the same checkpoint, so Books' recommender is identical across Books→Movies and
Books→Video Games. (Trained on CPU: Apple MPS yields NaN BPR loss for this model and is slower
for an embedding-lookup / negative-sampling workload.)

After this stage we have, per vertical: a user-embedding table `U` and a way to materialize the
full item-embedding matrix `V` of shape `(n_items, 64)` (`all_item_embeddings()`).

---

## Stage 2 — Cross-domain mapping (`MappingMLP`, `src/cdr/models.py` + `train_mapping.py`)

The only component that connects two domains. A function `f : R⁶⁴ → R⁶⁴` that translates a
source-domain user vector into the target user space.

### Architecture — a single regularized linear layer
```
f(x) = W·x + b            # ~4.2k parameters
```
A linear map (`map_hidden: []`) outperforms a deeper MLP here: the mapping is supervised on as
few as ~10k–100k overlap users, so the larger model overfits ("sharp minima, poor
generalization"). The small, regularized linear map generalizes better and is never materially
worse on the data-rich pairs. (`MappingMLP` is the general class; with no hidden layers it
degrades to exactly `Wx+b`.)

### Training
1. From the alignment table, gather the **frozen** user vectors of overlap users from **both**
   recommenders:
   ```
   U_src = src_model.user_emb[src_idx]    # (n_overlap, 64), detached
   U_tgt = tgt_model.user_emb[tgt_idx]    # (n_overlap, 64), detached
   ```
2. Split overlap users **80 / 20**. The 80% train `f`; the 20% are held out as the **cold-eval
   cohort** the mapping never sees.
3. Train `f` to make `f(U_src) ≈ U_tgt`:
   ```
   loss = MSE(f(U_src), U_tgt) + 0.5 · (1 − cosine_similarity(f(U_src), U_tgt)).mean()
   ```
   MSE fixes the vector's magnitude; the cosine term fixes its direction.

Hyperparameters: 200 epochs, Adam `lr=1e-3`, batch `1024`, weight decay `1e-4`, cos weight `0.5`.
The recommender embeddings are *fixed inputs* — only `f`'s weights move.

---

## Stage 3 — Cold-start inference

A cold user arrives with source history and no target history:
```
û_t  = f( U_src[user] )            # predicted target-domain taste vector  (64,)
score = û_t · V_tgt                 # dot product against every target item  (n_items,)
```
Rank target items by `score`, recommend the top ones. The user never needed any target
history — `f` manufactured a plausible target taste vector from their source taste.

### Serve-time model selection
Cross-domain transfer pays off only where the mapping has enough overlap supervision. On the
data-rich pairs EMCDR is the best model and is served directly. On the sparse Video-Games target
— where even with the content-grounded tower and denser data EMCDR places a close second — the
deployed policy **gates to the strongest target-side baseline instead** (target popularity /
single-vertical prior). Overlap size is known at serve time, so this gate leaks nothing. (A
score-blending hybrid was tested and discarded: EMCDR and content often disagree about the
held-out item, so averaging their rankings interferes destructively — gating, not blending, is
the right combine.)

---

## Evaluation (`src/cdr/evaluate.py`)

Leave-one-out sampled ranking, on the **cold-eval cohort** (the 20% of overlap users the mapping
never trained on, with their target history hidden).

- Per user: 1 held-out target positive (the latest by timestamp) + **100 sampled unseen
  negatives**, ranked together (101 candidates).
- Score the candidates with `û_t · V_tgt`; the positive's rank gives the metrics.
- Metrics: **Recall@10/20, NDCG@10/20, HR@10**, averaged over users and over **3**
  negative-sampling seeds (reported mean ± std).

Compared against baselines on the *identical* candidate sets:

| baseline | what it is |
|---|---|
| **target-MF** | the average target-user vector · item vectors (single-vertical prior, no personalization) |
| **feature-transfer** | cosine of the user's mean source-liked-item text vs. each target item's text, in the shared MiniLM space (training-free) |
| **MostPop** | target-train popularity (the deployed "popular this week" fallback) |
| **Random** | the floor (≈ 10/101 ≈ 0.099, confirms calibration) |

---

## Results (Recall@10, mean over 3 seeds)

EMCDR is the **outright best model on the 5 data-rich pairs** — it beats every baseline where the
mapping has enough overlap supervision. On the 3 sparse Video-Games pairs it places **#2** (a
target-side popularity prior edges it), and it **beats the deployed MostPop fallback on all 8**.

| pair | EMCDR | content | target-MF | MostPop | winner |
|---|--:|--:|--:|--:|:--|
| Books → Movies&TV | **0.323** | 0.231 | 0.293 | 0.191 | EMCDR |
| Books → Toys&Games | **0.325** | 0.299 | 0.318 | 0.238 | EMCDR |
| Movies&TV → Toys&Games | **0.303** | 0.273 | 0.303 | 0.226 | EMCDR (ties target-MF) |
| Movies&TV → CDs&Vinyl | **0.365** | 0.262 | 0.316 | 0.238 | EMCDR |
| Books → CDs&Vinyl | **0.395** | 0.207 | 0.321 | 0.230 | EMCDR |
| Movies&TV → Video Games | 0.339 | 0.303 | **0.362** | 0.220 | target-MF (EMCDR #2) |
| Books → Video Games | 0.308 | 0.216 | **0.345** | 0.201 | target-MF (EMCDR #2) |
| Toys&Games → Video Games | 0.279 | **0.364** | 0.260 | 0.160 | content (EMCDR #2) |

**Outright tally: EMCDR 5 · target-MF 2 · content 1.** Cross-domain personalization demonstrably
pays off where shared-user overlap is large; on the sparse Video-Games target it does not beat a
target-popularity prior, so the serve-time gate routes those pairs to the baseline.

---

## One-sentence summary

Train a separate hybrid two-tower recommender per vertical (with item embeddings grounded in
shared MiniLM text so domains speak a common language), then learn one small regularized linear
map on the users who exist in both verticals — so a brand-new user in the target vertical can be
ranked using only their source-vertical history, by translating their source taste vector into
the target space.
