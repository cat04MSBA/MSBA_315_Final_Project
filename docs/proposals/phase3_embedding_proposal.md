# Phase 3 Embedding Feature Proposal (v1)

**Group:** Embedding (5 of 5 in the Phase 3b incremental ablation)
**Status:** Awaiting planning-conversation review
**Date:** 2026-05-03

This is the v1 draft of the embedding-feature proposal. It is
structured to match the lexical, sentiment, topic, and character-
network proposals so the planning conversation can review it under
the same template. Polish guidance returned by the planning
conversation will be folded into a v2 before implementation begins,
matching the pattern used for the previous groups.

---

## 1. Why embeddings last

The embedding group sits last in the Phase 3b ablation queue for
three reasons.

**It is the most expensive group.** Sentence-transformer embedding
extraction over 1,713 films, each with a median of 880 dialogue
lines, produces roughly 1.4 million sentence embeddings. With a
small encoder (`all-MiniLM-L6-v2`, 384-dim, ~1,000 sentences/sec on
a modern CPU) the full corpus takes 25 to 45 minutes; with a larger
encoder (`all-mpnet-base-v2`, 768-dim) the time roughly doubles.
Saving the most expensive group for last lets us learn from the
previous four groups before paying the compute cost.

**It is the group with the strongest published prior.** Gross
(2025) used MovieSum + sentence-transformer embeddings to predict
Oscar nominations and reported encoded-dialogue embeddings as the
strongest single predictor. We are extracting from the same corpus
with similar machinery; if any feature group is going to lift, this
is the one.

**It overlaps conceptually with the topic group.** Both groups
extract a low-dimensional representation of dialogue content;
topic via LDA, embedding via a pre-trained transformer. Saving
embeddings for last means the topic group's standalone-lift
result is in hand before we evaluate the embedding group, which
informs the v2 polish (especially around dimensionality reduction).

---

## 2. The substantive design questions: model, pooling, dimensionality

Embedding feature design has three substantive choices that the
proposal must commit to before implementation: the encoder model,
the pooling strategy, and the dimensionality-reduction strategy.

### 2.1 Encoder model: `all-MiniLM-L6-v2`

**Decision (v1): `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
22M parameters, MIT license).**

Three encoder choices were considered:

* **`all-MiniLM-L6-v2`** — 384-dim, 22M parameters, ~1,000
  sentences/sec on CPU. Strong general-purpose sentence-similarity
  performance on the MTEB benchmark. The lightest of the three.
* **`all-mpnet-base-v2`** — 768-dim, 110M parameters, ~400
  sentences/sec on CPU. Better performance on harder semantic-
  similarity tasks; doubles the compute cost.
* **A dialogue-tuned encoder** (e.g., `sentence-transformers/
  paraphrase-MiniLM-L6-v2`) — same parameter count as
  `all-MiniLM-L6-v2`; trained on paraphrase pairs that include
  conversational and dialogue text. May be marginally better on
  this corpus.

The v1 commits to `all-MiniLM-L6-v2`. Three reasons. First,
compute: the lightest option lets us iterate on pooling and
dimensionality without re-running 30-minute extractions. Second,
the MTEB-benchmark performance gap between the three is small
(roughly 1-2 percentage points on most tasks); the corpus-specific
gain from mpnet may not justify the cost. Third, MiniLM is the
most-cited encoder in recent screenplay-prediction work, including
Gross (2025), so any positive standalone lift is interpretable
against a clear prior.

If the standalone-lift result is null, the proposal v2 will
revisit the encoder choice rather than re-running with mpnet
mechanically; the encoder is unlikely to be the bottleneck in a
null result whose pattern matches the previous Phase 3b groups.

### 2.2 Pooling strategy: per-line embedding, mean-pooled to film

**Decision (v1): per-dialogue-line embedding, simple mean-pool
across lines, one 384-dimensional vector per film.**

Three pooling choices were considered:

* **Per-line mean-pool** — embed every dialogue line, average
  across lines. Simplest and most robust on small corpora. Gives
  one 384-dim vector per film.
* **Per-scene mean-pool, then film mean-pool** — embed every line,
  average within each scene, then average across scenes. Gives the
  same shape but weights scenes equally regardless of dialogue
  density. Plausibly better-aligned with the unit a viewer
  experiences.
* **Per-line mean-pool followed by max-pool** — keep the maximum
  per-dimension value across lines instead of the mean. Captures
  outlier-line content better but loses the average-content signal.

The v1 commits to per-line mean-pool. The two-stage scene-then-
film pool is a reasonable alternative the diagnostic step compares
against; if the second-stage pool produces materially different
results (the diagnostic measures cosine similarity between the two
pooled vectors per film), the proposal v2 will revisit. Per-line
mean-pool is the documented default in Gross (2025) and the
direct prior.

Action text is excluded for the same reason cited in the topic and
sentiment proposals: keep the standalone-lift comparison clean.
Action-text embedding is a defensible Phase 3c combinations
addition, not a v1 default.

### 2.3 Dimensionality reduction: PCA to 32 dimensions

**Decision (v1): PCA fit on training-fold pooled embeddings,
reducing 384-dim to 32-dim.**

Three dimensionality-reduction choices were considered:

* **Raw 384-dim with HistGB only** — let the tree ensemble handle
  high dimensionality natively; skip PCA. Compatible with HistGB
  but degrades linear, KNN, and SVM performance through the curse-
  of-dimensionality on small samples.
* **PCA to 32 dimensions, fit on train fold only** — preserves most
  of the variance (the embedding-space's effective dimensionality
  is typically around 30-50 for this kind of corpus) while keeping
  the feature count manageable for all four model families.
* **UMAP to 8 dimensions, fit on train fold only** — non-linear
  alternative; preserves cluster structure better than PCA but
  adds a stochastic component (UMAP is non-deterministic without
  fixed seeds and even with seeds is sensitive to neighbour-graph
  construction) and requires `umap-learn` as a new dependency.

The v1 commits to **PCA-to-32**. The number 32 is on the higher
end of the sensible range; the diagnostic step measures the
cumulative variance explained at 32 components and the proposal
v2 will revisit if it lands below 70%. PCA is deterministic given
the input matrix and seed, requires no new dependency, and treats
all four model families equally.

A secondary "raw 384-dim with HistGB" run is included as an
ablation diagnostic (not as a feature column in the matrix); the
diagnostic step reports HistGB's OOF metrics on raw vs PCA-32
inputs to surface whether dimensionality reduction costs HistGB
performance. If the cost is large, the proposal v2 will move PCA-
32 to a non-default knob and use raw 384-dim plus a
HistGB-only-with-LDA-projection treatment for the linear / KNN /
SVM families. v1 keeps the cleaner "one matrix, four families"
structure.

---

## 3. Proposed features (32 total)

The 32 features are the components produced by PCA reduction of the
mean-pooled per-line MiniLM embeddings. They have no human-readable
labels (PCA components are linear combinations of the 384 raw
dimensions, themselves abstract), so the feature names are
`embed_pc_00` through `embed_pc_31`.

### 3.1 PCA components 0 through 31 (32 features)

**`embed_pc_00` through `embed_pc_31`.** The 32 leading principal
components of the mean-pooled embedding matrix, fit on the training
fold's 1,199 films and applied to the full 1,713-film corpus.

The first few components typically capture genre-level information
(comedy vs drama vs action), which the genre dummies in the
baseline already encode. The middle components (positions 5-15)
typically capture finer-grained content distinctions (era, register,
specific subject-matter clusters). The trailing components capture
within-genre stylistic variation. The model can extract signal from
any subset of these without needing to interpret them individually.

**Standardization at the model boundary.** PCA produces components
with variance descending from PC1 to PC32. The trainer's existing
`StandardScaler` (applied inside cross-validation folds) standard-
izes each PC to unit variance before the linear, KNN, and SVM
families train. HistGB's numeric branch is passthrough as before.

### 3.2 Why no auxiliary aggregates

The previous four Phase 3b groups each included one or two
auxiliary aggregate features alongside the main feature set
(`hapax_ratio_dialogue` for lexical; `sentiment_volatility_
concentration` for sentiment; `topic_concentration_entropy` for
topic; `network_dialogue_gini` for character-network). Embeddings
do not have a natural analog: the mean-pooled embedding is itself
the only film-level aggregate that the design produces. Within-film
embedding-variance-style aggregates (e.g., the spread of per-line
embeddings around the film mean) were considered and scoped out
on parsimony grounds; if the standalone-lift result is null and
the planning conversation wants to revisit, they are a defensible
v2 addition.

---

## 4. Pre-registered expected lift

These predictions are made before implementation. After
implementation, the actual lift over the Phase 3a revised
dialogue-only floor is recorded alongside these predictions in
`reports/tables/phase3_ablation.csv`. The pre-registered bands
apply to the linear family's OOF numbers.

The mechanism for predictive lift: pre-trained sentence embeddings
encode a large amount of information about dialogue style, register,
era, and subject matter. Even after PCA reduction, the 32 leading
components carry information that genre dummies, structural counts,
and the previous four feature groups cannot fully reproduce. The
literature prior is the strongest of any feature group in the
phase: Gross (2025) reports pooled MiniLM embeddings as the single
strongest predictor on a related task using the same corpus.

### Regression target `log_roi`

Predicted lift in OOF metrics (linear family):

* **RMSE: -0.050 to -0.015** (lower is better; reduction band).
* **MAE: -0.040 to -0.015**.
* **CVRMSE: -0.040 to -0.015**.

Mechanism: pre-trained embeddings capture stylistic and thematic
signal that genre and structural counts only crudely approximate.
The literature prior justifies the wider lower bound (a -0.05 RMSE
reduction is achievable; a -0.015 floor is the bottom of "non-null"
on our scale).

### Classification target `roi_gt_1` (gross-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: 0.000 to +0.020**.
* **PR-AUC: 0.000 to +0.020**.
* **F1: 0.000 to +0.010**.
* **log-loss: -0.025 to -0.005** (reduction band).

Mechanism: as before, the 80% positive base rate keeps available
headroom small. Embeddings may help on the unprofitable minority
by capturing dialogue patterns that audiences reliably reject.

### Classification target `roi_gt_2` (net-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: +0.025 to +0.060**.
* **PR-AUC: +0.020 to +0.045**.
* **F1: +0.005 to +0.025**.
* **log-loss: -0.030 to -0.010** (reduction band).

Mechanism: blockbuster vs mid-budget is the target with the most
headroom and the one where stylistic embeddings are most likely to
help (the dialogue style of franchise films, family adventures,
and four-quadrant comedies is reasonably distinctive and not fully
captured by genre or budget structure).

### Combined expectations

If embeddings land in their predicted band, this is the largest
single standalone lift of Phase 3b and validates the genre-
orthogonality interpretation cleanly. If embeddings land null, the
"every standalone group is null" verdict makes Phase 3c the
indispensable venue for any positive result and forces a re-think
of the Phase 4 model selection (specifically: more-flexible model
families, or adopting the embedding features as raw 384-dim inputs
into HistGB rather than PCA-32 inputs into linear).

---

## 5. Acknowledgement of out-of-scope features (defensibility)

Five classes of embedding-related features are considered and
explicitly deferred.

### 5.1 Embeddings on action text

Pre-trained sentence-transformer encoders are domain-portable, so
embeddings on action text are computationally trivial to add (the
MiniLM forward-pass treats action prose the same as dialogue).
Scoped out for parsimony in v1 (32 dialogue PCs is already on the
high side; doubling the channels doubles the column count without
a clear marginal mechanism). Defensible Phase 3c combinations
addition.

### 5.2 Distance-to-archetype features

Computing cosine similarity between each film's mean-pooled
embedding and a small set of "successful" / "unsuccessful" film
embeddings would produce auxiliary features similar in spirit to
the sentiment group's Reagan-archetype similarities. This is
specifically a leakage risk if "successful" centroids are computed
from training-set outcomes (target leakage). Scoped out for v1; a
defensible alternative formulation (cosine to corpus-fixed
templates derived from screenwriting-craft rubrics) is a
planning-conversation polish item for v2.

### 5.3 Per-scene or per-act embedding aggregates

Embedding the dialogue at scene-level and computing per-act mean
embeddings would produce act-level features that align with the
sentiment group's quartile-pooled trajectory. Crosses two pooling
units (scene then act) and is more naturally evaluated in Phase
3c combinations than as a standalone v1 default.

### 5.4 Fine-tuned encoder

Fine-tuning an encoder on the training-fold dialogue (or on a
training-fold dialogue-versus-outcomes contrastive task) would
produce corpus-specialized embeddings. The leakage risk here is
substantial (any training-fold-fit encoder must be fit
strictly per-fold for the cross-validation evaluation to remain
clean), and the implementation complexity is high. Scoped out
for v1 categorically; a defensible Phase 4 modelling-phase
extension if pre-trained embeddings carry signal but not enough.

### 5.5 Sentence-by-sentence embeddings as input to a sequence model

Treating the per-line embedding sequence as input to an LSTM or
transformer film-level encoder would produce a more expressive
representation than mean-pooling. This is the modelling-phase
direction (Phase 4) rather than the feature-extraction direction
(Phase 3); scoped out without ambiguity.

---

## 6. Feasibility concerns

### 6.1 New dependency: `sentence-transformers`

The `sentence-transformers` package is added to
`requirements.txt`. It pulls in `transformers`, `torch`, and
`tokenizers` as transitive dependencies, adding roughly 1 GB to
the disk-installed environment. Standard for embedding work.

### 6.2 Compute cost (CRITICAL)

Embedding extraction on the full corpus is the most expensive
single step in Phase 3. Estimated time on a single CPU:

* Per-line embedding: 25 to 45 minutes.
* Mean-pooling and PCA fit + transform: under 60 seconds.

The implementation **caches per-film mean-pooled embeddings to
disk** at `data/external/embeddings_minilm_pooled.parquet` after
the first run. Subsequent runs (e.g., re-running the ablation with
a different K for PCA) skip the embedding-extraction step entirely
and rerun only the PCA + feature derivation. The cache is
parameterized by the encoder name; switching encoders triggers a
re-run.

GPU availability would reduce extraction time by roughly 10x. The
implementation auto-detects CUDA and uses it if available, but
defaults to CPU with no manual configuration required.

### 6.3 No-leakage discipline (CRITICAL)

Pre-trained encoders applied as-is to all films do not leak
training-fold information to the cal/test sets. PCA, however, is
fit on data and **must be fit on the training fold only**:

* MiniLM forward-pass on every film's dialogue (1,713 films): no
  leakage, deterministic.
* Mean-pooling per film: no leakage.
* PCA fit on training-fold pooled embeddings (1,199 films): the
  fit step that requires train-only data.
* PCA transform on every film's pooled embedding (1,713 films):
  applied uniformly using the train-fold-fit PCA components.

The proposal v2 will include a `tests/test_embedding.py` test
asserting that adding cal/test films at fit time does not change
the train-film transform outputs (matching the topic-group leakage
test).

### 6.4 Determinism

MiniLM forward-pass is deterministic given the input tokens and
fixed PyTorch seeds. PCA via scikit-learn is deterministic given
the input matrix. The full pipeline is reproducible across reruns
on the same hardware; minor floating-point variation across CPU
versus GPU is expected but immaterial to the feature distribution.

### 6.5 Empty-text dialogue filter

Phase 2 Tier 1.3 filter and the lexical / sentiment / topic groups
already apply the empty-text filter at every dialogue iteration.
Embedding extraction inherits the filter via the same
`_dialogue_lines` helper introduced in `src.features.lexical`.

### 6.6 `data_quality_flag` films

The 30 films with degenerate scene structure have well-defined
dialogue lines (their scene boundaries are collapsed but the lines
themselves are preserved). Mean-pooling per film does not depend
on scene structure. The flagged films use the same embedding
pipeline as the rest of the corpus. The diagnostic step spot-
checks them.

### 6.7 PCA component sign ambiguity

PCA components are determined up to a sign flip; scikit-learn
deterministically chooses the sign such that the largest absolute-
value loading is positive. This makes the components reproducible
across reruns. No special handling is required.

### 6.8 Cache file size

A 1,713-film × 384-dim float32 parquet is roughly 2.5 MB; the
cache is small. The sentence-level intermediate (1.4M × 384-dim)
is roughly 2 GB and is not cached (only the per-film mean-pooled
result).

---

## 7. Implementation sketch

### 7.1 Module layout

* **Module:** `src/features/embedding.py`.
* **External dependencies:** `sentence-transformers`, `torch`,
  `transformers` (transitive). Added to `requirements.txt`.
* **Inputs at compute time:** `ParsedScreenplay` objects from
  `data/processed/screenplays_parsed.pkl`, plus the train-split
  IMDb ID list from `data/processed/split_assignments.parquet`.
* **External cache:** `data/external/embeddings_minilm_pooled.parquet`
  (1,713 × 384, ~2.5 MB). Re-used across runs unless the encoder
  name changes.
* **Output:** a `pd.DataFrame` indexed by `imdb_id` with the 32
  PCA-component feature columns.

### 7.2 Public API

```python
def extract_pooled_embeddings(
    parsed_corpus: dict[str, ParsedScreenplay],
    cfg: EmbeddingFeatureConfig | None = None,
) -> pd.DataFrame: ...
# Returns 1713 x 384, mean-pooled per film. Cached to disk.

def fit_embedding_pca(
    pooled_embeddings: pd.DataFrame,
    train_ids: Sequence[str],
    cfg: EmbeddingFeatureConfig | None = None,
) -> FittedEmbeddingPCA: ...

def compute_embedding_features(
    pooled_embeddings: pd.DataFrame,
    fitted: FittedEmbeddingPCA,
) -> pd.DataFrame: ...
# Returns 1713 x 32, PCA-projected.
```

### 7.3 Configuration knobs (`EmbeddingFeatureConfig`, frozen dataclass)

* `encoder_name`: `"sentence-transformers/all-MiniLM-L6-v2"`.
* `n_pca_components`: 32.
* `device`: `"auto"` (auto-detects CUDA).
* `batch_size`: 64.
* `random_state`: 42.

### 7.4 Determinism

All operations deterministic given input tokens, configuration, and
random seed. Re-running produces byte-identical pooled embeddings
and byte-identical PCA components on the same hardware.

### 7.5 Testing

* Smoke test: extract on the first 5 films, assert shape (5, 384)
  and no NaN. Fit PCA on the first 4 films with 4 components,
  transform the fifth, assert shape (1, 4) and no NaN.
* Unit test: PCA reconstruction on a fixed fixture produces output
  whose top 1 component captures more variance than the top 5
  combined of a random reduction.
* No-leakage test: fit PCA on `train_ids[:100]`, transform on
  `train_ids[:10]`. Then fit a second PCA on `train_ids[:100] +
  test_ids[:10]`, transform on `train_ids[:10]`. The two transform
  outputs match within numerical tolerance (signs of components
  may flip; absolute values are equal).
* Integration test: extract on 50 films, fit PCA on the first 40,
  transform on all 50, assert (50, 32) shape with no NaN.

### 7.6 Multi-family ablation through `save_run`

A new `src/experiments/run_embedding_ablation.py` mirrors the
lexical, sentiment, topic, and character-network runners.
Standalone-lift methodology: the augmented matrix joins the
structural baseline features and the 32 PCA-projected embedding
features. Lift is computed against the **Phase 3a revised dialogue-
only floor**.

A secondary diagnostic run extracts HistGB OOF metrics on a raw
384-dim embedding matrix (without the structural baseline) and on
the 32-PCA matrix; the comparison is reported in the handoff but
does not become a separate ablation row.

---

## 8. Post-implementation diagnostic checks

1. **PCA cumulative variance explained at K = 32.** Threshold:
   below 70% prompts a `n_pca_components` revisit. Diagnostic
   only; report value in handoff.
2. **Pairwise correlations between PCA components and structural
   baseline features.** Threshold: |r| > 0.85 prompts a pair
   review. Particular concern: the leading PCA components often
   correlate strongly with genre dummies and `release_year_parsed`.
3. **Pairwise correlations between PCA components and other Phase
   3b feature groups.** Particular concern: PC1 / PC2 often align
   with topic-distribution PC1 / PC2 because both capture coarse
   content distinctions.
4. **Encoder hash check.** Verify that the cached encoder is the
   exact `all-MiniLM-L6-v2` checkpoint by recomputing the
   embedding of a fixed reference sentence and checking against a
   committed expected vector.
5. **Pooling-strategy comparison.** Compute per-scene-then-film
   mean-pool embeddings as well as per-line mean-pool. Report
   the cosine similarity between the two pooled vectors per film;
   if median similarity is below 0.95, the pooling choice is
   non-trivial and the proposal v2 should revisit.
6. **`data_quality_flag` films vs unflagged.** Spot-check the
   embedding-PCA features of the 24 train-split flagged films;
   they should be within 1.0 z-score of the unflagged means.
7. **Univariate target correlations.** Computed on the train
   split. Particularly informative here: if any single PCA
   component exceeds |r| = 0.10 with any target, the pre-
   registered lift expectation gains support.
8. **Raw 384-dim HistGB diagnostic.** Run HistGB OOF on raw 384-
   dim embeddings (no structural baseline) and on 32-PCA + baseline.
   Report both AUC-ROC values; if raw substantially outperforms
   PCA + baseline on `roi_gt_2`, the proposal v2 will rework
   dimensionality reduction.

---

## 9. References

* Reimers, N., and Gurevych, I. (2019). "Sentence-BERT: Sentence
  Embeddings using Siamese BERT-Networks." *EMNLP*. The
  sentence-transformers framework citation.
* Wang, W., et al. (2020). "MiniLM: Deep Self-Attention Distillation
  for Task-Agnostic Compression of Pre-Trained Transformers."
  *NeurIPS*. The MiniLM model citation.
* Song, K., et al. (2020). "MPNet: Masked and Permuted Pre-training
  for Language Understanding." *NeurIPS*. The mpnet alternative
  citation.
* Gross, T. (2025). "Predicting Oscar-Nominated Screenplays with
  Sentence Embeddings." Direct prior on this corpus with this
  feature family.
* Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2023).
  "MTEB: Massive Text Embedding Benchmark." Source for encoder
  performance comparisons.

---

Proposal v1 for the **embedding** feature group is ready. Please
bring to the planning conversation for review before implementation.
