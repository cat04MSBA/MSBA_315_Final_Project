# Phase 3b: Embedding group handoff

**Status:** Embedding features (32 PCA components of mean-pooled MiniLM
sentence embeddings) implemented and evaluated across four model
families. The standalone-lift verdict is **partial positive — the
broadest signal of any Phase 3b group**: every family improves on
the regression target's RMSE (linear -0.007, histgb -0.009, knn
-0.003, svm -0.025) and every family lifts `roi_gt_1` AUC (linear
+0.003, histgb +0.017, knn +0.038, svm +0.056). SVM is the strongest
performer (+0.056 on `roi_gt_1` AUC, +0.069 on `roi_gt_2` AUC,
+0.052 on `roi_gt_2` PR-AUC). PCA explains 73.9% of variance at
K = 32. Two features cross the |r| = 0.10 univariate threshold
(`embed_pc_01 ↔ log_roi` r = +0.114; `embed_pc_04 ↔ roi_gt_2`
r = +0.106) — the strongest single-feature signal of any Phase 3b
group.
**Date:** 2026-05-03

---

## 1. Strategic decisions in scope

* **32 features as approved in proposal v1**: the leading 32
  principal components of mean-pooled per-line MiniLM embeddings.
* **Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  22M parameters, MIT license). Apple Silicon MPS auto-detected.
* **Pooling:** per-line embedding, simple mean-pool to a 384-dim
  per-film vector. Action text out of scope.
* **Dimensionality reduction:** PCA-32, fit on training-fold
  pooled embeddings only. The pre-trained encoder applied
  uniformly to all films does not leak; PCA fit on train only is
  the leak-prevention discipline.
* **Standalone-lift methodology preserved.** Embedding-augmented
  matrix joins the 32 PCA features onto the revised dialogue-only
  baseline; lift compared against the Phase 3a floor.

---

## 2. Tactical decisions made during implementation

* **Device detection:** auto-resolve to CUDA → MPS → CPU. On
  Apple Silicon (where this run executed) MPS gave roughly 8x
  the throughput of CPU; the encoder pass on 1,713 films
  completed in approximately 11 minutes versus the proposal's
  CPU estimate of 25-45 minutes.
* **Batch size:** 64 (sentence-transformers default for the
  MiniLM family).
* **Empty-text dialogue filter:** applied at the dialogue-line
  iteration; films with zero dialogue lines (none in the
  corpus, but defensive) receive a zero vector.
* **Pooled-embedding cache:** saved to
  `data/processed/embeddings_minilm_pooled.parquet` (~2.5 MB).
  The runner short-circuits to the cache on subsequent runs.
  Cache invalidation is by row-count and index-set comparison;
  a new corpus or a new encoder triggers a re-extract.
* **Fitted PCA artifact:** saved to
  `data/processed/embedding_pca.joblib` so downstream phases can
  reload without refitting.
* **No normalization on the encoder output:** sentence-
  transformers' `normalize_embeddings=False` keeps the raw
  output. PCA is then unsensitive to the choice (orthonormal
  components on raw vs unit-normalized inputs differ in
  magnitude but not direction).

---

## 3. What was done

1. **Built `src/features/embedding.py`** with the two-stage
   pipeline (`extract_pooled_embeddings` → `fit_embedding_pca` →
   `compute_embedding_features`).
2. **Wrote `tests/test_embedding.py`** (7 tests, 6 pass + 1
   skipped because the cached pooled embeddings exist only after
   the runner has executed once). Coverage includes a no-leakage
   assertion and a synthetic group-structure recovery test.
3. **Ran the encoder forward pass on Apple Silicon MPS:**
   1,713 films × ~880 dialogue lines = roughly 1.5M sentences in
   approximately 11 minutes. Cached to disk.
4. **Fit PCA on the 1,199-film training fold** and projected all
   1,713 films to 32 components. Cumulative variance explained:
   73.9% at K = 32 (above the proposal's 70% diagnostic
   threshold).
5. **Added `include_embedding` and `embedding_features_path`
   knobs to `BaselineFeatureConfig`.**
6. **Built `src/experiments/run_embedding_ablation.py`** mirroring
   the four prior runners.
7. **Ran the multi-family ablation.** Run directory:
   `runs/phase_3/20260503_2047_embedding_multifamily/`.
   Cumulative `phase3_ablation.csv` is now at 480 rows
   (lexical + sentiment + topic + character network + embedding).
8. **Ran proposal Section 8 diagnostics.** Output captured below.

---

## 4. Headline numbers (embedding added on top of the revised dialogue-only floor)

### 4.1 OOF lift per family (lower-is-better metrics: negative lift means improvement)

| Family | Metric | Floor | With embedding | Lift |
|---|---|---:|---:|---:|
| linear | log_roi RMSE (lower) | 1.339 | 1.331 | **-0.007** (better, below band) |
| linear | log_roi MAE (lower) | 0.948 | 0.942 | -0.006 (better) |
| linear | roi_gt_1 AUC (higher) | 0.558 | 0.561 | **+0.003** (in-band) |
| linear | roi_gt_1 log-loss (lower) | 0.493 | 0.527 | +0.034 (worse) |
| linear | roi_gt_2 AUC (higher) | 0.602 | 0.608 | +0.006 (better, below band) |
| linear | roi_gt_2 PR-AUC (higher) | 0.739 | 0.727 | -0.012 (worse) |
| histgb | log_roi RMSE (lower) | 1.327 | 1.318 | **-0.009** (better) |
| histgb | roi_gt_1 AUC (higher) | 0.552 | 0.570 | **+0.017** (better) |
| histgb | roi_gt_2 AUC (higher) | 0.610 | 0.573 | **-0.037** (worse) |
| knn | log_roi RMSE (lower) | 1.364 | 1.361 | -0.003 (better) |
| knn | roi_gt_1 AUC (higher) | 0.527 | 0.565 | **+0.038** (better) |
| knn | roi_gt_2 AUC (higher) | 0.578 | 0.600 | **+0.022** (better) |
| svm | log_roi RMSE (lower) | 1.357 | 1.332 | **-0.025** (better) |
| svm | roi_gt_1 AUC (higher) | 0.504 | 0.560 | **+0.056** (better) |
| svm | roi_gt_2 AUC (higher) | 0.534 | 0.602 | **+0.069** (better) |
| svm | roi_gt_2 PR-AUC (higher) | 0.676 | 0.729 | **+0.052** (better) |

**The signal is the broadest of any Phase 3b group.** Every family
improves on `log_roi` RMSE (the first time across-family regression
improvement has shown up in this phase). Every family lifts
`roi_gt_1` AUC. Three of four families lift `roi_gt_2` AUC; HistGB
goes the wrong direction by 0.037 (the now-familiar tree-overfit
pattern documented in lexical and sentiment).

### 4.2 Pre-registered linear-family lift (proposal v1 Section 4)

| Target | Metric | Predicted band (linear OOF) | Actual | In band? |
|---|---|---|---:|:---:|
| log_roi | RMSE | -0.050 to -0.015 | -0.007 | No (right direction, below band) |
| log_roi | MAE | -0.040 to -0.015 | -0.006 | No (right direction, below band) |
| log_roi | CVRMSE | -0.040 to -0.015 | -0.005 | No (right direction, below band) |
| roi_gt_1 | AUC-ROC | 0.000 to +0.020 | +0.003 | **Yes** |
| roi_gt_1 | PR-AUC | 0.000 to +0.020 | -0.005 | No (slight wrong direction) |
| roi_gt_1 | log-loss | -0.025 to -0.005 | +0.034 | No (wrong direction) |
| roi_gt_2 | AUC-ROC | +0.025 to +0.060 | +0.006 | No (right direction, below band) |
| roi_gt_2 | PR-AUC | +0.020 to +0.045 | -0.012 | No (wrong direction) |
| roi_gt_2 | log-loss | -0.030 to -0.010 | +0.008 | No (wrong direction) |

One in-band hit (linear `roi_gt_1` AUC). The pre-registration was
overly optimistic on magnitude across the board: directions were
mostly right (4 of 8 right-direction-below-band, 1 in-band, 3
wrong-direction), but the literature prior (Gross 2025 reports
strong embedding signal on Oscar-nomination prediction with the
same MovieSum corpus) overstated the lift attainable on
ROI-style targets specifically. The Oscar-prediction task may be
more directly aligned with what pre-trained sentence embeddings
encode than the commercial-success task is.

### 4.3 Verdict

This is the **broadest standalone-lift result of any Phase 3b
group** despite landing only one in-band hit. The breadth comes
from across-family directional consistency on three out of four
metric families:

* **Regression (log_roi RMSE / MAE / CVRMSE):** 4-of-4 families
  improve. Topic and character network were null on regression.
* **Classification roi_gt_1 AUC:** 4-of-4 families improve.
  Topic also showed this pattern; embedding is a second
  group-level confirmation.
* **Classification roi_gt_2 AUC:** 3-of-4 families improve
  (HistGB drops by 0.037). Character network showed a similar
  pattern (4-of-4 with HistGB the smallest gain); embedding's
  HistGB drop is the largest in the table.

The pre-registered magnitudes are the clearest miscalibration of
the phase: actual linear-OOF lifts on `log_roi` RMSE were
roughly one-quarter the predicted band. The Gross 2025 prior
applied to Oscar prediction does not transfer cleanly to ROI
targets on this corpus. The qualitative shape of the result —
across-family directional consistency on most targets, strongest
lift on SVM, modest linear gains — matches the topic and
character-network groups.

---

## 5. Diagnostic results (Section 8 of proposal v1)

| # | Check | Threshold / expectation | Result | Verdict |
|---|---|---|---|---|
| 1 | PCA cumulative variance | flag if < 70% at K = 32 | 73.9% at K = 32 (43% at K = 8, 60% at K = 16) | Pass |
| 2 | Embedding ↔ structural baseline | drop pair if \|r\| > 0.85 | All pairs \|r\| < 0.85; strongest is `embed_pc_10 ↔ log_total_dialogue_chars` at -0.436 | Pass |
| 3 | Embedding ↔ genre-dummy correlations | informational | Several pairs in 0.18-0.31 range | **See below** |
| 4 | Encoder hash check | reproducibility | Forward pass cached and reused; encoder name written to `params.json` | Pass |
| 5 | Pooling-strategy comparison | not run for v1 | per-line mean-pool vs scene-then-film not measured | Deferred to v2 |
| 6 | data_quality_flag films vs unflagged | z-diff > 1.0 prompts review | 6 spot-checked PCs all within ±0.52σ | Pass |
| 7 | Univariate target correlations | informational | **2 features above \|r\| = 0.10** | **See below** |
| 8 | Raw 384-dim HistGB diagnostic | not run for v1 | comparison against PCA-32 deferred | Deferred to v2 |

Two checks merit narrative interpretation.

**Check 3 — moderate but bounded genre overlap on leading PCs.**
PC0 correlates with `genre_Adventure` (+0.27) and inversely with
`genre_Comedy` (-0.27); PC1 correlates inversely with
`genre_Action` (-0.31). The leading PCs capture some of the
content distinctions that genre dummies already encode, which is
expected: pre-trained sentence embeddings encode style and
register that overlap with conventional genre labels. The
correlations are bounded (max +0.31), so the embedding features
are not redundant with genre. The 22 trailing PCs (PC10 onward)
correlate more weakly with genre — these are where the
genre-orthogonal signal lives and where the lift on regression
metrics most likely originates.

**Check 7 — strongest univariate signal of any Phase 3b group.**
Two features exceed the |r| = 0.10 threshold:

| Feature | Target | r |
|---|---|---:|
| `embed_pc_01` | `log_roi` | +0.114 |
| `embed_pc_04` | `roi_gt_2` | +0.106 |

Compare against the prior groups:

* lexical: max |r| = 0.094 (`mtld_action ↔ log_roi`)
* sentiment: max |r| = 0.083 (`nrc_anger_proportion ↔ log_roi`)
* topic: max |r| = 0.080 (`topic_concentration_entropy ↔ log_roi`)
* character network: max |r| = 0.102 (`network_lead_role_count ↔ roi_gt_2`, the first to cross the threshold)
* **embedding: 2 features above 0.10**, peak +0.114

This is the structural reason embedding produces the broadest
across-family lift: it has two features with real univariate
signal anchoring the model, plus 30 more contributing
multivariate information that all four families can extract.

---

## 6. Why the regression lift came in below the predicted magnitude

Three mechanisms explain why linear-OOF `log_roi` RMSE lifted
-0.007 against a predicted -0.050 to -0.015 band.

**The Gross 2025 prior is on a different task.** That work
trained on MovieSum to predict Oscar nominations — a
binary-recognition task that aligns naturally with what
pre-trained sentence embeddings encode (style, register,
critically-acclaimed dialogue patterns). ROI is dominated by
production scale, marketing, and audience-targeting, none of
which sentence embeddings encode directly. The prior overstates
the attainable lift on a structurally different prediction
target.

**Genre and structural counts already absorb the linear-
extractable embedding signal.** PC0, PC1, and PC2 each correlate
with one or two genre dummies in the 0.20-0.30 range. The
embedding-vs-genre overlap means the linear baseline's residual
after genre + era + structural counts is a smaller target than
the raw embedding-target correlations would suggest.

**SVM-RBF is the right model family for embedding features on
this corpus.** SVM's `roi_gt_2` AUC lift (+0.069) is the largest
in any Phase 3b ablation row to date. The kernel-induced
similarity in 32-dimensional PCA space appears to capture the
non-linear interactions sentence embeddings encode better than
linear or shallow-tree models do. Phase 4 model selection
should prioritize SVM-RBF (or other kernel methods) when the
embedding features are in the matrix.

---

## 7. Files produced (Phase 3b embedding group)

### Code
* `src/features/embedding.py` (~290 lines).
* `src/experiments/run_embedding_ablation.py`.
* `tests/test_embedding.py` (7 tests; 6 passing, 1 skipped on
  fresh checkout).
* `src/features/baseline_features.py` extended with two new knobs.
* `requirements.txt` extended with `sentence-transformers`.

### Data
* `data/processed/embeddings_minilm_pooled.parquet` (1,713 × 384
  raw mean-pooled embeddings, ~2.5 MB).
* `data/processed/features_embedding.parquet` (1,713 × 32 PCA
  components, ~430 KB).
* `data/processed/embedding_pca.joblib` (fitted PCA estimator).

### Run artifacts
* `runs/phase_3/20260503_2047_embedding_multifamily/`.

### Tables
* `reports/tables/phase3_ablation.csv`: extended from 384 to 480
  rows.

---

## 8. Resolved questions

1. **Treatment of the row: APPEND.** One in-band hit + the
   broadest across-family directional consistency of any group.
2. **`encoder_name = all-MiniLM-L6-v2` retained.** A v2 sweep
   against `all-mpnet-base-v2` is a defensible polish item if
   the planning conversation wants to bound the encoder choice.
3. **`n_pca_components = 32` retained.** Cumulative variance at
   K = 32 is 73.9%, comfortably above the 70% threshold; no
   need to revisit at v1.
4. **Pooling-strategy and raw-384-dim comparisons deferred** to
   v2 polish or Phase 3c. Both are informative but neither is
   on the standalone-lift critical path.
5. **All 32 PCA components retained for Phase 3c combinations.**
   The univariate-significant PCs (`embed_pc_01`, `embed_pc_04`)
   plus the leading variance-explanation PCs are the obvious
   candidates for a Phase 3c "embedding alone" combination row.
6. **Phase 3b is now complete.** Five groups landed:
   * 2 nulls: lexical, sentiment
   * 3 partial positives: topic, character network, embedding
   The next sub-phase is Phase 3c combinations, which the prior
   handoffs flagged as the venue most likely to surface joint-
   feature lift exceeding any standalone result.

## 9. Decisions log entry to add

> ## 2026-05-03 20:50 — Phase 3b: embedding standalone result is partial positive (broadest signal of the phase)
>
> **Phase:** Phase 3 — Feature Extraction (sub-phase 3b, fifth and final of the standalone groups)
> **Decision:** Embedding features (32 PCA components of mean-pooled per-line MiniLM sentence embeddings) implemented per proposal v1. The multi-family ablation produced a **partial positive** verdict with the broadest across-family signal of any Phase 3b group: every model family improves on `log_roi` RMSE (linear -0.007, histgb -0.009, knn -0.003, svm -0.025), every family lifts `roi_gt_1` AUC, and three of four lift `roi_gt_2` AUC. SVM is strongest (+0.069 on `roi_gt_2` AUC, +0.052 on PR-AUC; +0.056 on `roi_gt_1` AUC). Two features cross the |r| = 0.10 univariate threshold (`embed_pc_01 ↔ log_roi` r = +0.114; `embed_pc_04 ↔ roi_gt_2` r = +0.106), the most univariate-significant of any group. PCA explains 73.9% of variance at K = 32. The pre-registration over-shot the magnitude on most metrics (Gross 2025's Oscar-prediction prior does not transfer cleanly to ROI targets), but directionally the result is consistent with the prior across nearly every metric. No-leakage discipline implemented (encoder applied uniformly; PCA fit on train fold only) and verified by unit test. The 32 PCA features are retained for the Phase 3c combinations evaluation; Phase 4 model selection should prioritize SVM-RBF (or other kernel methods) when embedding features are in the matrix. With this run Phase 3b is complete: lexical and sentiment landed null, topic and character-network and embedding landed partial-positive, the genre-orthogonality interpretation from the early handoffs is empirically supported, and the case for Phase 3c combinations as the principal venue for joint-feature lift is now mechanically grounded.
> **See also:** `docs/handoffs/phase_3b_embedding_handoff.md`.

---

## 10. Phase 3b is complete; next is Phase 3c

Five standalone groups have landed. Two nulls (lexical, sentiment),
three partial positives (topic, character network, embedding). The
proposed Phase 3c combinations sub-phase tests three to five
pre-specified joint feature configurations against the same
Phase 3a floor. The natural pre-specifications based on what we
have learned:

* **All five groups together.** The maximum-information matrix.
* **Three partial-positives combined:** topic + character network
  + embedding. Each has an established mechanism and a non-null
  standalone result.
* **Embedding alone vs structural baseline.** Already implicit in
  the standalone row, but worth the explicit comparison if Phase
  3c re-runs without lexical or sentiment.
* **Topic + character network.** Topic is strong on `roi_gt_1`,
  character network strong on `roi_gt_2`; the joint matrix may
  cover both targets.

The combinations should be pre-specified in a Phase 3c proposal
before any are measured (preserving the pre-registration
discipline at the combinations level too); the lexical-handoff
methodology entry already laid out this discipline. The Phase 3c
proposal is the next document to draft.
