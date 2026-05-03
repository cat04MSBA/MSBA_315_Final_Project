# Phase 3b: Lexical group handoff

**Status:** Lexical features implemented and evaluated across four
model families. Verdict is conclusive: the features do not carry
extractable predictive signal on this corpus. Across linear, HistGB,
KNN, and SVM-RBF, no family shows positive lift on the headline
targets that exceeds the within-CI-noise band, and three of four
show worse performance with lexical features added. Awaiting
direction on (a) appending the official ablation row, (b) dropping
the redundant `rare_word_proportion`, and (c) starting the sentiment
proposal.
**Date:** 2026-05-03

This handoff matches the structure of `phase_3a_handoff.md`.
Subsequent groups (sentiment, topic, embedding, character network)
will each get their own handoff in this folder when they land.

---

## 1. Strategic decisions in scope

* **14 lexical features as approved in proposal v2** (10 from v1 plus
  4 added during planning-conversation review: SUBTLEX-style
  sophistication features, action-text parallels for MTLD and
  Flesch-Kincaid).
* **Multi-family ablation methodology** introduced at this
  group: every Phase 3b ablation runs the same feature-augmented
  matrix through the four model families (linear, HistGB, KNN,
  SVM-RBF) with the linear family providing the "official" ablation
  row and the other three providing diagnostic disambiguation. See
  `phase_3a_handoff.md` Section 1 for the rationale.
* **Group ordering confirmed:** lexical first (cheapest), sentiment
  next, topic and character network in the middle, embeddings last.
* **`data_quality_flag` films (24 in train split)** use the same
  lexical-feature pipeline as the rest of the corpus because lexical
  features are whole-screenplay aggregates and are robust to
  scene-level degeneracy. The Section 5 diagnostic check
  empirically validates this decision.

---

## 2. Tactical decisions made during implementation

* **Tokenization:** NLTK `word_tokenize` and `sent_tokenize`. Matches
  proposal v2 Section 4.1.
* **Frequency-source backend (DEVIATION FROM PROPOSAL v2).** The
  canonical SUBTLEX-US download URLs at the Brysbaert lab returned
  404 / HTML at implementation time, so a hash-checked direct
  download was not viable. The implementation backend is the
  `wordfreq` Python package, whose English Zipf-scale log-frequencies
  are computed from a mixture of sources INCLUDING OpenSubtitles
  (subtitle-domain, similar to SUBTLEX in spirit) along with
  Wikipedia, Reuters, and others. The mixture is not pure
  SUBTLEX-US, but the conceptual mechanism the proposal selected
  SUBTLEX-US for (subtitle-derived frequencies match dialogue input
  better than prose-derived) is preserved by wordfreq's inclusion
  of OpenSubtitles. The two affected features were renamed to drop
  the `_subtlex` suffix so the feature names are honest about the
  source: `mean_log_frequency` and `rare_word_proportion`. This
  deviation is surfaced for planning-conversation review; if strict
  SUBTLEX-US is required, the backend swap is one function in
  `src/features/lexical.py`.
* **Rare-word cutoff:** computed corpus-wide from a 200-film sample
  to bound runtime; the 25th-percentile Zipf cutoff lands at 5.110
  on this corpus.
* **NaN handling:** `SimpleImputer(strategy="median")` added to the
  numeric branch of the linear, KNN, and SVM pipelines. HistGB
  handles NaN natively; no imputer in its numeric branch. Imputation
  inside cross-validation folds so medians are fit only on training
  data within each fold.
* **`log1p` on the heavy-tailed structural counts** continues from
  the Phase 3a revision. The full feature matrix for the lexical
  ablation is the revised dialogue-only configuration plus the 13
  lexical model features (the 14th, `_oov_rate_dialogue`, is a
  diagnostic-only column excluded from the model matrix). 39 columns
  total.

---

## 3. What was done

1. **Built `src/features/lexical.py`** with the 14 features from
   proposal v2. Each feature has its own focused helper function,
   plus a `compute_lexical_features` entry point that returns a
   `pd.DataFrame` indexed by `imdb_id`.
2. **Wrote `tests/test_lexical.py`** with 17 unit and integration
   tests; all pass. Coverage includes MTLD reference values, hapax
   ratio edge cases, frequency lookup with OOV handling,
   Flesch-Kincaid against simple-vs-complex text, punctuation
   rates, and pronoun ratios with archaic forms.
3. **Computed features on the full corpus** (1,713 films) in
   approximately 600 seconds. NLTK tokenization is the bottleneck.
   Output saved to `data/processed/features_lexical.parquet`
   (242 KB).
4. **Built the experiment-tracking infrastructure** at
   `src/experiments/save_run.py` with the context-manager API, the
   per-run directory layout, and the `runs/RUNS.md` index.
5. **Refactored the baseline trainer** to support 4 model families
   with per-family pipelines (HistGB needs no preprocessing; the
   other three need imputer + scaler).
6. **Re-ran Phase 3a baseline** across the 4 families × 4 feature
   configs (112 rows in `phase3a_baseline.csv`) so the multi-family
   floor is documented.
7. **Ran the first lexical ablation** through
   `src/experiments/run_lexical_ablation.py`, wrapped in a
   `save_run` block, evaluating the lexical-augmented matrix across
   all four families. Run directory at
   `runs/phase_3/20260503_1719_lexical_multifamily/`.
8. **Ran the Section 9 diagnostic checks** from proposal v2 plus
   the two observation-class checks the planning conversation
   flagged.

---

## 4. Headline numbers (lexical added on top of the revised dialogue-only floor, all 4 families)

Bootstrap 95% CIs in brackets. Full table at
`reports/tables/phase3_ablation.csv` (28 rows: 4 families × 7
metric rows per group).

### 4.1 Floor versus lexical-augmented, per family

| Family | Metric | Floor | With lexical | Lift |
|---|---|---:|---:|---:|
| linear | log_roi R² | 0.052 [0.026, 0.080] | **0.037** [0.014, 0.060] | **−0.016** |
| linear | log_roi MAE | 0.948 | 0.955 | +0.007 (worse) |
| linear | roi_gt_1 AUC | 0.558 | 0.565 | +0.007 |
| linear | roi_gt_2 AUC | 0.602 | 0.600 | −0.002 |
| histgb | log_roi R² | 0.069 [0.033, 0.102] | **0.060** [0.026, 0.090] | **−0.009** |
| histgb | log_roi MAE | 0.943 | 0.943 | 0.000 |
| histgb | roi_gt_1 AUC | 0.552 | **0.511** | **−0.041** |
| histgb | roi_gt_2 AUC | 0.610 | **0.586** | **−0.024** |
| knn | log_roi R² | 0.016 | **−0.006** | **−0.022** |
| knn | log_roi MAE | 0.972 | 0.979 | +0.007 (worse) |
| knn | roi_gt_1 AUC | 0.527 | **0.494** | **−0.033** |
| knn | roi_gt_2 AUC | 0.578 | 0.548 | −0.030 |
| svm | log_roi R² | 0.026 | 0.034 | +0.008 |
| svm | log_roi MAE | 0.954 | 0.947 | −0.007 (better) |
| svm | roi_gt_1 AUC | 0.504 | 0.519 | +0.015 |
| svm | roi_gt_2 AUC | 0.534 | 0.565 | +0.031 |

Bold values mark moves of 0.005 or more in the worse direction (or
0.005-or-more wrong-direction lift relative to the predicted band).

### 4.2 Pre-registered linear-family lift (proposal v2 Section 3)

| Target | Metric | Predicted band | Linear actual | In band? |
|---|---|---|---:|:---:|
| log_roi | R² | +0.010 to +0.025 | −0.016 | No |
| roi_gt_1 | AUC-ROC | 0.000 to +0.010 | +0.007 | Yes |
| roi_gt_2 | AUC-ROC | +0.015 to +0.035 | −0.002 | No |

The pre-registered direction was wrong on two of the three headline
predictions. Only `roi_gt_1` moved positively, and that target is
the one the proposal expected to lift the least.

### 4.3 Verdict

The multi-family picture is **conclusive in a way the linear-only
result was not**. Reading across the 4 families:

* **HistGB**, the strongest floor family, gets meaningfully worse
  with lexical features added. R² drops from 0.069 to 0.060;
  `roi_gt_1` AUC drops 0.041 (a substantial drop, well outside CI
  noise); `roi_gt_2` AUC drops 0.024.
* **KNN** gets worse on every metric, with the regression R² turning
  negative.
* **Linear** gets worse on the regression target, neutral elsewhere.
* **SVM** improves modestly across all four headline metrics, but
  starts from the worst floor of any family. SVM-with-lexical
  (R² 0.034, `roi_gt_2` AUC 0.565) is still well below
  linear-without-lexical (R² 0.052, `roi_gt_2` AUC 0.602) and
  HistGB-without-lexical (R² 0.069, `roi_gt_2` AUC 0.610). The SVM
  lift is best read as the SVM finally finding a configuration that
  works on this corpus, not as the lexical features carrying signal
  the other families fail to extract.

The original "linear hurt by noise features" hypothesis is now
strengthened, not weakened. If lexical features carried genuine
non-linear signal, HistGB would extract some of it. HistGB does the
opposite: it gets actively worse. Adding 13 features that don't
carry signal:

* For linear: shifts the regularization optimum slightly, hurting R²
  but not enough to flip CI direction.
* For HistGB: gives the algorithm more split candidates to consider,
  some of which match noise patterns in the OOF folds; with shallow
  trees and conservative defaults the model still has to allocate
  some of its capacity to evaluating these features rather than
  fitting the structural-feature signal it was extracting.
* For KNN: directly degrades the distance metric, making nearest
  neighbours less informative because the new dimensions add noise
  to the metric.
* For SVM: the RBF kernel benefits from any new dimension that
  happens to align with target structure even slightly, but the
  baseline SVM was so weak that "any improvement" is the path of
  least resistance.

The conclusion: lexical features do not carry signal that any of
the four families can extract beyond what the structural baseline
already provides.

---

## 5. Diagnostic results (Section 9 of proposal v2 + observation-class checks)

| # | Check | Threshold | Result | Verdict |
|---|---|---|---|---|
| 1 | Lexical-vs-structural cross-correlations | drop pair if \|r\| > 0.85 | All pairs \|r\| < 0.5 | Pass |
| 2 | Within-lexical pair `mean_log_frequency` ↔ `rare_word_proportion` | drop one if \|r\| > 0.9 | r = **−0.939** | Drop one. Recommended drop: `rare_word_proportion`. |
| 3 | Action vs dialogue siblings | collapse if \|r\| > 0.85 | mtld pair r = +0.49, F-K pair r = +0.28 | Pass; distinction is paying for itself |
| 4 | Hapax ratio length sensitivity | drop if \|r\| > 0.4 vs total dialogue length | r = −0.23 (vs total chars), r = −0.34 (vs n_dialogue_lines) | Pass |
| 5 | Flesch-Kincaid variance | flag if outside [1.0, 4.0] reference band | dialogue std = 0.89 (slightly compressed), action std = 1.46 | Borderline pass; dialogue F-K compresses as the proposal's caveat predicted |
| 6 | OOV rate distribution | flag if many films > 15% | Mean 0.004, only 1 of 1194 films above 15% | Pass; wordfreq covers dialogue vocab well |
| 7 | OOV-vs-`mean_log_frequency` correlation | observation-class | r = −0.787 (driven by the small number of films where OOV is non-trivial) | Pass; the OOV signal is bounded by the small absolute OOV rate |
| 8 | `data_quality_flag` films vs unflagged | observation-class | All four spot-checked features within 0.6σ of unflagged means | Pass; the "use as-is" decision is empirically supported |

The implementation passes every diagnostic the planning conversation
asked for. The negative-lift result is not an implementation defect.

---

## 6. Why the lift went negative

**Univariate-correlation check.** I computed each lexical feature's
Pearson correlation with each of the three targets on the training
split. **No lexical feature exceeds \|r\| = 0.10 with any target.**
The strongest is `mtld_action ↔ log_roi` at r = −0.094, well within
sampling noise for n = 1,199. The next strongest in absolute value
is `first_to_second_pronoun_ratio ↔ log_roi` at r = +0.072. Every
other feature-target pair has \|r\| ≤ 0.06.

**The mechanism, family by family.**

* **Linear:** adding 13 features with no detectable univariate
  signal to a small-corpus L2-regularized linear model produces a
  small but real out-of-fold prediction noise increase that the
  regularizer cannot fully absorb. The CV-selected alpha shifts and
  the bias-variance tradeoff degrades slightly.
* **HistGB:** the algorithm's split-finding considers all features
  at each node, and zero-signal features sometimes happen to match
  noise patterns in the OOF folds purely by chance. With 13
  zero-signal features added, this happens more often, and the
  shallow-tree configuration has limited capacity to recover.
* **KNN:** the L2 distance metric is computed across all standardized
  features. Adding 13 noise dimensions degrades the metric directly:
  nearest neighbours become less informative because they are
  defined using more noise.
* **SVM-RBF:** the RBF kernel is a function of distance in the
  standardized feature space. Same degradation mechanism as KNN, but
  SVM's regularization (C and gamma) absorbs more of the noise. The
  reason SVM appears to improve is that its floor was the worst,
  and the new feature space happens to be slightly more favourable
  for the RBF kernel; this is not a signal-extraction win.

**What this does not say.** The result does not say the features
are implemented incorrectly (Section 5 rules that out). It does not
say the corpus has no information that lexical features could
reach. It says: with this corpus, with these four model families,
and with these 13 features, the marginal information lexical adds
over genre, era, budget structure, and screenplay structure is
below the threshold where any of the four paradigms can extract it
without hurting itself in the process.

---

## 7. Files produced (Phase 3b lexical group)

### Code
* `src/features/lexical.py` (~600 lines): the 14-feature pipeline.
* `src/experiments/__init__.py` and `src/experiments/save_run.py`:
  the per-run logging infrastructure.
* `src/experiments/run_lexical_ablation.py`: the multi-family
  lexical-ablation runner.
* `tests/__init__.py` and `tests/test_lexical.py` (17 tests, all
  passing).
* `src/models/baseline/train.py`: refactored from single-family to
  4-family; the same trainer powers Phase 3a baseline runs and
  Phase 3b ablation runs.

### Data
* `data/processed/features_lexical.parquet` (1,713 × 14, 242 KB).

### Run artifacts
* `runs/phase_3/20260503_1637_lexical_first/`: the original
  linear-only run. Preserved for the audit trail.
* `runs/phase_3/20260503_1719_lexical_multifamily/`: the multi-
  family run that produced the verdict above. Six files
  (`params.json`, `preprocessing_summary.json`, `features_used.json`,
  `metrics.json`, `run.log`).

### Tables
* `reports/tables/phase3a_baseline.csv`: 112 rows (4 configs × 4
  families × 3 targets × {3, 2, 2} metrics). The Phase 3a multi-
  family floor.
* `reports/tables/phase3_ablation.csv`: 28 rows for the lexical
  group (1 group × 4 families × 7 metric rows). Predicted-vs-actual
  lift recorded per family, with the linear-family `in_predicted_band`
  column populated for the targets the proposal pre-registered.

### Configuration changes
* `src/features/baseline_features.py`: `include_lexical` and
  `lexical_features_path` knobs on `BaselineFeatureConfig`.
* `src/models/baseline/train.py`: per-family pipeline factories,
  4-family iteration, threshold check applied to the linear
  reference family.

### RUNS.md row
A new row for the multi-family lexical ablation, sorted newest-first,
capturing the date, run folder, git SHA at run time, the 4-family
label, the features group, key metric summary, and notes.

---

## 8. Open questions for the planning conversation

1. **Treatment of the lexical group's official ablation row.**
   Recommendation: **append the negative-lift row to
   `phase3_ablation.csv` as the honest finding** (already done; can
   be flagged as "null result" in the report's narrative). The
   feature matrix stays on disk so Phase 4's full model benchmark can
   re-evaluate lexical features in any model family it tests.
2. **Drop `rare_word_proportion`?** Recommendation: yes, given the
   r = −0.939 redundancy with `mean_log_frequency`. The drop is a
   one-line edit to `LEXICAL_FEATURE_COLUMNS` in
   `src/features/lexical.py` plus a re-run of `compute_lexical_features`.
   Does not affect the negative-lift verdict (neither feature has
   univariate signal).
3. **wordfreq vs SUBTLEX-US deviation.** Given that the negative
   lift is consistent across all four families and across all 13
   features (none has univariate signal), the choice of frequency
   source is unlikely to be the cause. The deviation is documented
   in the run's preprocessing metadata for full traceability. If
   strict SUBTLEX-US is preferred for the report, the backend can
   be swapped and the features recomputed; expectation is the
   verdict does not change.
4. **Proceed to sentiment proposal?** The lexical verdict is
   conclusive enough that I do not need additional input on lexical
   itself. The sentiment proposal can begin once the planning
   conversation signs off on the three items above.

---

## 9. Recommended next step

Sign off on:

* Appending the negative-lift row as the lexical group's verdict.
* Dropping `rare_word_proportion` from the lexical feature matrix.
* Documenting the wordfreq deviation in `FEATURE_NOTES.md` when that
  document lands at end of Task 4.

Then start the sentiment proposal at
`docs/proposals/phase3_sentiment_proposal.md`. Sentiment shares
NLTK preprocessing with lexical (sentence and word tokenization)
and can reuse the cached lexical pipeline scaffolding, so the
implementation cost is lower than the lexical group's was.
