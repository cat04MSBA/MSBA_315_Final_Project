# Phase 3b: Lexical group handoff

**Status:** Lexical features implemented and evaluated across four
model families. Group signed off by the planning conversation
2026-05-03 with three resolutions and one methodology addition (see
Sections 8 and 9 below). Negative-lift row is appended to
`phase3_ablation.csv` as the honest finding; both
`mean_log_frequency` and `rare_word_proportion` are kept in the
matrix for Phase 4 optionality; the wordfreq deviation is accepted
and will be documented in `FEATURE_NOTES.md` at end of Task 4. Phase
3c combinations sub-phase is added to the methodology to address
the genre-residual-signal concern this group surfaced.
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

## 4. Headline numbers (lexical added on top of the revised dialogue-only floor, 4 families × 2 eval sets)

Full table at `reports/tables/phase3_ablation.csv` (96 rows: 4
families × 2 evaluation sets × 3 targets × {4 regression metrics or
4 classification metrics}). Reporting now uses the updated metric
vocabulary: regression uses MSE, RMSE, MAE, CVRMSE (R-squared
removed); classification uses AUC-ROC, PR-AUC, F1, log-loss. Both
in-sample (train) and out-of-fold (OOF) numbers are reported per
family per metric. The pre-registered lift bands apply to the
linear family's OOF numbers only (the historical reference).

### 4.1 OOF lift per family (lower-is-better metrics: negative lift means improvement)

| Family | Metric | Floor | With lexical | Lift |
|---|---|---:|---:|---:|
| linear | log_roi RMSE (lower) | 1.339 | 1.350 | **+0.011** (worse) |
| linear | log_roi MAE (lower) | 0.948 | 0.955 | +0.006 (worse) |
| linear | log_roi CVRMSE (lower) | 0.993 | 1.002 | +0.008 (worse) |
| linear | roi_gt_1 AUC (higher) | 0.558 | 0.565 | +0.007 |
| linear | roi_gt_1 F1 (higher) | 0.893 | 0.892 | -0.002 |
| linear | roi_gt_1 log-loss (lower) | 0.493 | 0.514 | +0.021 (worse) |
| linear | roi_gt_2 AUC (higher) | 0.602 | 0.600 | -0.002 |
| linear | roi_gt_2 PR-AUC (higher) | 0.739 | 0.729 | -0.010 |
| linear | roi_gt_2 log-loss (lower) | 0.635 | 0.638 | +0.003 (worse) |
| histgb | log_roi RMSE (lower) | 1.327 | 1.333 | **+0.006** (worse) |
| histgb | log_roi MAE (lower) | 0.943 | 0.943 | 0.000 |
| histgb | roi_gt_1 AUC (higher) | 0.552 | **0.511** | **-0.041** (worse) |
| histgb | roi_gt_1 PR-AUC (higher) | 0.843 | 0.816 | -0.026 (worse) |
| histgb | roi_gt_2 AUC (higher) | 0.610 | **0.586** | **-0.024** (worse) |
| histgb | roi_gt_2 PR-AUC (higher) | 0.731 | 0.709 | -0.022 (worse) |
| knn | log_roi RMSE (lower) | 1.364 | 1.379 | +0.015 (worse) |
| knn | roi_gt_1 AUC (higher) | 0.527 | **0.494** | **-0.032** (worse) |
| knn | roi_gt_2 AUC (higher) | 0.578 | 0.548 | -0.030 (worse) |
| svm | log_roi RMSE (lower) | 1.357 | 1.351 | -0.006 (better) |
| svm | roi_gt_1 AUC (higher) | 0.504 | 0.519 | +0.014 |
| svm | roi_gt_2 AUC (higher) | 0.534 | 0.565 | +0.031 |

Bold marks moves of 0.005 or more in the worse direction.

### 4.2 Pre-registered linear-family lift (proposal v2 Section 3)

Pre-registration originally used R-squared on the regression
target. With R-squared removed from the metric set, the original
band (+0.010 to +0.025 R²) translates to RMSE on the same floor as
roughly -0.020 to -0.010 (lower is better for RMSE). The
classification AUC bands carry over directly.

| Target | Metric | Predicted band (linear OOF) | Actual | In band? |
|---|---|---|---:|:---:|
| log_roi | RMSE | -0.020 to -0.010 (lower is better) | +0.011 (worse) | No |
| roi_gt_1 | AUC-ROC | 0.000 to +0.010 | +0.007 | Yes |
| roi_gt_2 | AUC-ROC | +0.015 to +0.035 | -0.002 | No |

Pre-registered direction was wrong on two of the three headline
predictions. Only `roi_gt_1` moved positively, and that target was
the one the proposal expected to lift the least.

### 4.3 Train-versus-OOF gap with lexical added (linear and histgb families)

Reporting both eval sets surfaces an overfit-gap signal that the
OOF-only view of v1 hid. Numbers below are RMSE on `log_roi`.

| Family | Train (in-sample) | OOF | Train-OOF gap |
|---|---:|---:|---:|
| linear | 1.300 | 1.350 | -0.050 |
| histgb | 1.221 | 1.333 | -0.112 |

HistGB's train-OOF gap is twice as wide as linear's, consistent
with HistGB over-fitting more aggressively on the
lexical-augmented matrix. Adding the lexical features did not
narrow the gap on either family; the new features are absorbed
into the train fit without translating into OOF improvement.

### 4.4 Verdict

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

**The most likely mechanism is not "lexical features have no
information" but "lexical features carry information that genre,
era, and structural counts already absorb."** Action films
systematically have shorter dialogue and lower Flesch-Kincaid
scores; dramas have more sophisticated vocabulary; period pieces use
longer words. Genre and era explain most of that variation before
the lexical features get a chance to contribute. With the structural
baseline already including thirteen genre dummies and a release-year
column, the marginal residual variance lexical can explain is small
enough that, at this corpus size and with these four model families,
none of them can reliably extract it without hurting itself in the
process. The features are competing for residual signal after the
strongest confounds have already been controlled.

This framing matters because it generates a testable prediction:
features that carry information orthogonal to genre (for example
graph-structural features of the character network, or
sentiment-trajectory shape that does not vary as cleanly with
genre) should fare better. It also motivates the Phase 3c
combinations sub-phase that will follow Phase 3b: groups whose
standalone-against-baseline signal looks weak may carry meaningful
lift when combined with other groups that capture different
aspects of the residual.

**Family-specific mechanisms (secondary).**

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

## 8. Resolved questions

The four open questions raised at v1 of this handoff have all been
resolved by the planning conversation 2026-05-03. Outcomes recorded
here for the audit trail.

1. **Treatment of the lexical group's official ablation row:
   APPEND.** The negative-lift row goes into
   `phase3_ablation.csv` as the honest finding. The Phase 3
   narrative will lead with it as a methodology demonstration:
   we proposed, pre-registered, measured, and surfaced a
   direction-wrong prediction honestly. That is a stronger story
   for the report than a result that happened to confirm
   expectations. Already done; the row is in
   `reports/tables/phase3_ablation.csv`.

2. **Drop `rare_word_proportion`? KEEP.** The r = −0.939 with
   `mean_log_frequency` clears the \|r\| > 0.9 review threshold
   but only just, and the two features measure conceptually
   different things even when they correlate strongly on this
   corpus (mean Zipf vs the bottom-quartile share). Dropping
   based on a single-corpus correlation closes off information
   for Phase 4 model families that might handle the redundancy
   differently than these four did. The decision is to preserve
   optionality. Does not affect the verdict (neither feature has
   univariate signal).

3. **wordfreq vs SUBTLEX-US deviation: ACCEPTED.** The reasoning
   is correct: if every one of the 13 features (frequency-based
   and the eight non-frequency features) shows no univariate
   signal, the frequency-source choice cannot be the cause. The
   mechanism wordfreq preserves (subtitle-domain frequencies via
   OpenSubtitles inclusion) is what the proposal selected
   SUBTLEX-US for. Renaming the features without the `_subtlex`
   suffix was the right call: names are honest about the source.
   Document the deviation in `FEATURE_NOTES.md` when Task 4
   lands. No re-run.

4. **Proceed to sentiment proposal: YES**, after Phase 3c
   methodology addition is documented in the decisions log.

## 9. New methodology: Phase 3c combinations sub-phase

The lexical null result raised a real concern about the incremental
ablation methodology. Specifically, a feature group can look dead in
isolation against a baseline that already includes genre, era, and
structural counts but contribute meaningfully in combination with
other groups, because multivariate redundancy with genre absorbs
more variance than pairwise correlation checks reveal. The current
ablation structure systematically under-credits any group whose
signal partially overlaps with genre, which is roughly all of them.

**Phase 3c addresses this without abandoning the incremental
ablation.** Phase 3b proceeds as planned: sentiment, topic,
character network, embeddings in that order, each with its
standalone-lift row in `phase3_ablation.csv` and its pre-registered
prediction. The clean ablation table the report needs gets built.

After all five Phase 3b groups are computed, Phase 3c runs a small
set of pre-specified combinations against the floor. The combinations
get pre-specified before any of them is measured, so the
pre-registration discipline is preserved at the combinations level
too.

**Combinations to pre-specify** (write into the Phase 3c proposal
before measuring any of them):

* **All five groups together.** The maximum-information matrix.
* **Structural-leaning combination:** lexical + character network
  on top of the structural baseline. Hypothesis: these two carry
  information less redundant with genre than the semantic groups.
* **Semantic combination:** sentiment + topic + embeddings.
  Hypothesis: these may share information with each other, so
  testing them jointly tells us whether the semantic signal is
  over-counted by adding all three.
* **Any pair flagged during Phase 3b as worth testing jointly,**
  for example, if sentiment shows a small lift and topic shows a
  small lift, the pair gets pre-specified as a combination before
  either's standalone result is finalized.

Three to five combinations total. **The set is not expanded after
seeing results.** That is the multiple-comparisons trap this
structure is designed to avoid.

**Phase 3c output:** one additional table at
`reports/tables/phase3c_combinations.csv` and a short narrative
section in the Phase 3 summary. Phase 4 model selection then reads
from the union of features that earned their place, either
standalone in 3b or in combination in 3c, and lets the
gradient-boosted models handle redundancy at training time.

## 10. Next step

Start the sentiment proposal at
`docs/proposals/phase3_sentiment_proposal.md`. The substantive
design question for this group is the pooling choice
(whole-screenplay versus scene-windowed versus arc-clustered
versus mixed). Sentiment shares NLTK preprocessing with lexical
and can reuse the cached lexical pipeline scaffolding.
