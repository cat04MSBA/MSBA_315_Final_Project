# Phase 4 Pre-Registration: Layer 1 Core Prediction Model

**Phase:** Phase 4 (Layer 1 of the four-layer triage system)
**Status:** Pre-registered; locked before benchmark execution
**Date:** 2026-05-03

This document fixes the Phase 4 candidate model roster, hyperparameter
grids, cross-validation scheme, statistical comparison procedure, and
escalation criteria **before any benchmark run touches the training
data**. Locking these choices is the multiple-comparisons firewall the
project's pre-registration discipline relies on, the same discipline
that governed every Phase 3b standalone group and the Phase 3c
combinations sub-phase.

The Phase 4 brief from the planning conversation (2026-05-03) and the
Phase 3 final summary together establish the strategic context. The
choices below are the executing chat's tactical implementation of that
strategy, with two deliberate deviations from the brief noted in
Section 13.

---

## 1. Purpose and scope

**In scope.** Train candidate model families on the Phase 3 feature
matrix to identify the strongest feed for Layers 2 through 4
(calibration, asymmetric-cost decision, scene-level SHAP). The
exercise produces a benchmark across families on three targets, a
paired statistical comparison across families on the primary target,
and a saved trained primary model artifact per target.

**Out of scope.** Calibration (Phase 5), cost-matrix construction
(Phase 6), SHAP explanations (Phase 7), test-set evaluation (Phase 8),
ensemble construction beyond what falls naturally out of model-family
comparison, and DistilBERT or any transformer fine-tune (deferred to
Phase 8 robustness exploration if the Phase 4 numbers fall below the
forward-expected band).

**The methodology novelty contribution.** The project's contribution
is the four-layer architecture, not single-model SOTA on box-office
prediction. Phase 4 is framed as model selection for the downstream
layers, not as a SOTA chase against the dialogue-prediction
literature.

---

## 2. Strategic decisions inherited (locked, not for re-litigation)

These are settled before Phase 4 begins. Sources are recorded in
`docs/PROJECT_CONTEXT.md` Section 8 and the Phase 4 brief.

1. **Three targets in parallel.** `log_roi` (regression), `roi_gt_1`
   and `roi_gt_2` (classification, threshold-consistent with the
   regression target). All three are reported through Phase 4. The
   formal primary-outcome decision is made at end of phase based on
   full Phase 4 evidence; the headline informally targets `roi_gt_2`
   per Phase 3c evidence.
2. **Train / calibration / test split is fixed.** 1,199 / 257 / 257
   films, stratified by `(primary_genre_bucketed, decade_bucket)`,
   seed 42. Read split membership from
   `data/processed/split_assignments.parquet` or the `split` column
   in `features.parquet`. The calibration set is reserved for
   Phase 5; the test set is opened once in Phase 8.
3. **Metric vocabulary.** Regression: MSE, RMSE, MAE, CVRMSE.
   Classification: AUC-ROC, PR-AUC, F1 (at 0.5 threshold), log-loss.
   Both in-sample (full-train fit) and out-of-fold values reported
   per metric. R-squared is not in the reported set.
4. **`save_run` per-run logging is mandatory.** Every model-fit batch
   produces a `runs/phase_4/<timestamp>_<name>/` directory with the
   five canonical files (`params.json`, `preprocessing_summary.json`,
   `features_used.json`, `metrics.json`, `run.log`) plus a
   `model.joblib` for runs that produce a single canonical artifact.
   The `RUNS.md` index gets a new row per run.

---

## 3. Model family roster (locked)

Six families, organized in two tiers. Tier reflects compute budget
allocation and reporting prominence, not methodological seriousness.

### 3.1 Primary tier (full hyperparameter search, headline benchmark)

| Family | Implementation | Role |
|---|---|---|
| Linear (Ridge / Logistic-L2) | `RidgeCV`, `LogisticRegression` with C grid via `GridSearchCV` | Linear baseline; constant comparator anchored from Phase 3 |
| HistGradientBoosting | `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` | Tree boosting with Phase 3 longitudinal continuity (the train-OOF gap finding is HistGB-specific) |
| Random Forest | `RandomForestRegressor`, `RandomForestClassifier` | Bagging ensemble; different inductive bias from gradient boosting; Phase 3 did not include it |
| SVM-RBF | `SVR`, `SVC(kernel="rbf")` | Phase 3c promotion; the largest single classification lift in Phase 3 |

LightGBM and XGBoost are explicitly **not** in the primary tier. The
brief considered both as alternatives to HistGB. The decision is to
keep HistGB for two reasons: (a) the Phase 3 train-OOF gap finding is
HistGB-specific and switching frameworks would lose the longitudinal
"did stronger regularization close the gap?" comparison; (b) on
n=1,199 the regularization-formulation differences between HistGB,
LightGBM, and XGBoost are second-order and the marginal information
from running multiple gradient-boosting frameworks is negligible.
Random Forest takes the slot the brief originally allocated to
XGBoost because RF's bagging-with-deep-trees inductive bias is
genuinely different from any boosting variant and Phase 3 did not
benchmark it.

### 3.2 Secondary tier (smaller search, included for breadth and feature-attribution narrative)

| Family | Implementation | Role |
|---|---|---|
| Lasso / L1-Logistic | `LassoCV`, `LogisticRegression(penalty="l1")` with C grid | Feature-selection lens; non-zero coefficients feed the Phase 4 attribution narrative without yet invoking SHAP |
| Linear-kernel SVM | `LinearSVR`, `LinearSVC` with C grid | Sanity check distinguishing "SVM benefits from kernel methods generally" from "the RBF kernel specifically aligns with the feature structure" |

The secondary tier produces benchmark rows but the paired statistical
comparison in Section 9 is run only on the primary tier to keep the
multiple-comparisons surface bounded.

### 3.3 Explicitly dropped from Phase 4

- **DistilBERT or any transformer fine-tune.** Per brief Section 2.
  Fine-tuning on n=1,199 train films invites overfit, the four-layer
  architecture is the project's novelty (not single-model SOTA), and
  Phase 4 already has six families to benchmark. If the Phase 4
  numbers land below the forward-expected band on `roi_gt_2`,
  DistilBERT becomes a Phase 8 robustness question.
- **KNN.** Phase 3 established KNN as a weak baseline on this corpus
  at the high feature-to-sample ratio; carrying it into Phase 4 adds
  no information.
- **Ensemble methods (stacking, weighted averaging) beyond what falls
  naturally out of family comparison.** Per brief; ensemble work is a
  Phase 4 end-of-phase question if individual-model performance
  warrants it.

---

## 4. Hyperparameter grids per family

All grids below are searched via `GridSearchCV` with 5-fold
stratified CV inside each family's hyperparameter selection. The
chosen best estimator is then re-fit and evaluated under the
repeated CV scheme of Section 5.

### 4.1 Primary tier

**Linear (Ridge regression).**
- `alpha`: `numpy.logspace(-3, 3, 13)` (13 cells).

**Linear (Logistic-L2).**
- `C`: `numpy.logspace(-3, 3, 13)` (13 cells).
- `class_weight`: `"balanced"` (Section 7).

**HistGradientBoosting (per brief Section 3, the conservative grid).**
- `max_depth`: `{2, 3, 4}` (3 cells).
- `learning_rate`: `{0.01, 0.02, 0.05}` (3 cells).
- `min_samples_leaf`: `{10, 20, 40, 80}` (4 cells).
- `max_iter`: 200 (capped, with `early_stopping=True` and validation-fraction-based early stopping).
- Total grid: 36 cells per (matrix, target).

**Random Forest.**
- `n_estimators`: `{200, 500}` (2 cells).
- `max_depth`: `{None, 6, 12}` (3 cells).
- `min_samples_leaf`: `{1, 5, 20}` (3 cells).
- `class_weight`: `"balanced"` for classification (Section 7).
- Total grid: 18 cells per (matrix, target).

**SVM-RBF (per brief Section 4).**
- `C`: `{0.1, 0.3, 1, 3, 10, 30}` (6 cells).
- `gamma`: `{"scale", 0.001, 0.003, 0.01, 0.03, 0.1}` (6 cells).
- `class_weight`: `"balanced"` for `SVC` (Section 7).
- Total grid: 36 cells per (matrix, target).

### 4.2 Secondary tier

**Lasso (regression) / L1-Logistic (classification).**
- Regression: `alpha`: `numpy.logspace(-3, 1, 9)` (9 cells).
- Classification: `C`: `numpy.logspace(-2, 2, 9)` (9 cells), with `solver="liblinear"` for L1 support.

**Linear-kernel SVM.**
- `C`: `{0.01, 0.1, 1, 10}` (4 cells).
- `class_weight`: `"balanced"` for `LinearSVC` (Section 7).

---

## 5. Cross-validation scheme

**Repeated stratified 5-fold cross-validation with 3 repetitions**
(`RepeatedStratifiedKFold` for classification, `RepeatedKFold` for
regression, both seeded at 42). This produces 15 fold-level
observations per (family, matrix, target), which is the number the
paired statistical test of Section 9 operates on.

**Why 3x5 repeated rather than plain 5-fold.** The Bayesian
correlated-t-test in Section 9 is sensitive to the degrees of freedom
of the per-fold difference series. With plain 5-fold the test has
4 degrees of freedom, which is genuinely thin and yields wide
posteriors that under-detect real differences. Repeated 5-fold with
3 repetitions yields 14 effective degrees of freedom for negligible
methodological complexity and roughly 3x compute for the outer
evaluation loop. The hyperparameter search inner loop remains plain
5-fold (the inner loop's role is selection, not statistical
inference).

**Stratification.** Classification folds stratify on the relevant
binary target. Regression folds stratify on `decade_bucket` to
control era variance, matching the Phase 3 split definition's
stratification logic.

---

## 6. Hyperparameter search procedure

For each (family, matrix, target):

1. Fit `GridSearchCV` (5-fold inner stratified CV) on the full train
   split to identify the hyperparameter cell that maximizes the
   primary metric (`AUC-ROC` for classification, negative `RMSE` for
   regression).
2. The best estimator is re-fit across the 15 outer folds of the
   repeated CV scheme of Section 5. The 15 OOF prediction series
   feed both the headline OOF metrics (Section 8) and the paired
   statistical comparison (Section 9).
3. The best hyperparameter cell is also re-fit on the full train
   split to produce the in-sample (train-eval-set) metrics and to
   produce the saved `model.joblib` artifact for the winner per
   target (Section 11).

This procedure is a single-level outer evaluation with hyperparameter
selection nested inside, not a full nested CV. Full nested CV (15
outer x 5 inner x N hyperparameter cells) would multiply compute by
roughly 5x without changing the ranking of families on this corpus
size; the single-level outer evaluation matches the Phase 3 trainer's
established pattern.

**Reproducibility.** All random states are seeded at 42. The
hyperparameter search is deterministic for a fixed input matrix.

---

## 7. Class-weighting policy (locked)

`roi_gt_1` is approximately 80% positive; `roi_gt_2` is approximately
64% positive. The classification base rates are imbalanced enough
that the choice between unweighted and weighted training materially
affects PR-AUC and F1 reporting.

**Decision:** all classifiers use `class_weight="balanced"` or its
equivalent. Specifically:

- `LogisticRegression`, `RandomForestClassifier`, `LinearSVC`, `SVC`:
  `class_weight="balanced"` directly.
- `HistGradientBoostingClassifier`: `sample_weight` derived from
  inverse class frequencies, passed through `GridSearchCV.fit`.
  `HistGradientBoostingClassifier` does not accept `class_weight`.

**Reporting consequence.** PR-AUC and F1 are reported as the
imbalanced-aware metrics; AUC-ROC is the threshold-free headline.
The choice is pre-registered as the apples-to-apples baseline so the
family comparison is not confounded by per-family default weighting.

The unweighted alternative is not run. If the report wants to show
both, that is a Phase 8 robustness-analysis addition.

---

## 8. Input matrix policy (locked, brief Section 1)

Run both:

- **`all_five`**: the full `features.parquet` (127 features). The
  maximum-information matrix; Phase 3c evidence shows SVM-RBF
  extracts substantial signal from it.
- **`standalone_positive_union`**: structural baseline + topic +
  character network + embedding (approximately 92 features). Drops
  the lexical and sentiment groups whose Phase 3b standalone
  ablations landed null. Honors the pre-registration discipline.

Both matrices are evaluated under the full primary-tier roster on
all three targets. The Phase 4 summary reports them side-by-side.
The final saved `model.joblib` per target uses the matrix that wins
on the primary metric for that target. If results are statistically
indistinguishable across matrices for a given target (per Section 9),
the parsimony tie-breaker selects `standalone_positive_union`.

The secondary tier runs on `all_five` only to keep the compute
budget bounded.

---

## 9. Statistical comparison: Bayesian correlated-t-test

Pairwise comparisons across primary-tier families are computed via
the **Bayesian correlated-t-test** (Benavoli, Corani, Demsar, Zaffalon
2017) on the per-fold OOF metric series. The test accounts for the
within-fold correlation across models that naive paired t-tests
ignore, and it produces a posterior probability that one model is
better than another by more than a region of practical equivalence
(ROPE) threshold rather than a null-hypothesis-significance p-value.

**Implementation.** The `baycomp` Python package's
`two_on_single` function (correlated-t with rho fixed at 1/k where
k=5 is the number of folds, rho=0.2 for k=5).

**ROPE values.** Pre-registered per metric:

- AUC-ROC: ROPE = 0.005 (half-width). Differences smaller than 0.005
  AUC are practically equivalent.
- PR-AUC: ROPE = 0.01 (half-width).
- F1: ROPE = 0.01 (half-width).
- RMSE / MAE / CVRMSE (regression): ROPE = 0.01 (half-width on the
  relevant target's scale; CVRMSE is dimensionless).

**Output.** For each (matrix, target) pair the comparison table
reports, for each family pair (A, B), three posterior probabilities
that sum to 1: P(A better), P(equivalent), P(B better). A family is
declared the winner of a pair if P(A better) >= 0.95.

**Comparisons reported.** All pairwise within the primary tier
(C(4, 2) = 6 pairs per matrix per target = 36 comparisons total
across both matrices and three targets). The Bayesian framework does
not require multiple-comparisons correction in the frequentist
sense, but the volume is documented so the reader knows the
inferential surface.

---

## 10. Pre-registered escalation criteria

Each criterion below triggers a planning-conversation escalation (no
unilateral resolution by the executing chat). The Phase 4 mandatory
end-of-phase escalation per `PROJECT_CONTEXT.md` Section 11 is
separate from these intra-phase triggers.

1. **Headline target ceiling (corpus-size signal).** If no primary-
   tier model on either matrix exceeds OOF AUC-ROC = 0.69 on
   `roi_gt_2` (the mid-band of the Phase 3 forward-expected 0.65 to
   0.72), surface the result as a corpus-ceiling escalation rather
   than treating it as the final model. Possible responses include
   re-examining the input matrix question, considering whether
   `roi_gt_2` is the right primary target, or accepting a
   mid-band-floor model with documented limitation.
2. **Universal train-OOF gap (corpus-size signal).** If the train-
   versus-OOF gap on AUC-ROC exceeds 0.10 across all four primary-
   tier families on the headline target (not just HistGB), surface
   as a corpus-size signal before treating it as a per-family
   problem. The current Phase 3 evidence is that HistGB has a 0.20
   to 0.27 gap; linear has a much smaller gap. A universal gap
   suggests n=1,199 is the binding constraint.
3. **Statistical tie at the top.** If two or more primary-tier
   families produce posterior P(A better) < 0.7 against each other
   on the headline target (i.e., Bayesian-equivalent within the
   chosen ROPE), surface for planning-conversation tie-breaker. The
   tie-breaker dimensions are interpretability and Phase 5
   calibration friendliness (linear and tree models calibrate
   naturally; SVM-RBF needs Platt scaling).
4. **Linear-kernel SVM matches RBF.** If the secondary-tier linear-
   kernel SVM produces statistically indistinguishable performance
   from primary-tier SVM-RBF on the headline target, surface the
   finding. The methodology story changes from "the RBF kernel
   aligns with the feature structure" to "SVM benefits from kernel
   methods generally", which has implications for the Phase 5 / 6
   discussion of model interpretability.

---

## 11. Deliverables

**Tables (`reports/tables/`).**
- `phase4_benchmark.csv`: one row per (matrix, family, target,
  eval_set, metric). Includes selected hyperparameter cell, value,
  bootstrap CI bounds, fold-level mean and standard deviation.
- `phase4_paired_tests.csv`: one row per (matrix, target, family_A,
  family_B, metric) with the three Bayesian posteriors and the
  declared winner.

**Figures (`reports/figures/`).**
- `phase4_train_oof_gap.png`: train-versus-OOF gap per family per
  target, side-by-side bar chart for both input matrices. The visual
  diagnostic that surfaces over- and underfit at a glance.
- `phase4_calibration_pre.png`: OOF reliability diagrams for the
  primary-tier classifiers on `roi_gt_2`. Bridges to Phase 5: tells
  Phase 5 in advance how badly each candidate needs calibration.

**Model artifacts (`data/processed/`).**
- `phase4_primary_model_log_roi.joblib`
- `phase4_primary_model_roi_gt_1.joblib`
- `phase4_primary_model_roi_gt_2.joblib`

  Each is the winning family for that target, fit on the full train
  split with the best hyperparameters from CV, on the matrix that
  won. Saved for Phase 5 (calibration) and Phase 8 (test
  evaluation).

**Documentation.**
- `docs/summaries/phase_4_summary.md`: the canonical Phase 4
  summary using the Section 7 template from
  `CLAUDE_CODE_GUIDELINES.md`.
- `docs/PROJECT_CONTEXT.md` Sections 8 and 9 updated.
- `runs/phase_4/<timestamp>_<name>/` per training-run batch with
  the five canonical files plus the per-target `model.joblib`.

**Notebook.**
- `notebooks/_build_phase_4_notebook.py` and
  `notebooks/phase_4.ipynb` for the report deliverable.

---

## 12. Compute budget and reduction levers

**Estimated upper bound** (per matrix):

| Family | Hyperparameter cells | Inner CV fits | Outer CV fits (3x5) | Total fits |
|---|---:|---:|---:|---:|
| Linear (Ridge/Logistic) | 13 | 65 | 15 | 80 |
| HistGB | 36 | 180 | 15 | 195 |
| Random Forest | 18 | 90 | 15 | 105 |
| SVM-RBF | 36 | 180 | 15 | 195 |
| Lasso/L1-Logistic | 9 | 45 | 15 | 60 |
| Linear-SVM | 4 | 20 | 15 | 35 |

Times 3 targets times 2 matrices for the primary tier (the
secondary tier runs on `all_five` only) yields approximately
4,200 individual model fits.

**Wall-clock estimate.** Linear and tree models fit in seconds.
SVM-RBF fits scale O(n^2) to O(n^3); at n_train ~ 960 per fold the
larger C/gamma corners can take 5 to 30 seconds per fit. Realistic
total wall-clock with `n_jobs=-1` on a modern laptop: 3 to 8 hours
per matrix for the primary tier, 30 to 60 minutes for the secondary
tier.

**Reduction levers**, applied in this order if the budget is binding:

1. Drop the secondary tier (Lasso, Linear-SVM) entirely.
2. Reduce SVM-RBF gamma grid from 6 cells to 3 (`{"scale", 0.01, 0.1}`).
3. Drop one input matrix (default to `all_five` per Phase 3c
   evidence).
4. Reduce CV repetition count from 3 to 2 (10 folds total). Below
   2 repetitions the paired Bayesian test loses too many degrees of
   freedom; do not reduce below.

CV fold count below 5 is not a reduction lever. The paired test
operates on per-fold differences and 5 folds is the floor.

---

## 13. Deviations from the brief

Two deliberate deviations, both raised in the planning conversation
on 2026-05-03 and approved.

1. **HistGB chosen over LightGBM as the gradient-boosting
   representative.** The brief allowed either; the executing chat
   recommended HistGB for Phase 3 longitudinal continuity on the
   train-OOF gap finding. Approved.
2. **Random Forest in the primary tier in place of XGBoost.** The
   brief listed XGBoost as a primary candidate; the executing chat
   recommended Random Forest because RF's bagging-with-deep-trees
   inductive bias is genuinely different from any boosting variant,
   while XGBoost vs HistGB vs LightGBM differ only in regularization
   formulation (second-order on n=1,199). Approved.
3. **CV scheme bumped from plain 5-fold to repeated 5-fold (3
   repetitions)** to give the Bayesian correlated-t-test sufficient
   degrees of freedom. Approved.

All three deviations are logged here for the audit trail and will
appear in `docs/PROJECT_CONTEXT.md` Section 8 with the Phase 4
summary.

---

## 14. What is locked and what is not

**Locked.** Roster (Section 3), grids (Section 4), CV scheme
(Section 5), search procedure (Section 6), class-weighting policy
(Section 7), input matrix policy (Section 8), statistical
comparison procedure (Section 9), escalation criteria (Section 10),
deliverables (Section 11).

**Not locked (tactical, executing chat decides during
implementation).** Library version pinning, helper-function
internal structure, plot styling, table column ordering,
intermediate artifact paths, log-message verbosity. These are
tactical implementation details that do not affect the benchmark
ranking.

**Pre-registration discipline.** The set above is not expanded
after seeing results. If implementation surfaces a methodology
issue (e.g., a numerical instability in `baycomp`, an unforeseen
class-imbalance interaction with HistGB's early stopping), surface
it as an issue and resolve in dialogue with the planning
conversation, not by silently revising the pre-registration.
