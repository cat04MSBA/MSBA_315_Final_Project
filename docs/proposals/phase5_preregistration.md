# Phase 5 Pre-Registration: Layer 2 Calibrated Uncertainty

**Phase:** Phase 5 (Layer 2 of the four-layer triage system)
**Status:** Pre-registered; locked before any calibration fits
**Date:** 2026-05-04

This document fixes the Phase 5 calibration methodology, conformal-
prediction approach, evaluation scheme, and escalation criteria
**before any calibration fit touches the calibration set**. The same
discipline that governed Phase 3 standalone groups, the Phase 3c
combinations sub-phase, and the Phase 4 model selection.

The Phase 4 close (2026-05-04 02:00) and the Phase 4 follow-up
(2026-05-04 09:25) together establish the Phase 5 inputs. The
choices below are the executing chat's tactical implementation of
the roadmap's Phase 5 specification with two deliberate decisions
called out in Section 13.

---

## 1. Purpose and scope

**In scope.** Wrap the Phase 4 per-target winners with two
complementary calibration techniques:

* **Probability calibration** (Platt or isotonic scaling) so that
  the classifier's reported `P(roi_gt_2 | features) = 0.7` actually
  corresponds to a 70% empirical positive rate on held-out films.
* **Conformal prediction** (split conformal via the `mapie`
  library) so that a 90%-confidence prediction set / interval
  empirically covers the true label / value 90% of the time.

The two together feed Phase 6's asymmetric-cost decision rule:
the calibrated probability drives the expected-cost calculation;
the conformal set / interval width drives the "Refer to human
reader" trigger when the model is uncertain.

**Out of scope.** Cost-matrix construction (Phase 6), scene-level
SHAP explanations (Phase 7), test-set evaluation (Phase 8). No
re-training of the Phase 4 base estimators in this phase; the
saved winners are calibrated as-is. Re-Phase-4 with the v2
enriched corpus is a parallel workstream and is not part of
Phase 5; if v2 lifts the headline target meaningfully, Phase 5
re-runs on the v2 winners using the same locked methodology
below.

---

## 2. Strategic decisions inherited (locked, not for re-litigation)

These are settled before Phase 5 begins.

* **Phase 4 winners are the calibration targets.** Per
  ``data/processed/phase4_primary_model_*.joblib``:
  * `log_roi`: RandomForest on `standalone_positive_union_mpnet`
    (RMSE 1.3102 OOF)
  * `roi_gt_1`: SVM-RBF on `standalone_positive_union_mpnet`
    (AUC 0.6353 OOF)
  * `roi_gt_2`: XGBoost on `standalone_positive_union_mpnet`
    (AUC 0.6520 OOF; the headline target)
* **Calibration set is the 257-film cal split** carved in Phase 3
  (`data/processed/split_assignments.parquet`, `split == "cal"`).
  This split has been untouched since Phase 3 carve and is the
  honest conformity surface for Phase 5.
* **Test set untouched.** The 257-film test split is reserved
  for Phase 8 only. No Phase 5 evaluation uses it. Per
  `PROJECT_CONTEXT.md` Section 6 and Section 11.
* **Bayesian correlated-t-test framework** carries forward from
  Phase 4 for any Phase 5 model-vs-model comparison (e.g.,
  Platt vs isotonic) on the held-out portion of the calibration
  set.
* **save_run discipline mandatory** for every calibration run.
  Each per-target calibration produces a ``runs/phase_5/`` directory
  with the canonical five files plus a `calibrated_model.joblib`.
* **Pre-registration discipline carries forward.** The methodology,
  metrics, escalation criteria locked here are not revised after
  seeing results.

---

## 3. Conformal prediction method per task (locked)

### 3.1 Classification targets (`roi_gt_1`, `roi_gt_2`)

**Method:** `mapie.classification.SplitConformalClassifier` with
`prefit=True` (the Phase 4 winner is already fit on the train
split). The conformity score `"lac"` (least ambiguous classifier;
the standard split-conformal score for classification) is the
default; we also evaluate `"aps"` (adaptive prediction sets) for
comparison since `"aps"` produces more informative sets when the
underlying probabilities are miscalibrated.

**Output:** for each test film, a prediction set ⊆ {0, 1} at each
chosen confidence level. The set can be {} (rare; only if both
classes have low conformity), {0}, {1}, or {0, 1}. The set
{0, 1} is the "Refer to human reader" indicator: the model cannot
distinguish between flop and hit at the chosen confidence level.

### 3.2 Regression target (`log_roi`)

**Method:** `mapie.regression.SplitConformalRegressor` with
`prefit=True`. The default conformity score `"absolute"` (residual
magnitude) gives marginal coverage; we also evaluate
`"gamma_conformity_score"` (residual / abs(prediction)) which gives
intervals with width that scales with the predicted magnitude.

**Output:** for each test film, a prediction interval [lower,
upper] at each chosen confidence level. Wider interval means more
uncertainty.

---

## 4. Probability calibration method per family (locked)

The classifier's raw probability output may be miscalibrated
(Phase 4's `phase4_calibration_pre.png` figure shows all four
primary families above the diagonal — systematic under-confidence
on roi_gt_2). Probability calibration corrects this so that
downstream phases get well-calibrated `P(positive | features)`.

| Family | Calibration method | Why |
|---|---|---|
| **xgboost** (winner on `roi_gt_2`) | Platt scaling (`CalibratedClassifierCV(method="sigmoid", cv="prefit")`) | XGBoost emits probabilities but they are biased; Platt scaling on a held-out fold is the standard fix. |
| **svm_rbf** (winner on `roi_gt_1`) | Platt scaling | SVC's `decision_function` outputs are not probabilities at all; Platt is the canonical conversion. |
| **random_forest** (regression — log_roi) | None | Regression target; calibration is via conformal intervals only. |

**Comparison run:** for both classification winners we also fit
`CalibratedClassifierCV(method="isotonic", cv="prefit")` and report
the two methods side-by-side in the reliability table. Selection
between Platt and isotonic is per-target by ECE on the held-out
calibration fold; the winning method is the saved artifact.

---

## 5. Confidence levels to evaluate (locked)

For both conformal sets / intervals and probability calibration:

| Level | Why |
|---|---|
| 0.50 | Median-coverage diagnostic; useful for cost-decision tuning |
| 0.80 | The "production" coverage Phase 6 will likely target |
| 0.90 | Mid-band calibration check |
| 0.95 | High-coverage stress test |

For the headline result, **0.90 confidence** is the
pre-registered primary level. Empirical coverage at 0.90 must land
within ±5 percentage points of nominal (i.e., 0.85 to 0.95) for
the calibration to be considered "successful." See Section 9 for
escalation triggers.

---

## 6. Cross-validation within the calibration set (locked)

The 257-film calibration set is used both to fit the conformal
quantiles / probability calibration AND to evaluate empirical
coverage. Naively using the same data for both is optimistic. The
honest approach: **5-fold cross-validation within the calibration
set**.

Procedure per (target, calibration method, confidence level):
1. Split the 257 calibration films into 5 stratified folds (seed
   42, stratified on the binary target for classification or on
   `decade_bucket` for regression).
2. For each fold: fit the calibration procedure (Platt + conformal)
   on the other 4 folds (~205 films); evaluate empirical coverage
   and reliability metrics on the held-out fold (~52 films).
3. Aggregate metrics across folds (mean and standard deviation).

The **deployed artifact** uses all 257 calibration films
(no held-out fold). The CV metrics estimate how well the deployed
artifact generalizes to unseen films.

This pattern matches Phase 4's repeated 5-fold logic; it lets the
Bayesian correlated-t-test compare Platt vs isotonic on the
per-fold metric arrays if the difference is interesting.

---

## 7. Reliability and coverage metric set (locked)

### 7.1 Probability-calibration metrics (classification only)

| Metric | Definition |
|---|---|
| **ECE** (Expected Calibration Error) | Weighted average |confidence − accuracy| across 10 equal-size probability bins |
| **MCE** (Maximum Calibration Error) | Maximum |confidence − accuracy| across the same bins |
| **Brier score** | Mean squared error between predicted probability and binary outcome |
| **Reliability curve** | Per-bin empirical accuracy vs predicted confidence (10 bins) |
| **Log-loss** | Carried over from Phase 4 metric vocabulary |

### 7.2 Conformal-coverage metrics

| Metric | Definition |
|---|---|
| **Empirical coverage** | Fraction of true labels / values inside the conformal set / interval at the nominal level |
| **Mean set size** (classification) | Average cardinality of the prediction set |
| **Singleton rate** (classification) | Fraction of films receiving a single-label prediction set (i.e., not "Refer") |
| **Mean interval width** (regression) | Average upper − lower in log_roi units |
| **Adaptivity** (regression) | Per-decile correlation between predicted magnitude and interval width |

### 7.3 Phase 6 readiness markers

* **Singleton rate at 0.90 confidence** — proxy for how often the
  system will emit a Greenlight / Pass without escalation. If
  below 50%, the system over-defers.
* **Refer rate by genre** (using the Phase 4 stratification) —
  validates that "Refer" fires more on character films than on
  genre films, consistent with the Phase 4 finding.

---

## 8. Output artifacts (locked)

### 8.1 Calibrated wrapper artifacts (per target)

`data/processed/phase5_calibrated_model_<target>.joblib` — a dict:

```
{
  "target": <str>,
  "phase4_winner": <Phase 4 model bundle reference>,
  "probability_calibrator": <fitted CalibratedClassifierCV or None for regression>,
  "conformal_wrapper": <fitted MAPIE Split* object>,
  "deployed_confidence_levels": [0.50, 0.80, 0.90, 0.95],
  "calibration_metrics": {
    "0.50": {"empirical_coverage": float, "mean_set_size": float, ...},
    "0.80": {...},
    ...
  },
  "ece": float,
  "brier_score": float,
  "platt_vs_isotonic_winner": "platt" | "isotonic" | None,
  "predict_proba": <function | None>,
  "predict_set": <function | None>,
  "predict_interval": <function | None>,
}
```

### 8.2 Tables

* `reports/tables/phase5_coverage.csv`: empirical coverage at each
  confidence level per (target, method, fold).
* `reports/tables/phase5_calibration_metrics.csv`: ECE, MCE, Brier,
  log-loss per (target, method, fold).
* `reports/tables/phase5_set_sizes.csv`: classification prediction
  set size distribution per (target, confidence level).
* `reports/tables/phase5_interval_widths.csv`: regression interval
  width distribution per (confidence level).
* `reports/tables/phase5_refer_by_genre.csv`: per-genre singleton
  rate / refer rate at 0.90 confidence.

### 8.3 Figures

* `phase5_reliability_post.png`: per-target post-calibration
  reliability diagrams for both Platt and isotonic candidates;
  overlay the diagonal and pre-Phase-5 curve from
  `phase4_calibration_pre.png` for direct comparison.
* `phase5_coverage_levels.png`: empirical-vs-nominal coverage
  curves at the four confidence levels per target.
* `phase5_set_size_distribution.png`: histogram of conformal
  prediction set sizes per classification target.
* `phase5_interval_width_distribution.png`: histogram of conformal
  interval widths for the regression target.
* `phase5_refer_by_genre.png`: bar chart of refer rate per genre
  on `roi_gt_2`, with the Phase 4 per-genre AUC overlaid for
  comparison.

### 8.4 Documentation

* `docs/summaries/phase_5_summary.md` per the Section 7 template.
* `docs/PROJECT_CONTEXT.md` Sections 8 (decisions log) and 9
  (phase status) updated.
* Per-run save_run directories under `runs/phase_5/<timestamp>_<name>/`.

---

## 9. Pre-registered escalation criteria (locked)

Each criterion below triggers a planning-conversation escalation
(no unilateral resolution by the executing chat). The Phase 5
mandatory end-of-phase escalation per `PROJECT_CONTEXT.md`
Section 11 is separate from these intra-phase triggers.

1. **Coverage failure**: empirical coverage at 0.90 nominal is
   outside [0.85, 0.95] for any target on the cross-validated
   evaluation. Indicates the calibration procedure does not
   generalize; would mean the underlying Phase 4 model needs
   rework.
2. **Over-deferral**: singleton rate at 0.90 confidence is below
   50% on the headline target (`roi_gt_2`). Indicates that more
   than half of films would route to "Refer to human reader",
   defeating the system's value proposition. Triggers either
   (a) re-tuning of the cost matrix design in Phase 6 to handle
   high refer rates, or (b) reconsideration of the Phase 4
   primary outcome variable.
3. **No improvement over uncalibrated**: post-Platt and
   post-isotonic ECE on the headline target are both worse than
   the raw `predict_proba` ECE. Indicates the calibration
   methodology is mis-applied or the model's probabilities are
   already well-calibrated and don't need post-hoc fixing.
4. **Conformity-score sensitivity**: the choice between
   `"lac"` and `"aps"` produces materially different (>5%
   absolute) coverage results on the headline target. Indicates
   the model's probabilities are unreliable and the
   non-conformity-aware MAPIE method (`"aps"`) should be the
   deployed default.

---

## 10. Compute budget

Phase 5 is fast compared to Phase 4. Estimated total wall-clock:
**under 30 minutes** including all five outputs.

| Step | Estimate |
|---|---|
| Load 3 Phase 4 winners + cal-set features | <1 minute |
| Probability calibration (Platt + isotonic, 5-fold within cal, 2 classifiers) | ~5 minutes |
| Conformal fitting (split conformal, 5-fold within cal, 3 targets × 2 conformity scores) | ~10 minutes |
| Reliability + coverage diagnostics | ~3 minutes |
| Per-genre refer-rate analysis | ~2 minutes |
| Figure rendering | ~3 minutes |
| Final artifact persistence | ~1 minute |

Reduction levers if needed:
1. Drop the `"aps"` comparison (keep `"lac"` only). Saves ~5 min.
2. Reduce CV repetitions from 5-fold to 3-fold. Saves ~30%.

---

## 11. Deviations from the roadmap

Two deliberate methodology decisions, both within the roadmap's
spirit but specific in implementation:

1. **5-fold CV within the calibration set** (instead of a single
   one-shot fit on all 257 films). Honest evaluation requires
   either a single train/cal/eval three-way split (already done
   in Phase 3) or cross-validation within the calibration set.
   The 5-fold approach matches the Phase 4 CV scheme and is
   compatible with the same Bayesian comparison framework. The
   deployed artifact still uses all 257 films.
2. **Both Platt and isotonic** scaling are evaluated, with per-
   target selection by ECE. The roadmap says "comparison of
   calibration quality before and after temperature scaling";
   Platt and isotonic are the two standard alternatives to
   temperature scaling for binary classification, both more
   appropriate than temperature scaling on this corpus size.

---

## 12. What is locked and what is not

**Locked.** Conformal method per task (Section 3), probability
calibration method per family (Section 4), confidence levels
(Section 5), CV scheme (Section 6), metric set (Section 7),
output artifacts (Section 8), escalation criteria (Section 9),
deviations (Section 11).

**Not locked (tactical, executing chat decides during
implementation).** Helper-function internal structure, plot
styling, table column ordering, intermediate artifact paths,
log-message verbosity, calibration-curve binning strategy
(equal-width vs equal-size bins; default equal-size).

**Pre-registration discipline.** The set above is not expanded
or revised after seeing results. If implementation surfaces a
methodology issue (e.g., MAPIE 1.4 API surprise, severely
imbalanced cal-set folds for `roi_gt_1` at 80% positive),
surface it as an issue and resolve via dialogue rather than
silently revising the pre-registration.
