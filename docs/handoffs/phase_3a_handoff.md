# Phase 3a: Baseline handoff

**Status:** Multi-family baseline complete. Reporting now spans both
in-sample fit on the train split and out-of-fold cross-validation on
the same train split, across the four model families. Phase 3b
feature ablations run against the OOF numbers as the primary
comparison; the train numbers serve as an overfit-gap diagnostic.
**Date:** 2026-05-03
**Last revised:** 2026-05-03; this handoff has accumulated four
extensions since first writing. Each is dated and incorporated below.

This is an interim handoff at the Phase 3a / 3b boundary, per the
brief's instruction to revert to the planning conversation with a
summary of 3a and ask on next steps. It is not the Phase 3 final
summary (that comes at end of Task 5, using the standard template).

---

## 1. Strategic decisions in scope

Locked in by the planning conversation (Phase 3 brief Section 2)
and implemented directly:

* **Three targets in parallel.** `log_roi` (regression), `roi_gt_1`
  and `roi_gt_2` (classification). Threshold-consistent: `roi_gt_1`
  is the same as `(log_roi > 0)` and `roi_gt_2` is the same as
  `(log_roi > ln 2)`.
* **Single train / cal / test split**, 70 / 15 / 15, seed 42,
  stratified by `(primary_genre_bucketed, decade_bucket)`.
* **Calibration set carved here**, reserved for Phase 5 conformal
  prediction. Not touched in 3a.

Four additional strategic decisions were made during execution and
are documented here for the audit trail:

* **2026-05-03 (mid-3a):** Baseline feature parameterization
  revised. `log1p` applied to the six heavy-tailed structural counts
  before z-scoring; `log_runtime` added to the deployable baseline.
* **2026-05-03 (3a/3b boundary):** First lexical ablation came in
  with negative drift on the linear baseline, prompting the
  question whether the issue was the features or the linear model
  family.
* **2026-05-03 (multi-family expansion):** Phase 3a baselines and
  every Phase 3b ablation now run across **four model families**
  (linear, HistGB, KNN, SVM-RBF) with distinct inductive biases.
* **2026-05-03 (metric vocabulary update):** The reported metric
  set is changed. Regression metrics are now MSE, RMSE, MAE, and
  CVRMSE (coefficient of variation of RMSE: RMSE divided by the
  absolute mean of the target). R-squared is removed from the
  reported set in favour of these absolute and normalized
  measures, which are more robust on small samples and easier to
  compare across feature configurations. Classification metrics
  are now AUC-ROC, PR-AUC, F1 (at the 0.5 threshold), and log-loss.
  In addition, **both in-sample (train) and out-of-fold (oof)
  values are reported** for every (family, target, metric)
  combination so the train-versus-oof gap is visible as an
  overfitting diagnostic. Held-out 15% test set and 15%
  calibration set remain untouched (Phase 8 and Phase 5
  respectively).

---

## 2. Tactical choices

* **Decade bucketing for stratification:** `pre_1980s` (one stratum
  pooling 103 films across 1932-1979), `1980s`, `1990s`, `2000s`,
  `2010s_2020s` (one stratum because 2020-2023 is too thin alone).
* **Rare-cell pooling:** composite (genre, decade) cells with fewer
  than 5 films are pooled into a single `rare|rare` stratum. 38
  films (2.2%) land in this pool.
* **Cross-validation:** 5-fold on the train split only (n = 1,199).
  `KFold` for regression, `StratifiedKFold` for classification. The
  trainer iterates folds manually so it can capture both hard and
  score predictions per fold from a single fit, then computes
  metrics over the concatenated OOF predictions.
* **In-sample (train) reporting:** for each (family, target,
  feature config) combination, a separate fit on the entire train
  split produces the in-sample predictions used for the train-side
  metrics. The same scaffold is used for hard predictions (F1) and
  score predictions (AUC, PR-AUC, log-loss) per family.
* **Bootstrap 95% CIs:** 1,000 resamples on the OOF predictions
  only (in-sample point estimates are reported without a bootstrap
  CI because the train predictions are not held-out).
* **Four model families:**
  * **`linear`:** RidgeCV regression (alpha grid logspace(-3, 3,
    13), LOO-GCV); LogisticRegressionCV classification with L2
    penalty, same C grid, inner 5-fold stratified CV optimizing
    AUC. Pipeline: `SimpleImputer(median) -> StandardScaler` on
    numeric features, passthrough on genre dummies.
  * **`histgb`:** `HistGradientBoostingRegressor` and
    `HistGradientBoostingClassifier`, `max_iter=300, max_depth=4,
    learning_rate=0.05, early_stopping=True`. No imputation or
    scaling in the numeric branch (algorithm handles NaN natively
    and is invariant to monotonic transforms).
  * **`knn`:** `KNeighborsRegressor` and `KNeighborsClassifier`,
    `n_neighbors=20, weights="distance"`. Pipeline: same
    impute-and-scale as linear.
  * **`svm`:** `SVR` and `SVC` with RBF kernel, `C=1.0,
    gamma="scale"`. Classification produces decision-function
    scores rather than calibrated probabilities, so log-loss is
    not computed for the SVM family (left as `NaN` in the table);
    AUC-ROC, PR-AUC, and F1 are unaffected.
* **Baseline feature set (original):** 7 structural metrics
  (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
  `total_dialogue_chars`, `total_action_chars`,
  `dialogue_to_total_text_ratio`, `parse_warning_count`),
  `release_year_parsed`, and `primary_genre_bucketed` one-hot (13
  dummies). Sanity-check baseline adds `log_budget` separately.
* **Baseline feature set (revised):** same as original with two
  changes: `log1p` applied to the six heavy-tailed structural
  counts before z-scoring; `log_runtime` added.

---

## 3. Headline results across 4 model families and 2 evaluation sets

Full table at `reports/tables/phase3a_baseline.csv` (384 rows: 4
feature sets × 4 families × 2 evaluation sets × 3 targets ×
{4 regression metrics or 4 classification metrics}). Bootstrap 95%
CIs apply to OOF only.

### 3.1 Revised dialogue-only baseline, OOF (the Phase 3b floor)

OOF values per family. The lower-is-better arrows mark MSE, RMSE,
MAE, CVRMSE, and log-loss; higher-is-better marks AUC-ROC, PR-AUC,
and F1. Bold marks the best family per metric.

| Target | Metric | Better | linear | histgb | knn | svm |
|---|---|---|---:|---:|---:|---:|
| log_roi | MSE | lower | 1.792 | **1.761** | 1.860 | 1.842 |
| log_roi | RMSE | lower | 1.339 | **1.327** | 1.364 | 1.357 |
| log_roi | MAE | lower | 0.948 | **0.943** | 0.972 | 0.954 |
| log_roi | CVRMSE | lower | 0.993 | **0.985** | 1.012 | 1.007 |
| roi_gt_1 | AUC-ROC | higher | **0.558** | 0.552 | 0.527 | 0.504 |
| roi_gt_1 | PR-AUC | higher | **0.846** | 0.843 | 0.829 | 0.831 |
| roi_gt_1 | F1 | higher | 0.893 | 0.893 | **0.894** | **0.894** |
| roi_gt_1 | log-loss | lower | **0.493** | 0.486 | 0.593 | n/a |
| roi_gt_2 | AUC-ROC | higher | 0.602 | **0.610** | 0.578 | 0.534 |
| roi_gt_2 | PR-AUC | higher | **0.739** | 0.731 | 0.725 | 0.676 |
| roi_gt_2 | F1 | higher | 0.777 | 0.777 | 0.741 | **0.784** |
| roi_gt_2 | log-loss | lower | **0.635** | 0.635 | 0.648 | n/a |

Note on log-loss: for `roi_gt_1` (80% positive base rate), HistGB's
log-loss of 0.486 is slightly lower than linear's 0.493; F1 saturates
near 0.89 across all families because predicting "positive" for most
films achieves high F1 mechanically given the imbalance.

### 3.2 Train-versus-OOF gap on the revised dialogue-only baseline (overfit diagnostic)

The gap between in-sample fit and OOF cross-validation indicates
how much each family is over-fitting the training data. Larger
absolute gaps mean more overfit.

| Family | Target | Metric | Train | OOF | Gap (train − OOF) |
|---|---|---|---:|---:|---:|
| linear | log_roi | RMSE | 1.320 | 1.339 | -0.019 |
| linear | roi_gt_1 | AUC-ROC | 0.635 | 0.558 | **+0.077** |
| linear | roi_gt_2 | AUC-ROC | 0.640 | 0.602 | +0.038 |
| histgb | log_roi | RMSE | 1.205 | 1.327 | **-0.122** |
| histgb | roi_gt_1 | AUC-ROC | 0.775 | 0.552 | **+0.223** |
| histgb | roi_gt_2 | AUC-ROC | 0.811 | 0.610 | **+0.201** |
| knn | log_roi | RMSE | (n/a, perfect on train) | 1.364 | (large) |
| svm | log_roi | RMSE | 1.195 | 1.357 | -0.162 |

**Interpretation.** HistGB substantially overfits the train data:
its in-sample AUC-ROC numbers are 22 percentage points above OOF on
both classification targets. The 300-iteration cap with depth 4 is
still flexible enough to memorize idiosyncrasies of the train split.
KNN with `weights="distance"` is mathematically perfect on
in-sample (every point is its own nearest neighbour with weight
infinity); the in-sample numbers for KNN are not informative and
the train-side reporting of KNN is an artifact of the schema. SVM
also shows a substantial gap. Linear is the most stable family by
this diagnostic, with smaller absolute gaps on every metric.

The OOF numbers are the right comparison point for Phase 3b
ablation lift; the train numbers are kept as a diagnostic so the
report can show that HistGB's apparent strength on OOF is achieved
despite a very flexible fit on train (i.e., HistGB is not lucky on
OOF, it is genuinely competitive even after substantial overfit
absorption by the validation split).

### 3.3 Revised with-budget sanity-check, OOF (ceiling, not deployable)

| Target | Metric | linear | histgb | knn | svm |
|---|---|---:|---:|---:|---:|
| log_roi | RMSE | 1.305 | **1.296** | 1.336 | 1.343 |
| log_roi | MAE | 0.947 | **0.929** | 0.962 | 0.940 |
| log_roi | CVRMSE | 0.969 | **0.962** | 0.991 | 0.996 |
| roi_gt_1 | AUC-ROC | 0.555 | **0.598** | 0.539 | 0.512 |
| roi_gt_1 | PR-AUC | 0.845 | **0.869** | 0.843 | 0.834 |
| roi_gt_1 | log-loss | 0.494 | **0.474** | 0.586 | n/a |
| roi_gt_2 | AUC-ROC | 0.603 | **0.621** | 0.594 | 0.555 |
| roi_gt_2 | PR-AUC | 0.738 | **0.751** | 0.733 | 0.704 |

Adding `log_budget` lifts performance modestly across all families.
HistGB benefits the most (log-loss drop of 0.012 on `roi_gt_1`), but
the absolute gains on the headline metrics remain small (e.g.,
linear `roi_gt_2` AUC moves from 0.602 to 0.603). Same survivorship-
bias interpretation as before: budget knowledge does not
substantially separate hits from misses within a corpus pre-filtered
on commercial recognition.

---

## 4. Threshold check

The Phase 3 brief's escalation rule (R² < 0.05 OR AUC-ROC < 0.55
across all targets on the deployable linear baseline) is no longer
literally applicable because R² has been removed from the metric
set as of 2026-05-03. The original gating decision (proceed to
Phase 3b) was made before the metric change and remains valid: the
linear-family OOF numbers cleared the brief's thresholds at the
time. From this point forward, per-group ablation lifts against
the floor are the primary signal rather than absolute-threshold
checks.

For interested readers, the equivalent check translated to RMSE:
`R² ≥ 0.05` on `log_roi` corresponds to `RMSE ≤ 0.975 × std(y) ≈
1.38`. The linear-family OOF RMSE on the revised dialogue-only
baseline is 1.339, which clears that translated threshold. HistGB
(1.327) clears it more comfortably.

---

## 5. Interpretation

Seven points worth flagging.

**1. The structural baseline carries non-trivial signal that
non-linear models extract better than linear, on OOF.** HistGB
beats linear on the regression target (RMSE 1.327 vs 1.339, MAE
0.943 vs 0.948) and on `roi_gt_2` AUC (0.610 vs 0.602) on the
out-of-fold predictions. The lift comes from non-linear interactions
among the structural features that linear regression cannot capture.

**2. HistGB substantially overfits in-sample.** Train numbers are
much better than OOF for HistGB (RMSE 1.20 train vs 1.33 OOF; AUC
0.78 train vs 0.55 OOF on `roi_gt_1`). The conservative defaults
(max_depth=4, learning_rate=0.05, max_iter=300, early_stopping=True)
are not enough to prevent the algorithm from memorizing
idiosyncrasies of the train split. This is a useful methodology
finding for Phase 4: HistGB hyperparameter search should explore
even more conservative regularization (lower `learning_rate`,
smaller `max_iter`, larger `min_samples_leaf`).

**3. The revision lift behaves as the planning conversation
predicted.** Headline post-revision OOF movements on the deployable
linear set: `log_roi` MAE drops 0.007 (better calibration on a
heavy-tailed target); `roi_gt_2` AUC lifts 0.020 to 0.602 with a
CI lower bound that fully clears the literature-default 0.55
threshold; RMSE moves modestly downward. With-budget shows a
larger improvement.

**4. `roi_gt_1` is the noisiest target across all families on
OOF.** AUC values cluster around 0.55 on dialogue-only and drift
toward 0.60 when budget is added. The 80%-positive base rate
makes the target signal-thin. F1 saturates at 0.89 across all
families because predicting the majority class gives high F1
mechanically.

**5. `roi_gt_2` is the cleanest target across all families on
OOF.** Linear 0.602, HistGB 0.610, KNN 0.578, SVM 0.534. The
"net-profitable" distinction tracks observable features more
crisply than the gross-profitable distinction. PR-AUC tells the
same story.

**6. Budget barely lifts deployable performance even after the
revision.** With-budget linear `log_roi` RMSE is 1.305 vs the
deployable 1.339, a meaningful but modest gain. AUC on the
classification targets moves modestly with budget added. Good news
for project framing: dialogue features in Phase 3b are not
competing against an obvious dominant budget signal.

**7. KNN and SVM are weak baselines on this corpus.** Both
underperform linear on most OOF targets. The likely cause is the
high feature-to-sample ratio (26 features on n = 1,199), which is
unfavourable for distance-based methods. Their inclusion in the
multi-family suite is for diagnostic completeness rather than as
candidates for Phase 4 selection.

---

## 6. Files produced (Phase 3a)

### Code
* `src/features/__init__.py`, `src/features/split.py`,
  `src/features/targets.py`, `src/features/baseline_features.py`
* `src/models/__init__.py`, `src/models/baseline/__init__.py`,
  `src/models/baseline/metrics.py`,
  `src/models/baseline/train.py` (4-family, 2-eval-set trainer)

### Data
* `data/processed/split_assignments.parquet`: one row per film
  with columns for `imdb_id`, the stratification cell, and the
  assigned split.

### Tables
* `reports/tables/phase3_split_diagnostics.csv`: per-stratum split
  counts (57 strata; every named stratum has at least one film in
  each split).
* `reports/tables/phase3a_baseline.csv`: 384 rows covering 4
  feature configurations × 4 model families × 2 evaluation sets
  (train, oof) × 3 targets × {4 metric rows per task}.

---

## 7. What's next

Phase 3b ablation runs against this 4-family, 2-eval-set floor.
Each feature group's lift is computed per family per eval set.
The linear-family OOF lift remains the "official" ablation row;
the other three families provide the cross-paradigm diagnostic;
the train-side numbers provide the overfit-gap context.

The first Phase 3b ablation row (lexical) is documented in
`docs/handoffs/phase_3b_lexical_handoff.md`.

After all five Phase 3b groups land, the Phase 3c combinations
sub-phase (added 2026-05-03) runs three to five pre-specified
joint-feature-set evaluations against the same floor.

---

## 8. Resolved questions

The four questions raised at the original 3a/3b handoff (numbers
strong enough to proceed, primary outcome preference,
`data_quality_flag` handling, group ordering) and the
multi-family-expansion question raised at the lexical handoff are
all resolved. Audit trail in the decisions log of
`docs/PROJECT_CONTEXT.md` Section 8.
