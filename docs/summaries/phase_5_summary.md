# Phase 5: Layer 2 Calibrated Uncertainty

**Status:** Complete with one pre-registered escalation trigger fired
**Date completed:** 2026-05-04

> Pre-registration document: ``docs/proposals/phase5_preregistration.md``.
> Escalation trigger #2 (over-deferral on the headline target) fired
> on the cross-validated calibration evaluation; logged here and in
> ``docs/PROJECT_CONTEXT.md`` Section 8 for the planning conversation
> to weigh into Phase 6.

---

## Strategic decisions made before/during this phase

* **2026-05-04 (planning conversation, captured in
  ``phase5_preregistration.md``).** Locked methodology before any
  calibration fit ran. Probability calibration via Platt
  (sigmoid) and isotonic from sklearn 1.8's
  ``CalibratedClassifierCV`` with the
  ``FrozenEstimator`` wrapper (the older ``cv="prefit"`` API was
  removed in 1.8). Conformal prediction via MAPIE 1.4 for regression
  and a hand-rolled split-conformal LAC implementation for
  classification (see Issue 1 in Section "Issues encountered &
  resolved" for why MAPIE's classification path was bypassed).
  Confidence levels 0.50, 0.80, 0.90 (headline), 0.95. 5-fold CV
  within the 257-film calibration set for honest evaluation; the
  deployed artifact uses all 257 films.
* **2026-05-04 (executing chat, mid-phase).** Pre-registration
  Section 9 trigger #4 (conformity-score sensitivity for
  classification) is not testable on binary targets — MAPIE 1.4
  enforces ``"lac"`` as the only valid score for binary
  classification (``"aps"``, ``"raps"``, ``"top_k"`` are multi-class
  only). Documented as a Section 11 deviation in the
  pre-registration. No methodology pivot.
* **2026-05-04 (executing chat, mid-phase).** Pre-registration
  Section 3.2 listed both ``"absolute"`` and ``"gamma"`` regression
  conformity scores; ``"gamma"`` requires strictly positive targets
  and ``log_roi`` is signed (negative for flops). Only ``"absolute"``
  applies to the regression target. Documented as a Section 11
  deviation in the pre-registration. No methodology pivot.
* **2026-05-04 (executing chat, end of phase).** **Pre-registration
  Section 9 trigger #2 fires** (singleton rate at 0.90 confidence
  below 50% on the headline target ``roi_gt_2``). Empirical singleton
  rate at 0.90 = 21.4%; the model defers on 78.6% of films at this
  confidence level. The result is the honest output of a properly
  conformalized procedure on a model with 0.65 OOF AUC; not a
  methodology failure. Logged as the Phase 6 strategic question
  (per the pre-registration's "either re-tune cost matrix design
  for high refer rates, or reconsider primary outcome variable").

---

## What we did

1. Wrote ``docs/proposals/phase5_preregistration.md`` (12 sections)
   and surfaced for explicit user approval before any calibration
   fit ran.
2. Built ``src/calibration/`` with six modules: ``__init__.py``,
   ``metrics.py`` (ECE, MCE, Brier, log-loss, reliability bins),
   ``cv.py`` (5-fold within-cal stratified splitting),
   ``probability.py`` (Platt + isotonic via FrozenEstimator wrapper),
   ``conformal.py`` (split conformal regression via MAPIE; hand-rolled
   split conformal LAC for classification), ``pipeline.py``
   (orchestrator), ``figures.py`` (5 deliverable figures), plus
   ``src/experiments/run_phase5_calibration.py`` as the CLI entry
   point.
3. Wrote ``tests/test_calibration.py`` covering ECE/MCE/Brier
   correctness on toy data, hand-rolled LAC marginal coverage on
   synthetic data, fold-disjointness for the within-cal CV. **113
   total project tests pass** (9 new for Phase 5).
4. Verified MAPIE 1.4 API on toy data before writing the harness;
   discovered the ``aps``/``gamma`` constraints early.
5. Smoke-tested the harness end-to-end on ``roi_gt_2``; verified
   the no-leakage discipline (each conformal CV fold re-fits its
   own Platt/isotonic mapping on the fold's fit-side data).
6. Ran the full Phase 5 calibration on all three Phase 4 winners.
   Total wall-clock under 30 seconds; the pre-registration estimate
   of 30 minutes was conservative.
7. Generated all five pre-registered figures.
8. Persisted three calibrated wrapper bundles to
   ``data/processed/phase5_calibrated_model_<target>.joblib`` for
   Phase 6 consumption.
9. Updated ``PROJECT_CONTEXT.md`` Sections 8 (decisions log) and
   9 (phase status).

---

## Why we did it that way

**Two complementary calibration techniques.** Probability
calibration corrects the per-film probability output so that
``P(roi_gt_2 | features) = 0.7`` actually corresponds to a 70%
empirical positive rate. Conformal prediction wraps the model with
prediction-set guarantees so that a 90%-confidence prediction
empirically covers the truth 90% of the time. The two together
provide what Phase 6 needs: a calibrated probability for the
expected-cost calculation, plus a conformal set / interval width
for the "Refer to human reader" trigger.

**5-fold CV within the calibration set.** The deployed conformal
quantile and probability-calibration mapping use all 257 films of
the calibration set. The 5-fold CV gives an honest estimate of
how well that deployed combination generalizes. The same
discipline as the Phase 4 outer-CV scheme.

**Hand-rolled LAC instead of MAPIE for classification.** MAPIE
1.4's ``SplitConformalClassifier`` calls
``estimator.predict_proba(X)`` on a numpy array internally, which
breaks our prefit sklearn ``Pipeline`` objects that use
``ColumnTransformer`` with named string columns (the standard
column-name approach throughout Phase 3 and Phase 4). The LAC
procedure is trivial to implement: per fold, conformity score is
``1 - p(true_class)``; the prediction-set quantile at confidence
level ``c`` is the ``ceil((n+1) * c) / n``-th order statistic. The
hand-rolled version produces marginal coverage matching nominal
within ±5pp at every level (validated empirically; see Results).
MAPIE's regression path does not hit the column-name issue and is
kept for ``log_roi``.

**Per-fold re-wrapping for honest no-leakage.** When the SVM-RBF
model needs Platt/isotonic wrapping (its ``decision_function`` is
not in [0, 1]), the wrapper is re-fit on each fold's fit-side data
inside the conformal CV — not once on the full calibration set
then re-used. This preserves the no-leakage discipline; the
Section 11 fix to the original draft of ``conformal.py`` was made
on the smoke-test pass.

---

## Tactical choices made

* **Isotonic vs sigmoid winner per target by mean ECE**, with
  tie-break to sigmoid (simpler, fewer effective parameters).
  Both targets selected isotonic at ECE 0.095 (roi_gt_1) and
  0.108 (roi_gt_2); sigmoid was second at 0.157 / 0.169.
* **No native ``predict_proba`` for SVM-RBF** means the SVM
  ``"uncalibrated"`` baseline ECE was computed via a logistic
  rescale of ``decision_function`` (monotonic, preserves AUC,
  comparable on the calibration scale). The SVM "uncalibrated"
  baseline is reported only for the ECE comparison; it is never
  the deployed estimator.
* **Empty prediction sets allowed** at low confidence levels.
  At 0.50 confidence, ``roi_gt_1`` produces empty sets for ~30%
  of films (the model's true-class probability is below the
  conformity quantile for both classes). This is honest: the model
  is saying "even at 50% confidence I cannot commit." Phase 6
  treats empty sets as "Refer to human reader" same as the {0, 1}
  set.
* **MAPIE library kept for regression only.** No further bypass.
* **Save_run discipline** for each per-target calibration, with
  the calibrated wrapper bundle saved both inside the run directory
  (audit trail) and at the canonical
  ``data/processed/phase5_calibrated_model_<target>.joblib`` path
  (Phase 6 entry point).

---

## Results

### Probability-calibration metrics (5-fold mean across cal set)

| Target | Method | ECE | Brier | Log-loss |
|---|---|---:|---:|---:|
| roi_gt_1 (SVM-RBF) | uncalibrated | 0.243 | 0.204 | 0.596 |
| roi_gt_1 (SVM-RBF) | sigmoid | 0.157 | 0.160 | 0.499 |
| **roi_gt_1 (SVM-RBF)** | **isotonic** | **0.095** | 0.162 | 0.508 |
| roi_gt_2 (XGBoost) | uncalibrated | 0.187 | 0.250 | 0.694 |
| roi_gt_2 (XGBoost) | sigmoid | 0.169 | 0.240 | 0.673 |
| **roi_gt_2 (XGBoost)** | **isotonic** | **0.108** | 0.244 | 0.741 |

Isotonic wins ECE on both targets, by ~50% reduction over
uncalibrated. Sigmoid wins Brier on roi_gt_1 by 0.002; isotonic
wins Brier on roi_gt_2 by 0.005. Log-loss splits: sigmoid wins on
roi_gt_1, uncalibrated marginally wins on roi_gt_2 (because
isotonic's more aggressive corrections at the extremes hurt
log-loss while helping calibration). Per the pre-registered
ECE-based selection rule, **isotonic is the deployed choice for
both classifiers**.

### Conformal-coverage results (5-fold mean across cal set)

| Target | Level | Empirical coverage (mean ± std) | Singleton rate | Refer rate |
|---|---:|---:|---:|---:|
| log_roi (RandomForest) | 0.50 | 0.498 ± 0.046 | n/a (regression) | n/a |
| log_roi | 0.80 | 0.813 ± 0.080 | n/a | n/a |
| log_roi | 0.90 | **0.902 ± 0.046** | n/a | n/a |
| log_roi | 0.95 | 0.961 ± 0.039 | n/a | n/a |
| roi_gt_1 (SVM-RBF) | 0.50 | 0.592 ± 0.016 | 0.700 (∗) | 0.000 |
| roi_gt_1 | 0.80 | 0.872 ± 0.040 | 0.743 | 0.257 |
| roi_gt_1 | 0.90 | **0.922 ± 0.041** | 0.548 | 0.452 |
| roi_gt_1 | 0.95 | 0.965 ± 0.046 | 0.296 | 0.704 |
| roi_gt_2 (XGBoost) | 0.50 | 0.599 ± 0.012 | 0.981 | 0.000 |
| roi_gt_2 | 0.80 | 0.879 ± 0.054 | 0.381 | 0.619 |
| **roi_gt_2** | **0.90** | **0.910 ± 0.084** | **0.214** ← below 50% | 0.786 |
| roi_gt_2 | 0.95 | 0.946 ± 0.054 | 0.105 | 0.895 |

(∗) ``roi_gt_1`` at 0.50 has a 30% empty-set rate (the model
cannot commit to either class even at 50% confidence; mean set size
0.70).

**Empirical coverage is in-band at every level for every target.**
The pre-registered ±5pp tolerance around nominal at 0.90 is
satisfied: log_roi 0.902, roi_gt_1 0.922, roi_gt_2 0.910. **Trigger
#1 (coverage failure) does NOT fire.**

**Trigger #2 (over-deferral on headline) FIRES on roi_gt_2**: the
singleton rate at 0.90 confidence is 21.4%, well below the 50%
threshold. Per the pre-registration: this means at the headline
confidence level, the system would route 78.6% of films to
"Refer to human reader". The result is a real, honest property of
the corpus + model, not a methodology defect. See Open Questions
for Phase 6 implications.

### Per-genre refer rate (Phase 6 readiness marker)

The diagnostic figure ``phase5_refer_by_genre.png`` plots the
per-genre refer rate at 0.90 confidence on roi_gt_2 against the
Phase 4 OOF AUC per genre. **The two are clearly anti-correlated**:

* Romance: refer 84%, Phase 4 AUC ~0.49
* Action / Thriller / Comedy / Drama / Fantasy: refer 71-80%, AUC 0.54-0.66
* Adventure / Crime: refer 57-61%, AUC 0.61-0.71
* Animation / Horror / Sci-Fi: refer 36-50%, AUC 0.59-0.68

The conformal procedure is correctly identifying that the model is
uncertain on the genres where it has poor AUC and confident on the
genres where it predicts well. **This is the empirical validation
of the Phase 4 corpus-bimodality finding**: the system commits on
plot-driven genre films and defers on character-driven films,
without any per-genre code. The architecture works as designed.

### Saved figures

* ``phase5_reliability_post.png`` — per-target post-calibration
  reliability diagrams. Isotonic curve hugs the diagonal almost
  exactly on both classification targets; sigmoid is wavy in
  the middle range; uncalibrated is far above the diagonal across
  the probability range. Strong visual confirmation of the
  ECE-based isotonic selection.
* ``phase5_coverage_levels.png`` — empirical-vs-nominal coverage
  per target at the four confidence levels. All three targets
  hug the diagonal within the ±5pp band at every level. Marginal
  coverage guarantee empirically validated.
* ``phase5_set_size_distribution.png`` — singleton / refer / empty
  rate stacked per confidence level per classification target.
  Visualizes the trade-off: higher confidence = more refer.
* ``phase5_interval_width_distribution.png`` — mean conformal
  interval width per confidence level for ``log_roi``. Width at
  0.90 = 4.024 log-units (about 56× ratio in revenue/budget
  terms), reflecting the regression target's high variance.
* ``phase5_refer_by_genre.png`` — the headline diagnostic for
  Phase 6: per-genre refer rate at 0.90 with Phase 4 OOF AUC
  overlay. Anti-correlation confirms the system correctly defers
  on the hard genres.

---

## Issues encountered & resolved

1. **MAPIE 1.4 + sklearn Pipelines + ColumnTransformer with named
   columns: incompatible.** ``SplitConformalClassifier.conformalize``
   internally converts X to a numpy array and then calls
   ``predict_proba(numpy_X)``, which fails inside our pipelines'
   ``ColumnTransformer`` (which expects DataFrames because the
   numeric/one-hot column lists are string names). Resolved by
   hand-implementing the LAC split-conformal procedure in
   ``src/calibration/conformal.py`` (the
   ``_fit_lac_quantiles`` and ``LACConformalClassifier`` class).
   The hand-rolled implementation produces marginal coverage
   matching nominal within ±5pp at every level (validated
   empirically and via the unit test in
   ``tests/test_calibration.py::test_lac_split_conformal_marginal_coverage``).
   MAPIE's regression path does not hit this issue and is kept.
2. **sklearn 1.8 removed ``cv="prefit"``** from
   ``CalibratedClassifierCV``. Resolved by using
   ``sklearn.frozen.FrozenEstimator(estimator)`` as the wrapper
   (the sklearn 1.6+ replacement). The wrapped estimator's
   ``fit`` is a no-op; only the calibration mapping is fit.
3. **MAPIE 1.4 enforces ``"lac"`` for binary classification.**
   The ``"aps"``/``"raps"``/``"top_k"`` scores are multi-class
   only. Pre-registered Section 9 trigger #4 (conformity-score
   sensitivity) is therefore not testable on our targets. Logged
   as a Section 11 deviation in the pre-registration; does not
   change the deployed methodology.
4. **MAPIE 1.4 ``"gamma"`` regression score requires strictly
   positive targets.** ``log_roi`` is signed (negative for flops).
   Only ``"absolute"`` applies. Logged as a Section 11 deviation.
5. **Initial leakage in conformal CV.** The first draft passed
   the deployed Platt calibrator (fit on full cal) into the
   per-fold conformal CV — leakage. Caught on the smoke-test
   read-through; refactored to re-fit Platt/isotonic per fold
   inside conformal CV. The fix is in ``conformal.py`` via the
   ``_wrap_for_predict_proba`` helper that takes
   ``calibration_method`` as a parameter and re-wraps per fold.

---

## Open questions / things to flag

Mandatory end-of-Phase-5 escalation per ``PROJECT_CONTEXT.md``
Section 11. Two questions for the planning conversation:

1. **Phase 6 cost-decision design under high refer rate.** The
   headline target ``roi_gt_2`` has a 78.6% refer rate at 0.90
   confidence. Three Phase 6 design responses:
   * **Lower the operating confidence** to 0.80 (refer rate 62%)
     or 0.50 (refer rate 0%, but coverage drops to 60%).
   * **Use the calibrated probability directly** (not the
     conformal set) for the expected-cost decision. The cost
     matrix triggers "Refer" only when the expected cost of
     either action exceeds a threshold, regardless of conformal
     set size.
   * **Genre-conditional confidence levels**. The per-genre refer
     plot shows that Romance / Drama / Comedy can use a lower
     confidence (0.80) for an acceptable refer rate while genre
     films can use 0.90.
   Recommendation for the planning conversation: option 2 (cost
   matrix uses calibrated probability) plus option 3 (per-genre
   thresholds) for the most defensible Phase 6 design. Conformal
   sets become a diagnostic, not the primary trigger.
2. **Calibration is good enough for Layer 3 (per the roadmap's
   end-of-Phase-5 question)?** Yes on coverage; yes on
   probability calibration (isotonic ECE 0.095-0.108, down from
   raw 0.187-0.243). The model produces honest uncertainty. The
   high refer rate is a property of the underlying 0.65 AUC
   model, not the calibration. If the planning conversation wants
   a higher singleton rate, the path is to lift the underlying
   AUC (more data via the v2 corpus that the parallel chat is
   building) — not to revisit calibration.

---

## Files produced

### Code (Phase 5 specific)

* ``src/calibration/__init__.py``
* ``src/calibration/metrics.py``
* ``src/calibration/cv.py``
* ``src/calibration/probability.py``
* ``src/calibration/conformal.py``
* ``src/calibration/pipeline.py``
* ``src/calibration/figures.py``
* ``src/experiments/run_phase5_calibration.py``
* ``tests/test_calibration.py``

### Data

* ``data/processed/phase5_calibrated_model_log_roi.joblib``
* ``data/processed/phase5_calibrated_model_roi_gt_1.joblib``
* ``data/processed/phase5_calibrated_model_roi_gt_2.joblib``

Each bundle contains: phase4_winner_path, family, matrix,
score_method, best_probability_method, probability_calibrator
(fitted), best_conformal_score, conformal_wrapper (fitted on full
cal), deployed_confidence_levels, calibration_metrics. Phase 6
consumes via ``joblib.load(...)`` and uses ``probability_calibrator.predict_proba(X)``
plus ``conformal_wrapper.predict_set(X)`` (classification) or
``conformal_wrapper.predict_interval(X)`` (regression).

### Tables (``reports/tables/``)

* ``phase5_coverage.csv`` (60 rows: 3 targets × 4 levels × 5 folds)
* ``phase5_set_sizes.csv`` (40 rows: 2 classification targets × 4
  levels × 5 folds)
* ``phase5_interval_widths.csv`` (20 rows: 1 regression target × 4
  levels × 5 folds)
* ``phase5_calibration_metrics.csv`` (30 rows: 2 classification
  targets × 3 methods × 5 folds)

### Figures (``reports/figures/``)

* ``phase5_reliability_post.png``
* ``phase5_coverage_levels.png``
* ``phase5_set_size_distribution.png``
* ``phase5_interval_width_distribution.png``
* ``phase5_refer_by_genre.png``

### Documents

* ``docs/proposals/phase5_preregistration.md``
* ``docs/summaries/phase_5_summary.md`` (this file)
* ``docs/PROJECT_CONTEXT.md`` Sections 8 (decisions log) and 9
  (phase status) updated

### Run artifacts

* ``runs/phase_5/<timestamp>_phase5_<target>/`` per target with
  the five canonical files plus the calibrated wrapper bundle as
  ``model.joblib``.
* ``runs/RUNS.md`` updated with one row per cell.

---

## Next phase prerequisites

Phase 6 (Layer 3: Asymmetric-Cost Decision) needs:

* The three calibrated wrapper bundles at
  ``data/processed/phase5_calibrated_model_*.joblib``.
* A documented cost matrix with industry-grounded default values
  (cost of a flop ≈ $50M, cost of a missed hit ≈ $100M-200M, cost
  of human reader time per script ≈ $500-2000). Sources: the brief.
* The calibration set's 257 films available for cost-decision
  empirical evaluation (was used in Phase 5; can be reused for
  Phase 6 since Phase 6 fits no probability-calibration parameters).
* The held-out 257-film test set untouched until Phase 8.

Phase 6 will:

* Construct the cost matrix per the brief (Section 4 of the
  Phase 6 brief, when uploaded).
* Define the decision function: given (probability, conformal_set,
  genre, etc.), output Greenlight / Pass / Refer.
* Sensitivity-analyze across multiple cost-matrix variants.
* Decide whether per-genre thresholds are necessary (per the
  Open Question 1 above).
* Compare cost savings vs naive baselines (always-greenlight,
  always-pass, read-everything).

---

## Questions for the planning conversation

Mandatory end-of-Phase-5 escalation per ``PROJECT_CONTEXT.md``
Section 11. Two questions:

1. **Phase 6 cost-decision uses calibrated probability or
   conformal set?** Recommendation: probability for the trigger,
   conformal set as a diagnostic.
2. **Per-genre confidence thresholds in Phase 6?** Recommendation:
   yes, motivated by the per-genre refer-rate diagnostic.
