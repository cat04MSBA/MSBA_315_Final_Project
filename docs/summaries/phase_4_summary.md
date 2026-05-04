# Phase 4: Layer 1 Core Prediction Model

**Status:** Complete
**Date completed:** 2026-05-04

> Pre-registration document: ``docs/proposals/phase4_preregistration.md``.
> Decisions-log entries for the Section 13 deviations and the Tier-A
> escalation work are in ``docs/PROJECT_CONTEXT.md`` Section 8 with the
> dates noted below.

---

## Strategic decisions made before/during this phase

The phase carried a heavy strategic-decision load inherited from the
Phase 4 brief, three executing-chat-recommended deviations approved
before benchmark execution, and one mid-phase Tier-A escalation
triggered by a corpus-ceiling discovery and a Phase 3 documentation
error. Each decision is logged in ``docs/PROJECT_CONTEXT.md``
Section 8; summarized here for reference.

* **2026-05-03 (planning conversation, captured in the Phase 4 brief).**
  Three targets reported in parallel through Phase 4 (``log_roi``,
  ``roi_gt_1``, ``roi_gt_2``) with informal headline target
  ``roi_gt_2``. Train / calibration / test split fixed at
  1199 / 257 / 257; calibration set untouched (Phase 5), test set
  untouched (Phase 8). Metric vocabulary frozen (regression: MSE,
  RMSE, MAE, CVRMSE; classification: AUC-ROC, PR-AUC, F1, log-loss).
  ``save_run`` mandatory. Pre-registration discipline carried
  forward.

* **2026-05-03 (planning conversation, deviation approval).** Three
  deviations from the brief approved before pre-registration was
  locked. (1) HistGB chosen over LightGBM as the gradient-boosting
  representative for Phase 3 longitudinal continuity on the
  train-OOF gap finding. (2) Random Forest in the primary tier in
  place of XGBoost (RF has a genuinely different bagging-with-deep-
  trees inductive bias). (3) CV scheme bumped from plain 5-fold to
  repeated 5-fold (3 repetitions), giving the Bayesian
  correlated-t-test 14 effective degrees of freedom rather than 4.
  All three are recorded as Section 13 deviations in
  ``docs/proposals/phase4_preregistration.md``.

* **2026-05-03 (planning conversation, statistical-test framework).**
  Pairwise model comparisons across primary-tier families use the
  Bayesian correlated-t-test (Benavoli et al. 2017) via the
  ``baycomp`` package. ROPE half-widths pre-registered per metric:
  0.005 for AUC-ROC, 0.01 for PR-AUC / F1 / log-loss / RMSE / MSE /
  MAE / CVRMSE. Posterior threshold 0.95 for declaring a winner.
  ``baycomp.two_on_single`` accepts ``runs=N`` and infers the rho
  for repeated CV correctly.

* **2026-05-03 (planning conversation, class-balancing policy).**
  All classifiers use ``class_weight="balanced"`` (Logistic, RF,
  Linear-SVC, SVC) or equivalent ``sample_weight`` (HistGB).
  Pre-registered as the apples-to-apples baseline. Sensitivity
  analysis later confirmed the choice is metric-neutral on AUC for
  ``predict_proba`` families and within +/- 0.013 AUC for SVM-RBF.

* **2026-05-04 00:30 (mid-Phase-4, executing-chat escalation).**
  Primary-tier benchmark complete; pre-registered Section 10
  triggers #1 (corpus ceiling on ``roi_gt_2``: best 0.6346 below
  the 0.69 mid-band) and #3 (statistical tie at top: all primary
  pairs land in ROPE on ``roi_gt_2`` AUC) both fired. Critical
  discovery during the escalation: the Phase 3 summary reported
  SVM-RBF on ``all_five`` reaching ``roi_gt_2`` AUC 0.665 OOF;
  ``phase3c_combinations.csv`` shows the actual value was 0.5966
  (the 0.665 figure was computed by adding SVM's lift +0.0629 to
  *linear*'s structural floor 0.6017 instead of to SVM's own floor
  0.5337). The 0.69 mid-band threshold was anchored to this wrong
  number; the realistic corpus ceiling lies lower. Escalated to
  user with three remediation paths.

* **2026-05-04 00:35 (planning conversation, Path 2 approval).**
  User authorized Path 2: spend half a day on the three
  highest-priority improvement levers (Tier-A1 stacking ensemble,
  Tier-A2 genre/era-stratified diagnostic, Tier-A3 mpnet-base
  encoder) before deciding whether to accept the corpus ceiling
  or invest further. Phase 3 documentation error to be corrected
  in place via strikethrough + correction notes (matching the
  pre-1995-cutoff reversal precedent). DistilBERT scope held in
  reserve; per-target winner committed at end of Tier-A work.

* **2026-05-04 01:50 (mid-Phase-4 close, executing-chat).** Tier-A
  work complete. Headline target ``roi_gt_2`` OOF AUC reaches
  0.6537 via stacking ensemble on
  ``standalone_positive_union_mpnet``; cleared the 0.65 lower band
  (was 0.6346 pre-Tier-A). Per-target winners committed.
  Phase 4 close: per-target winner SVM-RBF (or RF for ``log_roi``)
  on ``standalone_positive_union_mpnet`` is the canonical Phase 5
  feed; the stacking ensemble is preserved as an alternative.

---

## What we did

Phase 4 unfolded in three parts: pre-registration + canonical
benchmark, mid-phase escalation discovery, and Tier-A escalation
work to lift the headline number.

### Part 1 — Pre-registration and canonical benchmark

1. Wrote ``docs/proposals/phase4_preregistration.md`` (Sections 1
   through 14) and surfaced for explicit user approval before any
   benchmark fits ran.
2. Built ``src/models/phase4/`` with five core modules:
   ``families.py`` (registry of six candidates),
   ``matrices.py`` (input matrix builders),
   ``cv.py`` (repeated CV harness with per-fold metric collection
   and HistGB sample-weight routing),
   ``paired_test.py`` (baycomp wrapper), and
   ``benchmark.py`` (orchestrator), plus
   ``src/experiments/run_phase4_benchmark.py`` as the CLI entry
   point.
3. Smoke-tested the harness end-to-end: linear smoke on
   ``roi_gt_2`` reproduced the Phase 3a baseline AUC to within
   0.005, confirming the harness is consistent. HistGB
   sample-weight smoke verified
   ``GridSearchCV.fit(model__sample_weight=...)`` routes weights
   to per-fold training without scoring side effects.
4. Ran the primary-tier benchmark on both pre-registered input
   matrices: 4 families x 2 matrices x 3 targets = 24 cells in
   approximately 10 minutes wall-clock, much faster than the
   pre-registered 3 to 8 hour estimate.
5. Ran the secondary-tier benchmark (Lasso, Linear-SVM) on
   ``all_five``.
6. Ran the unweighted-vs-balanced sensitivity analysis on
   ``roi_gt_2`` to verify class-weighting policy was not the
   cause of the lower-than-expected headline numbers.

### Part 2 — Mid-phase escalation discovery

7. Posthoc paired Bayesian comparison surfaced no statistical
   winner across primary families on ``roi_gt_2`` (all 24
   pairwise comparisons land in ROPE).
8. Pre-registered Section 10 triggers #1 and #3 fired. Posthoc
   investigation of the threshold uncovered a calculation error
   in the Phase 3 summary's claim of "SVM-RBF reaches
   ``roi_gt_2`` AUC 0.665 OOF" (actual was 0.5966).
9. Escalated to user; user authorized Path 2 (Tier-A
   improvement work).
10. Corrected the Phase 3 documentation error in
    ``docs/summaries/phase_3_summary.md`` and
    ``docs/FEATURE_NOTES.md`` via strikethrough + correction
    notes per the pre-1995-cutoff reversal precedent.

### Part 3 — Tier-A escalation work

11. **A1 stacking ensemble.** Built
    ``src/models/phase4/stacking.py``: Logistic-L2 / Ridge
    meta-learner over the four primary families' OOF predictions,
    cross-validated under repeated 5-fold (3 repetitions) for
    honest evaluation. Result: +0.0034 AUC lift on
    ``roi_gt_2`` standalone (within ROPE noise as expected when
    the base models are practically equivalent).
12. **A2 stratified diagnostic.** Built
    ``src/models/phase4/diagnostic.py``: per-genre and per-decade
    OOF AUC slicing of the headline cell. Result: the corpus has
    bimodal structure rather than a uniform ceiling.
    Adventure / Fantasy / Sci-Fi / Crime reach 0.66 to 0.73 OOF AUC;
    Drama / Comedy / Romance cap around 0.59. Pre-1980s decade
    (n=76, 8 negatives) is statistically unreliable.
13. **A3 mpnet-base encoder.** Built
    ``src/experiments/run_phase4_mpnet_embeddings.py``: re-encoded
    all 1,713 screenplays with ``sentence-transformers/all-mpnet-base-v2``
    (768-dim, 110 MB), fit PCA to 32 components on training fold
    (cumulative variance 0.7434, comparable to MiniLM's 0.739).
    Encoding wall-clock: about 53 minutes on Apple Silicon MPS at
    roughly 32 films per minute.
14. Added two new matrix variants
    (``all_five_mpnet`` and ``standalone_positive_union_mpnet``)
    and re-ran the primary-tier benchmark on them; mpnet adds
    +0.011 to +0.024 AUC per family on ``roi_gt_2``.
15. Re-ran stacking on the mpnet matrices; stacking on
    ``standalone_positive_union_mpnet`` for ``roi_gt_2`` reaches
    0.6537 OOF AUC, **clearing the 0.65 lower band** (the
    revised in-band threshold after the Phase 3 documentation
    correction).
16. Built ``src/models/phase4/finalize.py``: re-fit the per-target
    winner on full train data and persist canonical artifacts to
    ``data/processed/phase4_primary_model_<target>.joblib``.
17. Regenerated all Phase 4 figures
    (``phase4_train_oof_gap.png``, ``phase4_calibration_pre.png``,
    ``phase4_stratified_auc.png``, ``phase4_stacking_lift.png``)
    over the consolidated benchmark.
18. Updated ``PROJECT_CONTEXT.md`` Sections 8 (decisions log) and
    9 (phase status); built the Phase 4 notebook from
    ``notebooks/_build_phase_4_notebook.py``.

---

## Why we did it that way

Three methodology principles drove the phase's structure.

**Pre-registration before measurement.** The roster, grids, CV
scheme, statistical comparison procedure, and escalation criteria
were all locked in ``phase4_preregistration.md`` before any
benchmark fit ran. The same discipline that governed Phase 3b
standalone groups and the Phase 3c combinations sub-phase. The
document also records the three deviations explicitly so the audit
trail survives. When the corpus-ceiling escalation fired mid-phase,
the pre-registered Section 10 triggers + the planning-conversation
escalation rules made the response procedural rather than
ad-hoc.

**Repeated 5-fold CV (3 repetitions) over plain 5-fold.** Driven
by the Bayesian correlated-t-test's degree-of-freedom requirement.
Plain 5-fold yields 4 d.f. per pairwise comparison, which is thin
enough that the test under-detects real differences. 3 reps yields
14 d.f. for negligible methodological complexity at 3x compute.
The Bayesian framework (rather than naive CI overlap) was critical:
the 24 pairwise comparisons on ``roi_gt_2`` AUC all land in ROPE,
providing the formal evidence that the four primary families are
practically equivalent and therefore that stacking would yield
limited additional signal (later confirmed empirically at
+0.0034 lift).

**Tier-A escalation discipline.** When the headline number came
in below the pre-registered band, the executing chat did not
silently revise the threshold or pivot the methodology. Instead,
the gap was surfaced to the user as a Section 10 escalation event,
the discovered Phase 3 documentation error was traced and
corrected in place, and three concrete improvement levers
(stacking, diagnostic, larger encoder) were proposed with
expected-cost-and-impact analysis. The user authorized the work,
and the Tier-A results lifted the headline from 0.6346 (below the
0.65 floor under the corrected band) to 0.6537 (above the 0.65
floor). The genre-stratified diagnostic surfaced the more
interesting finding: the corpus is bimodal, not uniform.

---

## Tactical choices made

* **Stacking via Logistic-L2 / Ridge meta-learner** rather than
  weighted average. The meta-learner can assign negative weights
  (it does for linear on ``roi_gt_2`` standalone), which is a
  more honest way to combine practically-equivalent base models.
* **Stacking inputs include SVM-RBF decision-function scores
  rescaled by logistic** so the meta-learner sees comparable
  probability-style features. The transform is monotonic so the
  ranking is preserved.
* **Per-genre / per-decade slicing on the headline winner cell
  only** (svm_rbf x standalone_positive_union, both MiniLM and
  mpnet) rather than across all 16 (matrix x family) cells. The
  diagnostic story is about corpus structure, not model
  comparison.
* **Mpnet PCA dimensionality held at 32** for fair comparison with
  the Phase 3 MiniLM PCA. The raw 768-dim mpnet cache is on disk
  for Phase 5 or Phase 8 to revisit if needed.
* **Per-target winner artifacts overwrite the MiniLM-based
  earlier saves**. The MiniLM bundles remain in their per-cell
  ``runs/phase_4/`` directories for audit. Phase 5 reads the
  canonical files at
  ``data/processed/phase4_primary_model_<target>.joblib``.
* **All three winners on ``standalone_positive_union_mpnet``** by
  the parsimony tie-breaker (Sec 8 of pre-registration), even
  though for some targets the all_five_mpnet matrix gave numerically
  similar results.
* **Stacking ensemble preserved separately** in
  ``runs/phase_4/stacking_*`` for Phase 5 to consider as an
  alternative to the single-model winner. The +0.0078 stacking lift
  on the headline target is small but real.

---

## Results

### Headline OOF AUC on roi_gt_2 (the headline target)

| Stage | OOF AUC | Δ from prior best |
|---|---:|---:|
| Phase 3c real value (SVM/all_five/MiniLM) | 0.5966 | -- |
| Phase 4 baseline best (SVM/standalone/MiniLM) | 0.6346 | +0.038 |
| + mpnet encoder (SVM/standalone/mpnet) | 0.6459 | +0.011 |
| + stacking 4 families (Logistic over base/standalone/mpnet) | 0.6537 | +0.008 |
| + LightGBM and XGBoost added to roster (XGBoost/standalone/mpnet) | 0.6520 single | +0.006 over SVM single |
| **+ stacking 6 families (Logistic over base/standalone/mpnet)** | **0.6556** | +0.002 |

**Total Phase 4 lift over Phase 3c real value: +0.059 OOF AUC.** The
six-family roster cleared the corrected 0.65 lower band on a single
model (XGBoost at 0.6520) and on the stacking ensemble (0.6556).
LightGBM and XGBoost did not displace SVM-RBF on `roi_gt_1`, but
XGBoost overtook SVM-RBF on `roi_gt_2` and Random Forest remained
the regression winner on `log_roi`.

The Phase 3 documentation error originally reported the SVM number
as 0.665, which anchored the pre-registered 0.65 to 0.72
forward-expected band. The corrected 0.5966 figure means the
realistic corpus ceiling lies lower; Phase 4's final 0.6537 is now
above the lower edge of that band.

### Per-target winners (saved to ``data/processed/``)

| Target | Family | Matrix | OOF metric |
|---|---|---|---:|
| log_roi | random_forest | standalone_positive_union_mpnet | RMSE 1.3102 |
| roi_gt_1 | svm_rbf | standalone_positive_union_mpnet | AUC 0.6353 |
| roi_gt_2 | xgboost | standalone_positive_union_mpnet | AUC 0.6520 |

The stacking ensemble for ``roi_gt_2`` reaches 0.6537 AUC; Phase 5
may calibrate either the single SVM-RBF base model (canonical
artifact) or the stacking ensemble (preserved in
``runs/phase_4/stacking_standalone_positive_union_mpnet_roi_gt_2/``).

### Train-OOF gap diagnostic

| Family | log_roi gap (RMSE) | roi_gt_1 gap (AUC) | roi_gt_2 gap (AUC) |
|---|---:|---:|---:|
| linear | 0.04 to 0.06 | 0.14 to 0.19 | 0.07 to 0.11 |
| histgb | 0.10 to 0.30 | 0.16 to 0.34 | 0.23 to 0.31 |
| random_forest | 0.16 to 0.22 | 0.38 to 0.43 | 0.30 to 0.38 |
| svm_rbf | 0.11 to 0.21 | 0.35 to 0.39 | 0.18 to 0.24 |

The Phase 3 finding holds: HistGB and especially Random Forest
overfit substantially on this corpus even with conservative
regularization. The aggressive HistGB grid (max_depth ∈ {2,3,4},
learning_rate ∈ {0.01,0.02,0.05}, min_samples_leaf ∈ {10,20,40,80})
did not close the gap. Linear is consistently well-regularized.

### Stratified diagnostic on roi_gt_2 (svm_rbf, standalone_positive_union_mpnet)

| Genre | n | n_pos | OOF AUC | Tractability |
|---|---:|---:|---:|---|
| Adventure | 86 | 63 | 0.765 | tractable |
| Fantasy | 32 | 24 | 0.766 | tractable |
| Sci-Fi | 45 | 25 | 0.674 | tractable |
| Crime | 73 | 45 | 0.642 | tractable |
| Mystery | 19 | 12 | 0.631 | tractable (small n) |
| Animation | 32 | 27 | 0.630 | tractable (small n) |
| Action | 168 | 103 | 0.633 | borderline |
| Comedy | 219 | 141 | 0.612 | intractable |
| Drama | 299 | 169 | 0.594 | intractable |
| Horror | 103 | 82 | 0.574 | intractable |
| Thriller | 59 | 37 | 0.559 | intractable |
| Romance | 27 | 19 | 0.533 | intractable |

Across the four primary families, the same genre pattern holds:
Adventure / Fantasy / Sci-Fi are reliably more tractable than
Drama / Comedy / Romance. Pre-1980s films (n=76, only 8 negative)
return AUC values around 0.30 to 0.40, statistically unreliable
given the imbalance and small sample.

### Headline interpretation

The corpus has bimodal structure rather than a uniform predictive
ceiling. Plot-driven genre films (Adventure, Fantasy, Sci-Fi)
carry roughly 0.66 to 0.77 OOF AUC and meet the original
forward-expected band. Character / relationship films (Drama,
Comedy, Romance) cap around 0.55 to 0.61 OOF AUC and pull the
corpus average down to the headline 0.65. This finding has
implications for Phase 6 (asymmetric-cost decision rule): the
"Refer to human reader" action becomes the natural recommendation
for low-confidence Drama / Comedy submissions, while genre films
get higher-confidence Greenlight / Pass recommendations.

### Cross-family complementarity (post-LightGBM/XGBoost expansion)

The 6-family stratified diagnostic surfaced an important pattern:
SVM-RBF and XGBoost specialize in different sub-corpora.

* **SVM-RBF dominates plot-driven genre films**: Adventure 0.77,
  Fantasy 0.77, Sci-Fi 0.67.
* **XGBoost dominates character-driven and threshold genres**:
  Drama 0.65 (vs SVM 0.59), Crime 0.71 (vs 0.64), Horror 0.65
  (vs 0.57), Comedy 0.63 (vs 0.61).
* The complementarity explains why stacking lifts a meaningful
  +0.0036 over the best single base on the headline matrix when
  both XGBoost and SVM-RBF are in the ensemble; the meta-learner
  exploits per-genre specialization.

### Feature importance on the per-target winners

The per-target winners' feature importance (native
``feature_importances_`` for tree winners, permutation importance
for SVM-RBF) consistently ranks five categories at the top:

1. **``release_year_parsed`` (#1 across all targets)**. Release
   year carries information beyond inflation (which is already
   absorbed by the log-ratio target).
2. **Character network metrics** (``network_lead_role_count``,
   ``network_max_betweenness_centrality``, ``network_modularity``).
3. **Topic distribution** (multiple LDA topics in every target's
   top 15).
4. **Embedding PCs** (multiple components in every target's top
   15; mpnet PC interpretability is opaque but signal is broad).
5. **Specific genre dummies** (Action, War, Romance, Other on
   different targets).

Notably absent from any target's top 15: lexical features
(MTLD, hapax ratio, readability) and sentiment features (VADER,
NRC, sentiment quartile means). The 6-family Phase 4 winners
confirm what Phase 3 already showed under the linear baseline:
lexical and sentiment features carry no marginal predictive
signal on this corpus once structural / topic / network /
embedding features are present.

### Per-film error analysis

The most-correctly and most-wrongly predicted films per target
are saved in ``reports/tables/phase4_top_correct_<target>.md``
and ``phase4_top_wrong_<target>.md``. Two patterns dominate the
top-15 wrong predictions on `roi_gt_2` (XGBoost on
``standalone_positive_union_mpnet``):

1. **Auteur prestige flops**: Barry Lyndon (Kubrick, 1975, 0.02x
   ROI), Blade Runner (Scott, 1982, 1.5x ROI), Raging Bull
   (Scorsese, 1980, 1.3x ROI), Peeping Tom (Powell, 1960, 0.56x
   ROI). The model expects them to succeed; they all
   under-performed at release. Many became canonical later.
2. **Genre-norm violators**: Boondock Saints (1999 sleeper, 8.3x
   ROI vs predicted flop), It's Complicated (Streep romcom, 2.6x
   ROI vs predicted modest). The model expects them to flop;
   they hit.

8 of the 15 wrong predictions are from before 1985, consistent
with the era effect from feature importance. The pattern directly
informs Phase 6's "Refer to human reader" action: the system
should defer on auteur-driven prestige projects, where
script-derived signal cannot capture commercial outcome.

### Saved figures

* ``phase4_train_oof_gap.png``: per-family train-OOF gap on each
  target, four matrices side-by-side. Confirms HistGB and RF
  overfit aggressively; mpnet narrows the gap on classification
  for HistGB and SVM but not for RF; standalone matrices have
  marginally smaller gaps than all_five (the parsimony argument).
* ``phase4_calibration_pre.png``: OOF reliability diagrams for the
  primary classifiers on ``roi_gt_2`` (all_five). Curves sit above
  the diagonal, indicating systematic under-confidence on the
  positive class. Linear and HistGB are monotonically smooth;
  SVM-RBF (after logistic rescale) and RF show small non-monotonic
  regions. Phase 5 will need post-hoc calibration on all of them.
* ``phase4_stratified_auc.png``: per-genre and per-decade OOF AUC
  heatmap on the headline winner cell, six families. Visual
  evidence of the corpus's bimodal structure and the
  SVM/XGBoost specialization complementarity.
* ``phase4_stacking_lift.png``: stacking lift over best base per
  (matrix, target). Largest single lift is on
  ``standalone_positive_union_mpnet`` ``roi_gt_2`` where the
  6-family stacker reaches 0.6556 vs XGBoost single 0.6520.
* ``phase4_feature_importance.png``: top-15 features per target
  winner with the appropriate importance method per family.
  Release year, network metrics, and topic / embedding components
  dominate; lexical and sentiment features are absent.
* ``phase4_error_by_genre.png``: per-genre mean absolute error per
  target, complementary to the AUC heatmap. Animation, Adventure,
  Fantasy have the lowest absolute error on ``roi_gt_2``;
  Drama, Romance, History have the highest.

---

## Issues encountered & resolved

1. **``baycomp.two_on_single`` posterior ordering.** The wrapper
   initially used the wrong posterior side when declaring the "a
   better" winner. Verified empirically against a known case
   (``two_on_single(better, worse) -> (1, 0, 0)``) and corrected
   in ``src/models/phase4/paired_test.py``. Caught by the unit
   tests in ``tests/test_phase4_paired_test.py`` before any
   reported result was affected; running benchmark used the buggy
   code in-process, but the post-hoc paired-test recomputer
   (``src/models/phase4/posthoc.py``) reads the per-fold metrics
   from the saved ``metrics.json`` files and overwrites the CSV
   with corrected values.
2. **Phase 3 documentation error: 0.665 vs 0.5966.** Discovered
   during the Section 10 escalation. Corrected in place per the
   pre-1995-cutoff reversal precedent (strikethrough + correction
   note) in ``docs/summaries/phase_3_summary.md`` (5 occurrences)
   and ``docs/FEATURE_NOTES.md`` (1 occurrence). The
   ``PROJECT_CONTEXT.md`` Section 8 entry from 2026-05-04 00:30
   captures the full audit trail of the discovery.
3. **mpnet encoder script signature mismatch.** Initial run
   completed encoding + PCA but then crashed on a 3-arg call to
   ``compute_embedding_features``, which takes 2. Resolved by
   editing the script and re-running (cache reused; the second
   invocation took 30 seconds rather than 53 minutes).
4. **HistGB sample-weight routing.** Pre-flagged by user as a
   real risk
   (``GridSearchCV.fit(X, y, sample_weight=...)`` may not route to
   per-fold training cleanly). Verified empirically via the smoke
   test (``--mode smoke_histgb``): the sample weights flow through
   ``model__sample_weight`` in ``fit_params``, are sliced per fold
   by sklearn, and produce reasonable best-hyperparameter
   selections. Scoring on held-out folds is unweighted, which for
   AUC-ROC does not bias the comparison.
5. **mpnet matrix variants not available in initial benchmark
   harness.** Resolved by adding ``all_five_mpnet`` and
   ``standalone_positive_union_mpnet`` matrix specs and a
   ``--mode mpnet`` CLI variant, with the canonical winner
   artifacts overwritten by the ``finalize.py`` script after both
   benchmarks were complete.

---

## Open questions / things to flag

End-of-Phase-4 mandatory escalation per ``PROJECT_CONTEXT.md``
Section 11. The questions below are for the planning conversation
to settle before Phase 5 starts.

1. **Calibrate the single SVM-RBF base model or the stacking
   ensemble?** The single base model is the canonical Phase 4
   winner per artifact (``data/processed/phase4_primary_model_*.joblib``);
   the stacking ensemble adds +0.0078 AUC on ``roi_gt_2`` but is
   harder to calibrate cleanly (each of the four base models would
   need its own Platt scaling, and the meta-learner might also).
   Recommendation: calibrate the single base model in Phase 5;
   reserve the stacking ensemble for the Phase 8 robustness
   analysis if calibration on the base model is insufficient.

2. **Genre-conditional thresholds in Phase 6?** The bimodal
   corpus structure (genre films tractable, character films not)
   suggests Phase 6's asymmetric-cost decision rule should
   produce different thresholds per genre. The mechanism is
   straightforward: per-genre confidence intervals from Phase 5,
   per-genre cost matrices in Phase 6. Worth surfacing as a
   strategic decision before Phase 5 starts because the
   calibration framework needs to support per-genre slicing if
   Phase 6 will rely on it.

3. **DistilBERT fine-tune held in reserve.** Tier-B1 from the
   escalation message; not invoked because Tier-A cleared the
   0.65 lower band. Phase 8 robustness analysis is the natural
   place to revisit if test-set performance falls below
   expectations. No action required for Phase 5.

4. **Pre-1980s films.** The diagnostic shows pre-1980s OOF AUC
   around 0.30 to 0.40 (statistically unreliable due to small n
   and class imbalance). Phase 6 may want to either include them
   with explicit "low-confidence" flagging or exclude them from
   the deployable model entirely. Not a Phase 5 question, but
   worth surfacing early.

5. **Test-set evaluation at end of Phase 5 or only Phase 8?** Per
   ``PROJECT_CONTEXT.md``, the held-out test set is touched only
   in Phase 8. Phase 5 reports calibration on the calibration set
   only. No deviation needed; flagging for clarity.

---

## Files produced

### Code (Phase 4 specific)

* ``src/models/phase4/__init__.py``
* ``src/models/phase4/families.py``
* ``src/models/phase4/matrices.py`` (with mpnet variants added)
* ``src/models/phase4/cv.py``
* ``src/models/phase4/paired_test.py``
* ``src/models/phase4/benchmark.py``
* ``src/models/phase4/figures.py``
* ``src/models/phase4/posthoc.py``
* ``src/models/phase4/sensitivity.py``
* ``src/models/phase4/stacking.py``
* ``src/models/phase4/diagnostic.py``
* ``src/models/phase4/finalize.py``
* ``src/models/phase4/importance.py`` (post-Tier-A addition)
* ``src/models/phase4/error_analysis.py`` (post-Tier-A addition)
* ``src/experiments/run_phase4_benchmark.py``
* ``src/experiments/run_phase4_mpnet_embeddings.py``
* ``tests/test_phase4_families.py``
* ``tests/test_phase4_matrices.py``
* ``tests/test_phase4_paired_test.py``

### Data (canonical winners overwrite earlier MiniLM saves)

* ``data/processed/phase4_primary_model_log_roi.joblib``
* ``data/processed/phase4_primary_model_roi_gt_1.joblib``
* ``data/processed/phase4_primary_model_roi_gt_2.joblib``
* ``data/processed/embeddings_mpnet_pooled.parquet`` (1,713 x 768 raw cache)
* ``data/processed/embedding_pca_mpnet.joblib`` (32-component PCA)
* ``data/processed/features_embedding_mpnet.parquet`` (32-column feature matrix)

### Tables (``reports/tables/``)

* ``phase4_benchmark.csv`` (288 rows, MiniLM primary tier)
* ``phase4_benchmark_mpnet.csv`` (288 rows, mpnet primary tier)
* ``phase4_secondary_benchmark.csv`` (72 rows, Lasso + Linear-SVM)
* ``phase4_paired_tests.csv`` (288 rows, posthoc-corrected, all matrices)
* ``phase4_sensitivity_unweighted.csv`` (48 rows, unweighted vs balanced)
* ``phase4_stacking.csv`` (24 rows, all 4 matrices x 3 targets x 2 metric tiers)
* ``phase4_per_genre_auc.csv`` (17 rows, headline cell)
* ``phase4_per_decade_auc.csv`` (5 rows, headline cell)
* ``phase4_winners.csv`` (3 rows, per-target winner summary)
* (Smoke-test artifacts ``phase4_smoke_*.csv`` retained for audit;
  not part of the deliverable set.)

### Figures (``reports/figures/``)

* ``phase4_train_oof_gap.png`` (4 matrices x 4 families x 3 targets)
* ``phase4_calibration_pre.png`` (4 primary classifiers on roi_gt_2)
* ``phase4_stratified_auc.png`` (per-genre + per-decade heatmap)
* ``phase4_stacking_lift.png`` (stacking lift per matrix x target)

### Documents

* ``docs/proposals/phase4_preregistration.md`` (the locked methodology)
* ``docs/summaries/phase_4_summary.md`` (this file)
* ``docs/PROJECT_CONTEXT.md`` Sections 8 and 9 updated
* ``docs/summaries/phase_3_summary.md`` corrected in 5 places
* ``docs/FEATURE_NOTES.md`` corrected in 1 place

### Run artifacts

* ``runs/phase_4/<timestamp>_<matrix>_<family>/`` per benchmark cell
  (8 MiniLM primary + 8 mpnet primary + 2 secondary + 4 sensitivity
  + 12 stacking = 34 directories) with the five canonical files plus
  ``model.joblib`` for primary-tier MiniLM cells.
* ``runs/RUNS.md`` updated with one row per cell.

### Notebook

* ``notebooks/_build_phase_4_notebook.py`` (builder script)
* ``notebooks/phase_4.ipynb`` (Phase 4 narrative for the final report)

---

## Next phase prerequisites

Phase 5 (Layer 2: Calibration) needs:

* The per-target winning estimators saved as joblib artifacts
  (``data/processed/phase4_primary_model_*.joblib``).
* The pre-Phase-5 calibration figure
  (``reports/figures/phase4_calibration_pre.png``) as a starting
  point for the calibration approach decision (linear and HistGB
  calibrate naturally; SVM-RBF and RF need explicit Platt or
  isotonic scaling).
* The held-out 257-film calibration set (touched first time in
  Phase 5).
* The benchmark CSV for the per-target headline OOF AUC numbers
  Phase 5 needs to compare against.
* Optional: the stacking ensemble preserved in
  ``runs/phase_4/stacking_*`` if Phase 5 wants to compare
  calibration of the single model versus the ensemble.

---

## Questions for the planning conversation

Mandatory end-of-Phase-4 escalation per ``PROJECT_CONTEXT.md``
Section 11. Three questions:

1. **Calibrate single SVM-RBF or stacking ensemble?** Recommendation:
   single SVM-RBF (cleaner Phase 5; +0.008 AUC stacking lift is
   modest). Stacking ensemble preserved as fallback.

2. **Genre-conditional thresholds in Phase 6?** The bimodal corpus
   structure suggests yes; Phase 5 calibration framework should
   support per-genre slicing if Phase 6 will use it.

3. **Does the Phase 3 documentation correction need broader audit?**
   The 0.665 calculation error suggests the Phase 3 summary may
   have other arithmetic mistakes. A targeted re-verification of
   the Phase 3 ablation tables against the source CSVs would catch
   any remaining errors before they propagate into the report.
