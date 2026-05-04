# Phase 8 Pre-Registration: End-to-End Integration & Final Test-Set Evaluation

**Phase:** Phase 8 (Block C — Integration and Delivery)
**Status:** Pre-registered; locked before any test-set load
**Date:** 2026-05-04

This document fixes the Phase 8 end-to-end integration scope, the
final test-set evaluation protocol, the error-analysis cuts, the
example-output curation rules, and the escalation criteria **before
the held-out test set is touched for the first time**. Same
discipline as Phases 3 through 7.

The held-out 257-film test set has been untouched across the
project per ``PROJECT_CONTEXT.md`` Section 6 ("No data leakage").
Phase 8 is the first and only phase to evaluate on it. After Phase 8
the test-set numbers are frozen; nothing in Phase 9 (report &
presentation) revisits the test set with a new model or a new
threshold. Anything we wish we had measured but did not is reported
honestly as a limitation.

---

## 1. Purpose and scope

**In scope.**

1. Combine the four layers (Phase 4 prediction, Phase 5 calibrated
   probability + conformal interval, Phase 6 cost-asymmetric
   decision, Phase 7 SHAP attribution) into a single end-to-end
   pipeline function: **input** = a parsed screenplay (or, in the
   reproducible-runner path, a row of the saved Phase 3 feature
   matrix); **output** = the full triage report (decision + the
   action's rationale + calibrated probability + conformal interval
   + top-k SHAP contributors).
2. Final evaluation on the **held-out 257-film test set**:
    * Predictive performance metrics for all three targets
      (``log_roi`` regression, ``roi_gt_1`` and ``roi_gt_2``
      classification) on the un-calibrated Phase 4 winner.
    * Calibration coverage and probability-calibration ECE on the
      Phase 5 wrapper for the two classification targets, plus
      conformal-interval coverage and median width for ``log_roi``.
    * Decision-level cost (system vs Phase 6 baselines) under the
      default cost matrix and under the operationally meaningful
      refer-cost sweep.
    * SHAP attribution stability on the test set (test SHAP-vs-
      native rank correlation; test top-15 SHAP comparison to the
      cal-set top-15 from Phase 7).
3. Error analysis cuts on the test set: per-genre, per-decade,
   per-budget-tier, per-screenplay-length-tier. The cuts are pre-
   specified in Section 4.
4. A curated example-output gallery on well-known films from the
   test set, prepared for the Phase 9 presentation.
5. A simple results dashboard (HTML report or notebook) that lets
   reviewers inspect per-film decisions.

**Out of scope.**

* Any re-fit, re-tune, or re-selection of the Phase 4 winners,
  Phase 5 calibrators, Phase 6 cost matrix, or Phase 7 explainer.
  All artifacts are loaded from disk in their Phase-end state.
* Re-training any model on (train + cal). The held-out cal set is
  retained as the calibration set; the test set is the test set.
* Out-of-distribution validation on Cornell Movie-Dialogs. The
  roadmap (Section: Phase 8) flags this as **optional, decided
  based on time and Phase 4 results**. Per the Phase 4 corpus-
  bimodality finding (genre films tractable / character films not)
  the OOD value-add is unclear, and Phase 8's compute budget is
  better spent on tighter test-set diagnostics. We document the
  decision to defer in the Phase 8 summary; the Cornell loaders
  remain available for follow-up.

**Primary target.** The headline triage decision is ``roi_gt_2``
(per Phase 4 / 5 / 6 / 7). All four layers terminate on
``roi_gt_2``; ``log_roi`` and ``roi_gt_1`` carry through Phases 4
and 5 only.

---

## 2. Strategic decisions inherited (locked)

The Phase 8 evaluation reads, and does not revise, the following
choices made earlier in the project:

* **Per-target Phase 4 winners**, all on
  ``standalone_positive_union_mpnet`` (92 features):
    * ``log_roi`` — Random Forest (RMSE 1.3102 OOF).
    * ``roi_gt_1`` — SVM-RBF (AUC 0.6353 OOF).
    * ``roi_gt_2`` — XGBoost (AUC 0.6520 OOF).
* **Phase 5 calibration** chosen by ECE (classification) /
  coverage gap (regression):
    * ``roi_gt_1`` and ``roi_gt_2`` — isotonic + LAC split-conformal.
    * ``log_roi`` — absolute-residual split-conformal.
* **Phase 6 default cost matrix** (USD, sourced from
  ``PROJECT_CONTEXT.md`` Section 1):
    * Greenlight + flop: $50M (lost production budget).
    * Pass + hit: $100M (foregone revenue, conservative midpoint of
      the $100-$200M industry range).
    * Refer (any outcome): $5K (one human-reader pass).
* **Phase 6 decision rule.** Expected-cost minimization with tie-
  break to Refer. The deployable cost matrix is the global default;
  the per-genre variant is reported as a sensitivity check (Phase 6
  showed it does not lift).
* **Phase 7 explainer.** TreeSHAP (path-dependent) on the XGBoost
  ``roi_gt_2`` winner. SVM-RBF excluded from per-film SHAP; Phase 4
  permutation importance is the substitute. RandomForest TreeSHAP
  on ``log_roi`` is run as a secondary diagnostic.

---

## 3. End-to-end pipeline function (locked)

### 3.1 Function signature

```python
def triage_report(
    *,
    imdb_id: str,
    feature_row: pd.Series,           # already-extracted, 92 columns
    parsed_screenplay: ParsedScreenplay | None = None,
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
    confidence_levels: list[float] = [0.50, 0.80, 0.90, 0.95],
    top_k_shap: int = 5,
) -> TriageReport: ...
```

The function takes one film and returns a structured ``TriageReport``
dataclass with fields:

* ``imdb_id``, ``movie_name`` (resolved from the master corpus).
* ``calibrated_probability_roi_gt_2`` (float in [0, 1]).
* ``conformal_set_at_each_level`` — dict
  ``{level: {"lower": p_lo, "upper": p_hi, "set": [labels]}}``.
* ``log_roi_point_prediction`` and ``log_roi_interval_at_each_level``
  for the regression layer.
* ``recommended_action`` (Greenlight / Pass / Refer).
* ``expected_cost_per_action`` (Greenlight / Pass / Refer dollar
  values).
* ``decision_rationale`` (the Phase 6 rationale string).
* ``top_positive_shap_contributors`` — list of (feature, signed
  value) of length ``top_k_shap``.
* ``top_negative_shap_contributors`` — same format.
* ``shap_rationale`` — composed natural-language sentence joining
  Phase 6 + Phase 7 (the Phase 7 per-film rationale string format).

Pure-function semantics: no global state, no caching that depends
on previous calls. Determinism: identical inputs produce identical
outputs.

### 3.2 Reproducibility constraint

The end-to-end pipeline is exercised in **two modes** during Phase 8:

1. **Batch (test-set) mode**: feature rows are read from
   ``data/processed/features.parquet``, conformal/calibration
   wrappers are loaded once, and ``triage_report`` is applied per
   row. This is the path used to compute the headline test-set
   numbers.
2. **Single-film demo mode**: one example film is run through the
   raw screenplay path (``data/processed/screenplays_parsed.pkl``)
   to confirm the pipeline reproduces the batch result. This
   exercises the same code path as a deployed system would and is
   the proof that the assembled pipeline is genuinely end-to-end.

The batch mode produces the headline numbers; the single-film mode
is a smoke test on at least three example films.

---

## 4. Final test-set evaluation (locked)

### 4.1 Test set definition

* The 257-film test split per ``data/processed/split_assignments.parquet``
  (``split == "test"``). Stratified by ``primary_genre_bucketed`` ×
  ``decade_bucket`` with rare-cell pooling, fixed seed 42.
* No test-set film has ever been used for fitting, tuning, or
  selection of any Phase 4 / 5 / 6 / 7 artifact. Phase 8 verifies
  this assertion programmatically (set difference of test imdb_ids
  vs train + cal imdb_ids must be the test split itself).

### 4.2 Predictive performance (Layer 1)

For each target, compute the metric set established in Phase 3:

* **Regression (``log_roi``)**: MSE, RMSE, MAE, CVRMSE.
* **Classification (``roi_gt_1``, ``roi_gt_2``)**: AUC-ROC, PR-AUC,
  F1 at 0.5, log-loss.

Each metric is reported with a 95% bootstrap confidence interval
(2,000 resamples, seed 42, percentile method). Bootstrap is over
films, not folds — there is one test-set point estimate per metric.

### 4.3 Calibration & uncertainty (Layer 2)

For the two classification targets:

* **ECE** (10 equal-size bins, the Phase 5 default) on the test
  set, computed against the deployed isotonic-calibrated probability.
* **Reliability diagram** (one figure per target).
* **Brier score** and **log-loss** on the calibrated probability.
* **Conformal coverage** at 0.50 / 0.80 / 0.90 / 0.95: empirical
  coverage = fraction of test films whose true label is in the
  prediction set at the level. Pre-registered tolerance: ±5pp at
  the 0.90 nominal level. Out-of-band → escalation.
* **Singleton rate** at 0.90 (the Phase 5 trigger #2 metric):
  fraction of test films for which the conformal set is a single
  class.

For the regression target:

* **Conformal interval coverage** at the four levels and **median
  interval width**. Same ±5pp tolerance at 0.90.

### 4.4 Decision quality (Layer 3)

Evaluated on the test set under the default cost matrix:

* **System total cost** in USD.
* **Action distribution** (Greenlight / Pass / Refer percentages).
* **Cost-savings vs five baselines** (the Phase 6 comparators):
  Always-Greenlight, Always-Pass, Read-Everything, Random,
  Genre-prior. Reported as absolute total cost and relative
  multiplier.
* **Refer-cost sensitivity sweep** at $0K / $5K / $25K / $100K /
  $1M / $25M (same set as Phase 6). The transition point under
  which the system flips from "defer almost everything" to "commit"
  must reproduce the Phase 6 finding within ±10pp at each refer-
  cost level.

### 4.5 Attribution quality (Layer 4)

* **Test-set SHAP-vs-native rank correlation** (Spearman ρ) on the
  ``roi_gt_2`` XGBoost winner. Pre-registered floor: 0.5
  (matching the Phase 7 trigger #1).
* **Cal-set vs test-set SHAP top-15 overlap**: Jaccard similarity
  of the top-15 features (cal Phase 7 vs test Phase 8). Pre-
  registered floor: 0.6 ("the system is reading the same things on
  test that it read on cal"). Below 0.6 → flag in escalation.
* **Test-set top-10 mean |SHAP|** table for the report.

### 4.6 Error analysis cuts (locked)

The set is pre-specified before any test-set evaluation runs. Cuts
do not subset the test set further than is necessary to compute
the cut-level metric (e.g. per-genre AUC needs both classes in the
cell; cells with one class are reported as NA).

* **By primary genre (bucketed)**: per-genre AUC, F1, refer rate,
  realized total cost (default cost matrix).
* **By decade bucket** (pre-1980, 1980s, 1990s, 2000s, 2010s,
  2020s; same buckets as the stratified split): per-decade AUC,
  F1, refer rate.
* **By budget tier** (under $10M / $10-$50M / $50-$150M / over
  $150M, the four standard industry tiers): per-tier AUC, F1,
  refer rate, realized total cost.
* **By screenplay-length tier** (under 60 scenes / 60-130 / 131-
  200 / over 200 — quartile boundaries on the train split):
  per-tier AUC, F1.
* **Most-correct and most-wrong galleries** (top 15 each) for
  ``roi_gt_2``. Selection criterion: most-correct = lowest log-loss
  on the predicted positive; most-wrong = highest log-loss
  regardless of direction. Manually inspected for thematic patterns
  (auteur prestige flops vs genre-norm violators, the Phase 4
  finding).

### 4.7 Example-output gallery (locked)

Five films selected from the test set per the following rules,
locked before any test-set load:

1. **High-confidence Greenlight** — film with the highest
   calibrated probability that the system actually recommends
   Greenlight.
2. **High-confidence Pass** — if any exist; if not, the lowest-
   probability Refer.
3. **High-uncertainty Refer near 0.50** — the film closest to
   ``calibrated_probability == 0.5``.
4. **Adventure / Fantasy / Sci-Fi true positive** — a positive
   ``roi_gt_2`` film in the genre-tractable cluster correctly
   identified.
5. **Drama / Comedy / Romance defer** — a film in the genre-
   intractable cluster the system correctly defers on.

If a category has no eligible film on the test set, the closest-
matching film is documented and the substitution noted.

For each example: full ``TriageReport`` rendered as a Markdown
fragment (decision + probability + intervals + top SHAP contributors
+ rationale), plus film name and basic metadata.

---

## 5. Pre-registered escalation criteria (locked)

Five potential triggers, each requiring escalation to the planning
conversation if it fires. None of these block the phase from
completing; they require an honest write-up regardless of result.

1. **Predictive-performance gap.** Test-set ``roi_gt_2`` AUC is
   more than 10pp below the OOF value (0.6520 - 0.10 = 0.5520 floor).
   Probable cause: covariate shift between cal/train and test, or
   the corpus-bimodality finding intensifying at test time.
2. **Calibration coverage out-of-band.** Empirical coverage at the
   0.90 nominal level is outside [0.85, 0.95] on any of the three
   targets. Probable cause: the cal set was not representative of
   the test set (split-induced shift). Phase 9 reports the gap
   honestly; no re-calibration on test.
3. **Decision-cost regression vs cal.** The system's test-set
   total cost is more than 2× the cal-set total cost ($1.3M cal
   value); rule of thumb threshold is $2.6M test ceiling. Probable
   cause: a small handful of the 3 expected Greenlight decisions
   land on flops, swamping the $5K refer cost with $50M production
   losses.
4. **Test-vs-cal SHAP top-15 overlap below 0.6.** The system is
   "reading different things" on test than it did on cal. Probable
   cause: feature drift across the split. Reported as a limitation,
   does not change deployable.
5. **End-to-end smoke-test mismatch.** The single-film demo path
   produces a different ``TriageReport`` from the batch path on
   any of the three smoke-test films. This is a code defect, not a
   methodology defect, and must be fixed before the phase closes.
   No discretion: blocks completion.

If trigger #5 fires it is fixed in code; the other four are
reported in the Phase 8 summary and the Phase 9 report's
limitations section.

---

## 6. Output artifacts (locked)

### 6.1 Code

* ``src/evaluation/__init__.py``
* ``src/evaluation/pipeline.py`` — end-to-end ``triage_report``
  function and the batch-runner.
* ``src/evaluation/test_eval.py`` — final test-set metrics across
  the four layers.
* ``src/evaluation/error_analysis.py`` — the four error-analysis
  cuts and the most-correct / most-wrong galleries.
* ``src/evaluation/example_outputs.py`` — example-film selection
  + Markdown rendering of the gallery.
* ``src/evaluation/figures.py`` — the Phase 8 deliverable figures.
* ``src/experiments/run_phase8_evaluation.py`` — CLI entry point.

### 6.2 Tables (``reports/tables/``)

* ``phase8_test_metrics.csv`` — per-target predictive performance
  with bootstrap CIs (Layer 1).
* ``phase8_calibration_test.csv`` — ECE / Brier / log-loss / Spearman-
  on-conformal-coverage per target (Layer 2).
* ``phase8_coverage_test.csv`` — empirical coverage at all four
  confidence levels per target.
* ``phase8_decision_evaluation_test.csv`` — system + 5 baselines
  total-cost + action distribution under the default cost matrix.
* ``phase8_decision_sensitivity_test.csv`` — refer-cost sweep
  on the test set.
* ``phase8_error_by_genre.csv``
* ``phase8_error_by_decade.csv``
* ``phase8_error_by_budget_tier.csv``
* ``phase8_error_by_length_tier.csv``
* ``phase8_top_correct_roi_gt_2.csv`` — top-15 by lowest log-loss.
* ``phase8_top_wrong_roi_gt_2.csv`` — top-15 by highest log-loss.
* ``phase8_per_film_outputs.csv`` — full ``TriageReport`` for all
  257 test films (the deployable per-film table).
* ``phase8_shap_test_vs_cal.csv`` — top-15 overlap diagnostic.
* ``phase8_example_gallery.md`` — five curated examples in
  Markdown.

### 6.3 Figures (``reports/figures/``)

* ``phase8_calibration_test.png`` — reliability diagrams, both
  classification targets side-by-side.
* ``phase8_coverage_test.png`` — coverage-vs-nominal curves at all
  four confidence levels per target.
* ``phase8_decision_costs_test.png`` — system vs five baselines,
  log-scale bar chart matching the Phase 6 figure.
* ``phase8_decision_sensitivity_test.png`` — refer-cost sweep
  curve.
* ``phase8_per_genre_metrics_test.png`` — per-genre AUC and refer
  rate, two panels.
* ``phase8_top_shap_test.png`` — top-20 mean |SHAP| on test.

### 6.4 Documents

* ``docs/proposals/phase8_preregistration.md`` — this file.
* ``docs/summaries/phase_8_summary.md`` — postmortem using the
  standard Section 7 template.
* ``PROJECT_CONTEXT.md`` Sections 8 (decisions log) + 9 (phase
  status: Complete) updated.

### 6.5 Run artifacts

* ``runs/phase_8/<timestamp>_phase8_evaluation/`` save_run dir.
* ``runs/RUNS.md`` updated.

---

## 7. Compute budget

End-to-end on the 257-film test set is read-mostly: the Phase 4 +
5 + 6 wrappers all already-fit; only TreeSHAP on the test set
requires actual compute. From the Phase 7 timings (under 1 minute
on 257 cal films) the test-set TreeSHAP is comfortably under 5
minutes. Total Phase 8 wall-clock budget: **30 minutes** including
all figure generation and table writes. Budget overrun → escalate
without cutting the methodology.

---

## 8. Deviations from the roadmap

* **Cornell Movie-Dialogs OOD validation deferred** (roadmap
  Section: Phase 8 lists it as optional). Rationale: the Phase 4
  corpus-bimodality finding makes the OOD lift unclear, and
  Phase 8's compute is better spent on tighter test-set diagnostics
  per Sections 4.2 - 4.7. Cornell loaders remain available for a
  follow-up if Phase 9 review surfaces the need.
* **No re-fit on (train + cal).** Some tutorials in Phase-8-style
  evaluations refit the calibrator on (train + cal) before final
  test scoring; we keep the Phase 5 cal-only calibrator because
  re-fitting changes the artifact that Phase 6 / 7 were built
  against and would compromise the four-layer pipeline's
  consistency.

---

## 9. What is locked and what is not

**Locked** (no changes after this document is written):

* The four-layer artifact set (Phase 4 / 5 / 6 / 7 winners on disk).
* The 92-feature ``standalone_positive_union_mpnet`` matrix.
* The default cost matrix.
* The test-set predictive / calibration / decision / attribution
  metric set in Section 4.
* The four error-analysis cuts in Section 4.6.
* The five example-film selection rules in Section 4.7.
* The five escalation triggers in Section 5.
* The output artifact list in Section 6.

**Open / discretionary** (Claude Code resolves during execution
and documents in the summary):

* Plotting choices (colors, axis scales, titles), as long as the
  semantics match Section 6.3.
* Bootstrap implementation details (resample size 2,000 fixed; the
  weighting-by-genre alternative is documented as not used).
* The exact wording of the per-film rationale text (Phase 6 + 7
  templates concatenated; minor wording polish allowed).
* The Markdown rendering format of ``phase8_example_gallery.md``.

If a discretionary choice surfaces an unforeseen issue (e.g. a
test-set film breaks the parsed-screenplay loader), the issue is
documented in the summary's "Issues encountered & resolved"
section and a fix is shipped without re-opening the methodology.

---

## 10. End-of-phase deliverable

* All tables / figures / documents in Section 6 written.
* Five-film example gallery curated and rendered.
* Five escalation triggers checked; firing or not, each is
  reported.
* Phase 8 summary in ``docs/summaries/phase_8_summary.md``.
* ``PROJECT_CONTEXT.md`` Sections 8 (append-only decisions log
  entry) and 9 (Phase 8 status → Complete) updated.

No mandatory checkpoint at the end of Phase 8 per the roadmap
(``PROJECT_ROADMAP.md`` line 314: "results are what they are at
this point; discussion of meaning happens in Phase 9").
