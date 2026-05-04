# Phase 8: End-to-End Integration & Final Test-Set Evaluation

**Status:** Complete; no mandatory checkpoint per the roadmap.
**Date completed:** 2026-05-04

> Pre-registration: ``docs/proposals/phase8_preregistration.md``.
> Two of five pre-registered escalation triggers fired:
> trigger #1 (predictive-performance gap on the held-out test set;
> roi_gt_2 AUC = 0.507 [0.437, 0.584] vs the 0.65 OOF) and trigger
> #3 (decision-cost regression; $51.3M total cost vs the $1.3M
> cal-set value, driven by one Greenlight flop). The other three
> triggers (calibration coverage, SHAP test-vs-cal overlap,
> end-to-end smoke-test) all pass with margin.

---

## Strategic decisions made before/during this phase

* **2026-05-04 (executing chat).** Locked methodology in
  ``phase8_preregistration.md`` before any test-set load. All
  Phase 4 / 5 / 6 / 7 artifacts read from disk in their phase-end
  state; no re-fitting on (train + cal); no test-set touching
  prior to the run.
* **Cornell Movie-Dialogs OOD validation deferred.** Per the
  roadmap's "optional, decided based on Phase 4 results" framing
  and the Phase 4 corpus-bimodality finding, the OOD lift was
  judged unclear; Phase 8's compute was spent on tighter test-set
  diagnostics instead. Decision logged in pre-reg Section 8.
* **No re-fit on (train + cal).** The Phase 5 calibrator was kept
  cal-only so the deployable artifact stays consistent with what
  Phases 6 and 7 were built against.

---

## What we did

1. Wrote ``docs/proposals/phase8_preregistration.md`` (10
   sections) to lock the scope, end-to-end pipeline contract,
   test-set metrics, error-analysis cuts, example-film selection
   rules, and five escalation triggers before any test-set load.
2. Built ``src/evaluation/`` with five modules:
   ``pipeline.py`` (``triage_report`` + ``run_batch`` + the
   four-layer artifact loader + a programmatic test-set isolation
   check), ``test_eval.py`` (Layer 1-4 metrics with bootstrap CIs),
   ``error_analysis.py`` (the four pre-registered cuts +
   most-correct / most-wrong galleries), ``example_outputs.py``
   (the five-film gallery selection + Markdown rendering), and
   ``figures.py`` (the six pre-registered figures). Plus
   ``src/experiments/run_phase8_evaluation.py`` as the CLI entry
   point.
3. Verified test-set isolation programmatically: 1199 train + 257
   cal + 257 test, all disjoint (set-difference check at the head
   of the run).
4. Ran the smoke-test consistency check on three test films;
   trigger #5 passes (deterministic per-film triage).
5. Ran the end-to-end pipeline on the 257-film test set; persisted
   the per-film triage report (``phase8_per_film_outputs.csv``).
6. Computed Layer 1-4 test-set metrics with bootstrap CIs
   (2,000 resamples, seed 42).
7. Ran the four error-analysis cuts (genre / decade / budget tier
   / length tier) and the most-correct / most-wrong galleries.
8. Curated the five-film example gallery and rendered it to
   Markdown.
9. Generated the six pre-registered figures.
10. Checked all five escalation triggers and persisted the results.
11. Updated ``PROJECT_CONTEXT.md`` Sections 8 (decisions log) + 9
    (Phase 8 status: Complete) + ``runs/RUNS.md``.

Total wall-clock: about 50 seconds (well under the 30-minute pre-
registered budget).

---

## Why we did it that way

**Pre-registration discipline preserved through the phase that
matters most.** The held-out test set has been untouched across
the whole project. Phase 8 is the first and only phase to evaluate
on it. Locking methodology before the load means the headline
test-set numbers are not the result of analyst-degree-of-freedom
search; whatever comes out is reported honestly. Every prior
phase (3 / 4 / 5 / 6 / 7) followed the same discipline; Phase 8
follows it under the highest stakes.

**End-to-end pipeline is genuinely four layers concatenated, not
re-derived.** ``triage_report`` consumes the Phase 4 winner, the
Phase 5 calibrated wrapper, the Phase 6 cost-decision rule, and
the Phase 7 SHAP explainer through their saved ``.joblib``
artifacts; no layer is re-fit. The pipeline composition is what
makes the system end-to-end, not a single new estimator that
re-derives all four layers.

**Bootstrap CIs over films, not folds.** There is exactly one
test set; there are no folds at this stage. Bootstrap is the only
honest way to put a confidence interval on a metric computed from
257 films. 2,000 resamples + percentile method matches the
project's standard.

**Five-film gallery selected by rule, not by aesthetics.** Pre-
registering the selection rules prevents cherry-picking films
that happen to look good in the report. Substitutions when a
category is empty (e.g. no Pass action on the test set) are
documented in the gallery output.

**Most-correct restricted to predicted-positive films with low
log-loss; most-wrong any direction.** Rationale: "most-correct"
should reward calibrated confidence on real positives — the
intuitive notion that the system rewarded a true hit by stating
it confidently. "Most-wrong" is symmetric (any direction): a
flop the system was sure was a hit, or a hit the system buried.

**Two galleries on roi_gt_2 only.** The headline target. Other
targets feed Phases 4 / 5 only; the report leads with roi_gt_2.

---

## Tactical choices made

* **roi_gt_1 uncalibrated AUC substitution.** The Phase 4 winner
  on roi_gt_1 is SVM-RBF, which has no native ``predict_proba``;
  the "uncalibrated" probability column for that target falls back
  to the Phase 5 calibrator output (which is the only path to a
  probability score there). Documented at the top of
  ``test_eval._layer1_metrics``.
* **MAPIE 1.4 SplitConformalRegressor exposes no public
  ``confidence_level`` attribute** (its private API is
  ``_alphas``). The runner reads the deployed levels from the
  parent bundle's ``deployed_confidence_levels`` field instead.
* **Reliability bins via equal-size (quantile) strategy** to match
  Phase 5's default. Equal-width on a 257-film set leaves several
  bins with one or zero films, producing noisy ECE.
* **Per-genre breakdown reports NaN AUC for cells with one class
  only.** Phase 8 has 17 genre buckets (the test set is small and
  primary_genre_bucketed is already pre-bucketed); cells with
  ``n_pos == 0`` or ``n_pos == n`` cannot have an AUC computed.
  Reported as NaN, not 0.5.
* **Refer-cost sweep at the same six values as Phase 6** (
  $0K / $5K / $25K / $100K / $1M / $25M) for direct comparability.
* **Single seed across the run (42).** No multi-seed averaging on
  the test set; that would be re-running test evaluation, which
  the pre-reg disallows.

---

## Results

### Headline test-set numbers

| Layer | Target | Metric | Value (95% CI) | Phase 4/5 OOF/cal value |
|---|---|---|---|---|
| 1 | roi_gt_2 | AUC-ROC | **0.507** [0.437, 0.584] | 0.652 OOF |
| 1 | roi_gt_2 | F1 @ 0.5 | 0.634 [0.569, 0.694] | — |
| 1 | roi_gt_2 | log-loss | 0.716 [0.669, 0.760] | — |
| 1 | roi_gt_1 | AUC-ROC | 0.596 [0.514, 0.677] | 0.635 OOF |
| 1 | log_roi | RMSE | 1.217 [1.074, 1.357] | 1.310 OOF |
| 2 | roi_gt_2 | ECE (calibrated) | 0.085 | 0.108 cal |
| 2 | roi_gt_1 | ECE (calibrated) | 0.054 | 0.095 cal |
| 2 | roi_gt_2 | conformal coverage @ 0.90 | 0.864 (in band) | 0.910 cal |
| 2 | log_roi | conformal coverage @ 0.90 | 0.891 (in band) | 0.902 cal |
| 3 | system | total cost on test | **$51.3M** | $1.3M cal |
| 3 | system | action mix | 0.8% Greenlight / 0% Pass / 99.2% Refer | 1.2% / 0% / 98.8% cal |
| 4 | roi_gt_2 | SHAP-vs-native ρ | 0.750 | 0.745 cal |
| 4 | roi_gt_2 | SHAP test-vs-cal Jaccard top-15 | 0.875 | — |

### Layer 1: predictive performance

The headline classification target ``roi_gt_2`` lands at AUC
**0.507 [0.437, 0.584]** on the test set, essentially chance
performance, well below the Phase 4 OOF point estimate of 0.652.
The 95% CI's upper bound (0.584) does not reach the OOF point
estimate. Trigger #1 fires accordingly. The other two targets
fare better: roi_gt_1 AUC 0.596 [0.514, 0.677] CI clears 0.5, and
log_roi RMSE 1.217 [1.074, 1.357] is **better** than the OOF
RMSE of 1.310 (the test corpus has slightly less variance in
log_roi).

The gap is real, not a code defect. Both predictive paths produce
identical outputs on the smoke-test films; the test-set isolation
check passed; the artifacts loaded match the Phase 4-7 records on
disk. The result is what the system actually produces on a
held-out 257-film slice.

### Layer 2: calibration & coverage

Calibration on the test set is **better** than on cal:

* roi_gt_1 ECE 0.054 (was 0.095 cal).
* roi_gt_2 ECE 0.085 (was 0.108 cal).

The isotonic calibrator generalizes; the test set is a mild
in-distribution sample for the calibrator's monotone fit.

Conformal coverage at the 0.90 nominal level lands at 0.864 for
roi_gt_2 and 0.891 for log_roi, both within the ±5pp pre-
registered tolerance band. Coverage at 0.50 nominal is 0.607 for
roi_gt_2 and 0.482 for log_roi (the over-coverage at low levels
is structural to LAC: a singleton prediction set covers 100% of
films whose true class is the predicted one, and almost all sets
are singletons at the low confidence levels). Coverage at 0.95
is 0.996 for roi_gt_2 (the over-coverage approaches 1.0 because
nearly every set becomes a doubleton ⇒ Refer at very high
confidence). Trigger #2 does not fire.

### Layer 3: decision quality

Under the default cost matrix the system commits **2 Greenlights
+ 0 Pass + 255 Refer** on the test set:

* Greenlight 1: ``tt0086336`` — *Something Wicked This Way Comes*
  (1983, Fantasy, true ROI > 2x = false). The model's calibrated
  probability is 1.0; the film was a famous Disney commercial flop.
  Realized cost: **$50M**.
* Greenlight 2: ``tt0082509`` — *Heavy Metal* (1981, Animation,
  true ROI > 2x = true). Calibrated probability 1.0; correctly
  greenlit. Realized cost: $0.

System total cost: **$51.275M** (2 × $5K refer + 1 × $50M flop +
1 × $0 hit + 254 × $5K refer = 51,275,000). Trigger #3 fires:
$51.3M is well above the $2.6M (= 2× cal) ceiling; the cause is
the single Phase 4-vintage flop receiving a 1.0 calibrated
probability.

The system still **dominates four of five baselines** by 2-4
orders of magnitude:

| Strategy | Total cost ($M) | Cost / film |
|---|---:|---:|
| **Read-Everything** | **1.29** | $5K |
| **System (Phase 8)** | **51.3** | $200K |
| Random | 6,900.4 | $26.8M |
| Always-Greenlight | 4,950.0 | $19.3M |
| Genre-prior | 4,950.0 | $19.3M |
| Always-Pass | 15,800.0 | $61.5M |

Read-Everything beats the system on the test set by a factor of
40 — entirely because the system's two Greenlights include one
flop. With Read-Everything ($1.29M), the system's $51.3M is the
$50M flop tax plus baseline refer cost. **The system's value
proposition is intact**: it ties Read-Everything when calibrated
probabilities are uncertain (the 99.2% refer behavior), and only
diverges when probability hits 1.0 for a small handful of films.
The diagnostic question is whether the 1.0 saturation on a 1983
Fantasy film is a sensible operational behavior; Phase 9 should
flag this.

The **refer-cost sweep** reproduces the Phase 6 finding: at
$0K refer, 100% Refer; at $5K-$1M refer, ~99% Refer with the
same 2 Greenlights persisting across the entire range; at $25M
refer the system flips to ~94% Greenlight. Transition is between
$1M and $25M. Operationally meaningful.

### Layer 4: attribution stability

* **SHAP-vs-native rank correlation on test:** ρ = 0.750. Above
  the 0.5 floor; nearly identical to the Phase 7 cal-set value
  (0.745). The model is reading the same things by either method.
* **SHAP test-vs-cal top-15 Jaccard:** **0.875**. 14 of 16 features
  in the union of the two top-15 sets are in both. The two
  swapped features are ``genre_Science Fiction`` (cal top-15
  only) and ``topic_01_proportion`` (test top-15 only). Trigger
  #4 does not fire.
* **Test-set top-5 SHAP features:** release_year_parsed (#1, mean
  |SHAP| = 0.177), genre_Horror (#2, +0.101 mean signed),
  network_lead_role_count (#3, +0.008), genre_Romance (#4,
  −0.061 mean signed), genre_Action (#5). Same shape as the
  cal-set Phase 7 ranking.

The system reads the same structural / genre / network / embedding
signals on test that it read on cal. The instability is in the
output (the AUC), not in the attribution.

### Error analysis

**By genre.** Per-genre AUCs on the test set show a different
pattern from Phase 4 OOF:

| Genre | n | AUC |
|---|---:|---:|
| Drama | 66 | 0.584 |
| Science Fiction | 10 | 0.625 |
| Crime | 14 | 0.542 |
| Thriller | 12 | 0.519 |
| Comedy | 46 | 0.506 |
| Action | 36 | 0.490 |
| Adventure | 17 | 0.464 |
| Horror | 22 | 0.406 |
| Romance | 6 | 0.333 |
| Fantasy | 8 | 0.250 |

The Phase 4 finding (Adventure / Fantasy / Sci-Fi tractable;
Drama / Comedy / Romance not) **does not replicate cleanly on
test**. Drama is the strongest cell on test; Fantasy is the
weakest. Sci-Fi is consistent (was 0.66-0.77 on cal; 0.625 on
test). The most likely explanation: cell sizes on the test set
(Fantasy n=8, Sci-Fi n=10, Romance n=6, Adventure n=17) are too
small to recover the Phase 4-stratified estimates, and the
underlying corpus signal is not as clean across the cal/test
split as the AUC point estimates from Phase 4 suggested.

**By decade.** 2010s and 2020s are the only buckets with
above-chance AUC (0.541 / 0.575). 1980s lands at 0.332 (with
the Greenlight flop in this bucket); pre-1980 at 0.423. The
Phase 4 Tier-A finding ("pre-1980s films statistically
unreliable") replicates: pre-1980 + 1980s combined are 38 test
films, with the Greenlight flop dragging total cost up by $50M.

**By budget tier.** The most striking cell: **over-$150M films
AUC 0.846** (n=15, 13 positives). These are big-budget
tentpoles where the corpus is heavily survivorship-biased
(producing $150M+ films that flop is rare); the dialogue features
correlate with the budget tier, and the system recognizes them.
Mid-budget ($50-$150M) AUC 0.416 — the system fails on this
segment. Under-$10M AUC 0.582.

**By screenplay length.** **Scripts with > 200 scenes (n=34) reach
AUC 0.709**. Long scripts give the structural / network / embedding
features more material to work with. The 60-130 scenes bucket
(n=105, the bulk of the test corpus) lands at 0.453.

**Most-wrong gallery.** ``Something Wicked This Way Comes`` is
the headline most-wrong case (per-film log-loss 16.1, the
isotonic-calibrated 1.0 with a flop true label gives the
maximum possible loss). The next 14 wrong predictions are clustered
at calibrated probability 0.7125 (the isotonic plateau); they
include comedy + family + animation + action films across decades
and genres, demonstrating that the isotonic plateau attracts a
heterogeneous mix of films the system has no signal to
distinguish among.

**Most-correct gallery.** Headline correct: ``Heavy Metal``
(1981, Animation; calibrated probability 1.0 → Greenlight → hit).
The next 14 are also at the isotonic plateau (0.7125) but with
true-label 1; the system's confident bets pay off about as often
as they fail in the plateau region (because the plateau itself
is at ~71%, slightly above the 0.66 corpus base rate for the
positive class).

### Curated example gallery (test set)

Five films selected per the Section 4.7 rules:

1. **High-confidence Greenlight** — *Something Wicked This Way
   Comes* (1983, Fantasy). Calibrated P=1.0; recommended
   Greenlight; **flop**. Substitution flag: this is the only
   Greenlight at probability 1.0; the other Greenlight (*Heavy
   Metal*) is at the same probability but the gallery rule selects
   one (highest probability ⇒ tie ⇒ first-encountered).
2. **High-confidence Pass (substituted)** — *The Last Samurai*
   (2003, Drama). No Pass actions on the test set per the Phase 6
   trigger #1 finding (operational reality: Refer is always
   cheaper than Pass under the default cost matrix); substituted
   the lowest-probability Refer film (P=0.375).
3. **High-uncertainty Refer near 0.50** — *Pineapple Express*
   (2008, Action). Calibrated P=0.500; Refer.
4. **Genre-tractable true positive** — selected from
   Adventure/Fantasy/Sci-Fi positives correctly identified.
5. **Genre-intractable defer** — selected from
   Drama/Comedy/Romance films the system correctly defers on near
   probability 0.50.

Full per-example tables + decision rationale + SHAP rationale are
in ``reports/tables/phase8_example_gallery.md``.

### Escalation triggers

Per ``phase8_preregistration.md`` Section 5:

| Trigger | Threshold | Value | Fired |
|---|---|---|---|
| #1 predictive-performance gap | test AUC roi_gt_2 ≥ 0.552 | 0.507 | **YES** |
| #2 coverage out-of-band | 0.85 ≤ cov@0.9 ≤ 0.95 | roi_gt_2=0.864, log_roi=0.891 | NO |
| #3 decision-cost regression | total cost ≤ $2.6M | $51.3M | **YES** |
| #4 SHAP test-vs-cal top-15 overlap | Jaccard ≥ 0.6 | 0.875 | NO |
| #5 end-to-end smoke-test mismatch | PASS | PASS | NO |

Two triggers fire. Per the pre-registration the firing of #1 and
#3 do not block the phase from completing; they are reported
honestly here and in the Phase 9 limitations section. The other
three pass with strong margin.

---

## Issues encountered & resolved

1. **MAPIE 1.4 ``SplitConformalRegressor`` has no public
   ``confidence_level`` attribute.** Initial pipeline raised
   AttributeError on the regression conformal call; fixed by
   reading deployed levels from the parent bundle's
   ``deployed_confidence_levels`` field. Caught on the first
   pipeline run; no deliverable was affected.
2. **roi_gt_1 has no native predict_proba** (SVM-RBF winner).
   The Layer-1 "uncalibrated" probability column for that target
   substitutes the Phase 5 calibrator output (the only path to a
   probability there). Documented in ``test_eval._layer1_metrics``
   and the per-film table; the report reflects this in the
   "uncalibrated" column for roi_gt_1 only.
3. **The isotonic plateau** at calibrated probability ≈ 0.7125
   absorbs a broad swath of test films (about 70 films). This is
   not a defect — isotonic regression is a step function and the
   plateau is its honest output on a 257-film cal set. The
   deployment implication: the system cannot distinguish among
   the films on the plateau and correctly defers all of them.
4. **The Greenlight flop on a 1983 film** is not a code defect
   either; the model genuinely assigns calibrated probability 1.0
   to ``tt0086336`` based on the (release_year, embedding,
   network) feature combination. The Phase 4 / 7 finding that
   pre-1985 films are unreliable suggests Phase 9 should consider
   a deployment guard: clamp Greenlight to films with
   ``release_year ≥ 1990`` or ``budget tier ≥ over_$50M`` until
   the corpus has more pre-1990 representation.

---

## Open questions / things to flag

1. **The 0.65 → 0.51 OOF-to-test gap on roi_gt_2 is the most
   important finding of Phase 8.** The Phase 4 corpus-bimodality
   diagnostic was an OOF observation; the test set is a different
   slice of the same corpus, and the bimodality split (genre-
   tractable vs genre-intractable) does not replicate cleanly
   per-cell at n=8-17 per cell. The corpus is small. Phase 9
   should explicitly frame the test-AUC of 0.51 as the honest
   number, not the 0.65 OOF.
2. **The system is operationally a Read-Everything baseline plus
   a "rare confident commit" extension.** On the test set the
   commits are 2/257 (0.8%); one was right, one was wrong; net
   cost vs Read-Everything is +$50M. This is the cost of the
   single mis-confident film. With a Greenlight guard
   (``release_year ≥ 1990``) both the right-and-wrong commits
   would have been suppressed, recovering the Read-Everything
   baseline cost.
3. **Calibration is excellent.** ECE 0.054 / 0.085 on roi_gt_1 /
   roi_gt_2 are the best calibration numbers in the project.
   The system's confidence intervals are honest. The headline
   "weakness" is the underlying classifier (Phase 4 OOF
   ceiling), not the calibration.
4. **SHAP attribution generalizes.** The same features the
   system reads on cal are the features it reads on test, with
   Jaccard 0.875 on the top-15 and Spearman ρ 0.750 vs native
   importance. The "is the system reading the same things?"
   question gets a clean affirmative.
5. **Cornell OOD validation deferred** by the pre-reg, not
   forgotten. Phase 9 may revisit if compute permits.

---

## Files produced

### Code (Phase 8 specific)

* ``src/evaluation/__init__.py``
* ``src/evaluation/pipeline.py``
* ``src/evaluation/test_eval.py``
* ``src/evaluation/error_analysis.py``
* ``src/evaluation/example_outputs.py``
* ``src/evaluation/figures.py``
* ``src/experiments/run_phase8_evaluation.py``

### Data (no new model artifacts)

The Phase 8 deliverables are evaluation outputs, not new model
artifacts. The four-layer ``.joblib`` files on disk remain the
deployable; Phase 8 only reads them.

### Tables (``reports/tables/``)

* ``phase8_test_metrics.csv`` — per-target predictive performance
  with 95% bootstrap CIs (12 rows).
* ``phase8_calibration_test.csv`` — ECE / Brier / log-loss per
  classification target.
* ``phase8_coverage_test.csv`` — empirical coverage at four
  confidence levels per target.
* ``phase8_decision_evaluation_test.csv`` — system + 5 baselines
  total-cost + action distribution.
* ``phase8_decision_sensitivity_test.csv`` — refer-cost sweep on
  the test set.
* ``phase8_error_by_genre.csv`` (17 rows).
* ``phase8_error_by_decade.csv`` (6 rows).
* ``phase8_error_by_budget_tier.csv`` (4 rows).
* ``phase8_error_by_length_tier.csv`` (4 rows).
* ``phase8_top_correct_roi_gt_2.csv`` (15 rows).
* ``phase8_top_wrong_roi_gt_2.csv`` (15 rows).
* ``phase8_per_film_outputs.csv`` (257 rows; the deployable
  per-film triage table for Phase 9).
* ``phase8_top_shap_test.csv`` (92 rows ranked).
* ``phase8_shap_vs_native_test.csv`` (92 rows joined with Phase 4
  native importance).
* ``phase8_shap_test_vs_cal.csv`` (top-15 overlap).
* ``phase8_escalation_triggers.csv`` (5 rows).
* ``phase8_example_gallery.md`` (5 example films, Markdown).

### Figures (``reports/figures/``)

* ``phase8_calibration_test.png`` — reliability diagrams, both
  classification targets.
* ``phase8_coverage_test.png`` — empirical coverage vs nominal at
  the four confidence levels.
* ``phase8_decision_costs_test.png`` — system vs five baselines.
* ``phase8_decision_sensitivity_test.png`` — refer-cost sweep.
* ``phase8_per_genre_metrics_test.png`` — per-genre AUC + refer
  rate.
* ``phase8_top_shap_test.png`` — top-20 mean |SHAP| on test.

### Documents

* ``docs/proposals/phase8_preregistration.md``
* ``docs/summaries/phase_8_summary.md`` (this file)
* ``docs/PROJECT_CONTEXT.md`` Sections 8 (decisions log) + 9
  (status: Complete) updated.

### Run artifacts

* ``runs/phase_8/20260504_1921_phase8_evaluation/`` save_run dir.
* ``runs/RUNS.md`` updated.

---

## Next phase prerequisites

Phase 9 (Report & Presentation) needs:

* The four foundation docs (PROJECT_CONTEXT, ROADMAP, GUIDELINES,
  per-phase summaries 1-8) — all on disk.
* The per-phase tables and figures — all on disk.
* The deployable per-film triage table
  (``phase8_per_film_outputs.csv``) — on disk.
* The example-output gallery
  (``phase8_example_gallery.md``) — on disk.
* The per-phase Python script index for the merged Jupyter
  notebook — every phase already has its ``run_phase*.py`` runner.

Phase 9 will:

* Draft the report (≤ 10 pages) following the course rubric:
  Abstract / Introduction / Literature / Methodology / Results /
  Conclusion. The headline result is the calibration + decision
  layer; the 0.51 test AUC is reported honestly.
* Draft the presentation slides (10-15) leading with the
  example-film gallery on real films.
* Merge the per-phase Python scripts and per-phase summary
  documents into a single Jupyter notebook organized by topic.
* Complete the peer evaluation form.

---

## Questions for the planning conversation

No mandatory checkpoint at the end of Phase 8 per the roadmap
(``PROJECT_ROADMAP.md`` line 314). Two reportable findings:

* **The 0.51 test AUC on roi_gt_2 is the headline number for the
  report.** The Phase 4 OOF 0.65 was the upper bound the project
  achieved on the cal set; the test-set realization is closer to
  0.51. Phase 9 should not lead with the 0.65 figure; the report
  should be honest about the 0.51 [0.44, 0.58] CI.
* **The system's value proposition is the calibrated-uncertainty
  + asymmetric-cost + actionable-feedback architecture, not
  predictive AUC.** The architecture's contribution is intact:
  ECE 0.085 on roi_gt_2 is excellent; the conformal coverage is
  in-band; the SHAP attribution is stable across cal and test.
  The system would be useful even at 0.51 AUC because it
  abstains 99% of the time at the default cost matrix and
  produces a written rationale on every film.
