# Phase 6: Layer 3 Asymmetric-Cost Decision

**Status:** Complete with one pre-registered escalation trigger fired
**Date completed:** 2026-05-04

> Pre-registration: ``docs/proposals/phase6_preregistration.md``.
> Trigger #1 (decision-rule degeneracy under default cost matrix)
> fires as anticipated in Section 4.1 of the pre-reg; the cost
> asymmetry mathematically forces Refer to dominate at any reasonable
> human-reader cost. The sensitivity sweep across refer costs is
> therefore the operationally relevant deliverable.

---

## Strategic decisions made before/during this phase

* **2026-05-04 (executing chat).** Locked methodology in
  ``phase6_preregistration.md`` before any decision evaluation
  ran. Cost-matrix defaults from ``PROJECT_CONTEXT.md`` Section 1
  ($50M flop / $100M miss / $5K refer). Decision rule:
  expected-cost minimization with tie-break to Refer. Sensitivity
  sweeps on cost asymmetry, refer cost, base magnitudes, per-genre
  cost matrices.
* **Phase 5 escalation Q1 resolution.** Probability-driven
  decision rule (not conformal-set-driven). The conformal set is
  reported as a per-film diagnostic but does not gate the action.
  Rationale: Phase 5's 78.6% refer rate at 0.90 conformal
  confidence is an artifact of fixed-confidence thresholds; the
  per-film expected-cost calculation is a more flexible signal
  for the decision.
* **Phase 5 escalation Q2 resolution.** Per-genre cost-matrix
  variant evaluated as a sensitivity, not the deployed default.
  The per-genre variant uses per-genre median budget / median
  revenue from the train split. The default deployment uses a
  global cost matrix with the option to switch on per-genre via
  the saved bundle's ``per_genre_cost_matrices`` slot.

---

## What we did

1. Wrote ``docs/proposals/phase6_preregistration.md`` (11 sections)
   anchoring all design choices before any evaluation ran.
2. Built ``src/decision/`` with seven modules:
   ``__init__.py``, ``cost_matrix.py`` (CostMatrix dataclass +
   pre-registered sweep variants), ``rule.py`` (per-film
   expected-cost decision with rationale string), ``baselines.py``
   (Always-Greenlight, Always-Pass, Read-Everything, Random,
   Genre-prior), ``evaluation.py`` (total-cost calculation),
   ``pipeline.py`` (orchestrator), ``figures.py`` (four
   pre-registered figures), plus ``src/experiments/run_phase6_decision.py``
   as the CLI entry.
3. Wrote ``tests/test_decision.py`` (14 tests) covering cost-matrix
   constants from the brief, decision-rule edge cases at p=0 and
   p=1, batch-length validation, evaluation total-cost correctness,
   sensitivity-variant set membership. **127 total project tests
   pass.**
4. Ran the full Phase 6 pipeline on the 257-film calibration set:
   default cost matrix + 13 sensitivity variants + 5 baselines +
   per-genre cost-matrix variant. Total wall-clock: under 5 seconds.
5. Generated all four pre-registered figures.
6. Persisted the decision-pipeline bundle to
   ``data/processed/phase6_decision_pipeline_roi_gt_2.joblib`` for
   Phase 8 consumption.
7. Updated ``PROJECT_CONTEXT.md`` Sections 8 (decisions log) and
   9 (phase status).

---

## Why we did it that way

**Expected-cost minimization is the canonical Bayes-decision
formulation under known costs.** Given calibrated probabilities
from Phase 5 and a sourced cost matrix from the brief, the
mathematically optimal decision per film is the action with
minimum expected cost. No further calibration or hyperparameter
selection is needed; the rule has zero free parameters once the
cost matrix is fixed.

**The cost asymmetry encodes the studio's economic reality.**
Greenlighting a flop loses ~$50M of production budget; passing on
a hit loses ~$100M of foregone revenue (the brief lists $100M-
$200M; we used the conservative $100M end). The asymmetry is the
project's central justification for the decision layer existing
at all.

**Refer cost is the operating-point knob.** With $5K refer cost
vs $50M-$100M error costs, Refer is cheaper than commit-and-be-wrong
across almost the entire probability range. The
operationally meaningful sensitivity is therefore the refer-cost
sweep, not the asymmetry sweep. A studio with constrained human-
reader capacity should set refer cost = (annual reader budget) /
(films reviewed per year) to recover their effective per-film
human-reader opportunity cost; the system will then naturally
target that refer rate.

**Tie-break to Refer is the conservative default.** When two
actions have equal expected cost (e.g., at the exact Refer-vs-
Greenlight crossover), the system defers to a human. This matches
the project's framing where the human reader is the failsafe.

---

## Tactical choices made

* **CostMatrix as a frozen dataclass** with ``expected_cost`` and
  ``realized_cost`` methods; sweep variants are instances of the
  same class with overridden fields.
* **Per-film rationale strings** built from a template per chosen
  action, embedded in the per-film output schema. Helpful for
  Phase 7 to extend with SHAP-driven explanations.
* **Genre-prior baseline** uses train-split positive base rate per
  genre, threshold 0.5. Approximates a "studio executive heuristic":
  greenlight Action / Adventure / Sci-Fi which historically
  positive-rate; pass Drama / Romance which historically negative-
  rate.
* **Per-genre cost matrices** computed once from per-genre median
  budget + median revenue on the train split (genres with ≥30
  films only); fall back to the global default for thinner genres.
* **save_run discipline** for the canonical Phase 6 run; the
  decision-pipeline bundle is saved both inside the run directory
  (audit trail) and at the canonical
  ``data/processed/phase6_decision_pipeline_roi_gt_2.joblib`` path
  (Phase 8 entry point).

---

## Results

### Default cost matrix (the headline)

| Strategy | Total cost (USD) | % Greenlight | % Pass | % Refer |
|---|---:|---:|---:|---:|
| **System** | **$1.3M** | **1.2%** | **0.0%** | **98.8%** |
| Read Everything | $1.3M | 0.0% | 0.0% | 100.0% |
| Always Greenlight | $5,000M | 100.0% | 0.0% | 0.0% |
| Always Pass | $15,700M | 0.0% | 100.0% | 0.0% |
| Random | $6,850M | 30.4% | 35.4% | 34.2% |
| Genre prior | $5,000M | 100.0% | 0.0% | 0.0% |

The system **ties Read-Everything at $1.3M total cost** and beats
every commit baseline by 3-4 orders of magnitude. The
genre-prior baseline ends up identical to Always-Greenlight because
every primary genre has a train-split positive rate above 50% (a
property of the survivorship-biased corpus).

### Trigger #1 fires as predicted

The system commits on only 1.2% of films (3 of 257) under the
default cost matrix. **All asymmetry variants** (1:1, 1:2, 1:4, 2:1
flop:miss) and **all base-magnitude variants** (×0.1, ×1, ×10)
produce identical action distributions: 1.2% Greenlight / 0% Pass
/ 98.8% Refer. The cost asymmetry alone does not change behavior
because Refer ($5K) is so much cheaper than any error.

The pre-registration anticipated this (Section 4.1): "the 'Refer
rate' is therefore controlled primarily by the **refer cost
parameter**, not by the model's probability output."

### Refer-cost sweep (the operationally meaningful sensitivity)

| Refer cost | % Greenlight | % Pass | % Refer | Total cost (M) |
|---:|---:|---:|---:|---:|
| $0K | 0.0% | 0.0% | 100.0% | $0.0 |
| $5K (default) | 1.2% | 0.0% | 98.8% | $1.3 |
| $25K | 1.2% | 0.0% | 98.8% | $6.4 |
| $100K | 1.2% | 0.0% | 98.8% | $25.4 |
| $1M | 1.2% | 0.0% | 98.8% | $254 |
| $25M | 95.3% | 0.0% | 4.7% | $4,950 |

The transition from "defer almost everything" to "commit almost
everything" happens between $1M and $25M refer cost. **At any
realistic per-film human-reader cost ($1K-$100K), the system
defers ~99% of films.** The interpretation: under the default
asymmetry, the system's value is in flagging every film for human
review at trivial marginal cost; the model's calibrated probability
informs which 1-2% of films can be safely committed without
human input.

### Per-genre action breakdown (default cost matrix)

| Genre | n | n_pos | % Greenlight | % Pass | % Refer |
|---|---:|---:|---:|---:|---:|
| Drama | 66 | 42 | 0% | 0% | 100% |
| Comedy | 47 | 28 | 2.1% | 0% | 97.9% |
| Action | 35 | 20 | 0% | 0% | 100% |
| Horror | 22 | 14 | 9.1% | 0% | 90.9% |
| Adventure | 18 | 14 | 0% | 0% | 100% |
| Crime | 14 | 9 | 0% | 0% | 100% |
| Thriller | 12 | 5 | 0% | 0% | 100% |
| Sci-Fi | 11 | 7 | 0% | 0% | 100% |

Horror has the highest Greenlight rate (9.1%), reflecting its high
positive base rate (14/22 ≈ 64%) combined with the calibrated
probability spike on a few high-confidence Horror films. Drama
defers 100%, consistent with the Phase 4 finding that Drama's OOF
AUC was the lowest among large genres.

### Per-genre cost-matrix variant

The per-genre variant produces total cost $1.3M (identical to
global default). **The per-genre tuning does not move the needle**
because Refer dominates under any reasonable cost matrix when the
underlying probability distribution clusters around 0.6-0.7 (the
positive base rate). This fires escalation #3 (no genre-tuning
lift) — the genre signal is already captured by the calibrated
probability.

### Saved figures

* ``phase6_baselines_comparison.png``: log-scale bar chart showing
  the system at $1.3M against Always-Pass at $15,700M; the value
  proposition of cost-asymmetric decision-making is immediately
  visible.
* ``phase6_cost_curve.png``: dual-axis sensitivity plot of total
  cost (red, log scale) and refer rate (blue) vs the refer-cost
  parameter (log scale, $0 to $25M). The transition between $1M
  and $25M refer cost is the operating-point selection diagnostic.
* ``phase6_action_distribution.png``: stacked-bar action proportion
  per cost-matrix variant. Visualizes the dominance of Refer
  across all variants except the $25M refer-cost extreme.
* ``phase6_per_genre_actions.png``: per-genre action proportion
  under the default cost matrix. Visual companion to Phase 5's
  ``phase5_refer_by_genre.png``.

---

## Issues encountered & resolved

1. **Genre-prior baseline degenerates to Always-Greenlight.** Every
   primary genre on the train split has a positive base rate above
   50% (the survivorship-biased corpus has 80% gross-profitable
   films overall, and even the lowest-base-rate genres are above
   the threshold). Reported as-is rather than retuning the
   threshold; the genre-prior baseline turns out to be the same as
   Always-Greenlight in the corpus's survivorship-biased regime.
2. **Tie-break test edge case.** First version of
   ``test_high_refer_cost_forces_commit`` used p=0.5 with refer
   cost $25M, which produces an exact Greenlight-vs-Refer tie at
   $25M; the tie-break-to-Refer rule then picked Refer, failing the
   test. Adjusted the test to use p=0.7 (Greenlight wins
   unambiguously at $15M < $25M < $70M).
3. **All cost-matrix variants in the asymmetry sweep produce
   identical action distributions.** The dominance of Refer over
   commit at any reasonable refer cost makes the asymmetry sweep
   non-informative. Reported as the headline result rather than
   redesigned; the refer-cost sweep is the operationally
   meaningful one.

---

## Open questions / things to flag

Phase 6 has **no** mandatory end-of-phase escalation per the
roadmap; the four intra-phase pre-registered triggers are noted
here for the planning conversation:

1. **Trigger #1 (decision-rule degeneracy) FIRES.** System
   commits on only 1.2% of films under the default cost matrix.
   **Not a methodology defect** — the cost asymmetry mathematically
   forces this outcome. The deployed system's value is in (a)
   flagging the 1-2% high-confidence commits for which a human
   reader's time would be wasted, (b) providing the calibrated
   probability + conformal interval to inform the human reviewer's
   read of the other 98%, and (c) giving studios a refer-cost
   parameter to trade off human-reader budget against expected
   error losses.
2. **Trigger #3 (no genre-tuning lift) FIRES.** Per-genre cost
   matrices produce identical total cost to the global default
   ($1.3M each). The genre signal is already captured by the
   calibrated probability; per-genre cost matrices add no value.
3. **Triggers #2 and #4 do NOT fire.** System beats Random by 4
   orders of magnitude; refer-cost sweep produces a smooth cost
   curve (no discontinuities).
4. **For Phase 7 (SHAP)**: the per-film rationale strings already
   in the decision pipeline output are the natural anchor for
   SHAP attribution. Phase 7 should extend the rationale with
   per-feature contributions ("Greenlight recommended; the model's
   probability of 0.85 is driven by feature X (+0.12), feature Y
   (+0.08), feature Z (+0.06)").
5. **For Phase 8 (test-set evaluation)**: the decision pipeline
   bundle at ``data/processed/phase6_decision_pipeline_roi_gt_2.joblib``
   is the deployable entry point. Phase 8 will re-run the same
   decision rule on the held-out 257 test films (untouched in any
   prior phase). The cost-savings comparison vs naive baselines
   will be the headline Phase 8 result.

---

## Files produced

### Code (Phase 6 specific)

* ``src/decision/__init__.py``
* ``src/decision/cost_matrix.py``
* ``src/decision/rule.py``
* ``src/decision/baselines.py``
* ``src/decision/evaluation.py``
* ``src/decision/pipeline.py``
* ``src/decision/figures.py``
* ``src/experiments/run_phase6_decision.py``
* ``tests/test_decision.py``

### Data

* ``data/processed/phase6_decision_pipeline_roi_gt_2.joblib``
  (deployable bundle; Phase 8 entry point)

### Tables (``reports/tables/``)

* ``phase6_decisions.csv`` (257 rows: per-film decision under default)
* ``phase6_baselines.csv`` (6 rows: 5 baselines + system)
* ``phase6_sensitivity.csv`` (13 rows: 12 cost-matrix variants + per-genre)
* ``phase6_per_genre_actions.csv`` (17 rows: per-genre action breakdown)

### Figures (``reports/figures/``)

* ``phase6_baselines_comparison.png``
* ``phase6_cost_curve.png``
* ``phase6_action_distribution.png``
* ``phase6_per_genre_actions.png``

### Documents

* ``docs/proposals/phase6_preregistration.md``
* ``docs/summaries/phase_6_summary.md`` (this file)
* ``docs/PROJECT_CONTEXT.md`` Sections 8 (decisions log) + 9
  (phase status) updated

### Run artifacts

* ``runs/phase_6/<timestamp>_phase6_decision/`` with the five
  canonical save_run files plus the decision pipeline as
  ``model.joblib``.
* ``runs/RUNS.md`` updated.

---

## Next phase prerequisites

Phase 7 (Layer 4: SHAP Explanations) needs:

* The Phase 4 winner artifacts (already on disk for SHAP feature
  attribution).
* The Phase 5 calibrated wrappers (for calibration-aware attribution
  if useful).
* The Phase 6 decision pipeline for embedding SHAP-driven
  per-feature contributions into the per-film rationale strings.
* The held-out test set untouched.

Phase 7 will:

* Run TreeSHAP on the XGBoost roi_gt_2 winner (native support).
* Compute SHAP for SVM-RBF and RandomForest via permutation /
  KernelSHAP.
* Produce per-film attribution: which features drove the
  recommended action.
* Optionally: scene-level attribution via per-scene feature
  recomputation or scene-removal counterfactuals.
* Validate scene-level attributions are stable across model variants.

---

## Questions for the planning conversation

Phase 6 has no mandatory end-of-phase escalation. Two informal
questions worth flagging:

1. **Operating point for the deployed system.** What refer cost
   should the canonical pipeline ship with? The default $5K
   yields 99% refer, which means the system flags every film for
   human review (low risk, high human-reader cost). Setting refer
   cost to a higher value (e.g., $1M-$5M) would force the system
   to commit on a meaningful fraction of films. The right answer
   depends on the deployed studio's human-reader capacity. Ship
   with $5K and a documented sensitivity sweep; let downstream
   users tune to their context.
2. **Phase 7 scope.** Scene-level SHAP is the most ambitious part
   of the original brief and the project's most novel contribution.
   Worth surfacing whether to invest the time on scene-level
   attribution (potentially 4-6 hours) or stay at feature-level
   (1-2 hours). My recommendation: feature-level first, then
   scene-level if time permits.
