# Phase 6 Pre-Registration: Layer 3 Asymmetric-Cost Decision

**Phase:** Phase 6 (Layer 3 of the four-layer triage system)
**Status:** Pre-registered; locked before any decision-rule fit
**Date:** 2026-05-04

This document fixes the Phase 6 cost-matrix structure, decision rule,
sensitivity-analysis protocol, baseline comparisons, and escalation
criteria **before any decision evaluation runs against the
calibration set**. Same discipline as Phases 3, 4, and 5.

The Phase 5 close (2026-05-04 13:30) and the
``phase5_calibrated_model_*.joblib`` artifacts establish the inputs.
Phase 6 reads the calibrated probability ``P(roi_gt_2 | features)``
and (optionally) the conformal prediction set as inputs to the cost-
based decision rule. The Phase 5 escalation Q1 (probability-driven
vs conformal-driven trigger) and Q2 (per-genre vs single threshold)
are resolved here in Section 4.

---

## 1. Purpose and scope

**In scope.** Convert the Phase 5 calibrated probability into one of
three actions (**Greenlight**, **Pass**, **Refer to human reader**)
by minimizing expected cost under a sourced cost matrix. Sensitivity-
analyze the decision behavior across cost-matrix variants. Compare
the system's total cost against four naive baselines. Surface
per-genre triage breakdowns. Persist a decision-pipeline artifact
that, given a screenplay, returns the action plus an itemized
rationale.

**Out of scope.** Re-fit of any Phase 4 model. Re-calibration of any
Phase 5 wrapper. Test-set evaluation (Phase 8 only). Construction of
any per-character or scene-level rationale text (those are Phase 7's
SHAP layer).

**Primary target.** ``roi_gt_2`` only. The other two targets
(``log_roi`` regression, ``roi_gt_1`` classification) feed Phase 6
only as auxiliary signals if useful (the regression target's
conformal interval can refine the per-film expected revenue
estimate; tested as a sensitivity variant). The headline decision
target stays ``roi_gt_2``.

---

## 2. Strategic decisions inherited (locked)

* **Phase 5 calibrated wrappers are the input.** Three artifacts at
  ``data/processed/phase5_calibrated_model_*.joblib``. The headline
  one for Phase 6 is ``phase5_calibrated_model_roi_gt_2.joblib``
  (XGBoost on ``standalone_positive_union_mpnet``, isotonic-
  calibrated, conformal-wrapped at 0.50 / 0.80 / 0.90 / 0.95).
* **Calibration set is the evaluation surface.** 257 films,
  ``split == "cal"`` in ``split_assignments.parquet``. Already used
  by Phase 5 to fit the calibration mapping; Phase 6 fits no
  additional probability calibration, so re-using the same 257 films
  for cost-decision evaluation is honest. The held-out 257-film
  test set stays untouched until Phase 8.
* **Phase 5 escalation Q1 (probability vs conformal trigger).**
  Resolution: **probability-driven** decision rule. The conformal
  set is reported as a diagnostic in the per-film output but does
  not gate the action. Rationale: Phase 5 demonstrated that the
  conformal set at 0.90 confidence drives a 78.6% refer rate, which
  defeats the system's value as a triage tool. The calibrated
  probability is the more flexible signal: per-film expected costs
  can place a film in Greenlight / Pass / Refer based on the cost
  matrix's actual asymmetries, not on a fixed-confidence threshold.
* **Phase 5 escalation Q2 (per-genre thresholds).** Resolution:
  **per-genre cost-matrix variants** evaluated as a sensitivity
  analysis. The default deployment uses a single global cost matrix.
  The per-genre variant is reported alongside as evidence of how
  much per-genre tuning would help.

---

## 3. The cost matrix (locked)

Binary classification of ``roi_gt_2`` (1 = film makes ≥2x its
budget; 0 = film does not). Three actions: Greenlight, Pass, Refer.
Cost is in USD, all values negative as expected loss (positive cost
means worse outcome).

### 3.1 Default cost matrix

|  Action ↓ / True → | 0 (flop, ROI<2) | 1 (hit, ROI≥2) |
|---|---:|---:|
| **Greenlight** | $50,000,000 | $0 |
| **Pass** | $0 | $100,000,000 |
| **Refer** | $5,000 | $5,000 |

Sources from ``docs/PROJECT_CONTEXT.md`` Section 1:

* **$50M cost of greenlighting a flop** — typical mid-budget feature
  loss when the production goes ahead and revenue does not at least
  cover the budget. This is the headline number from the brief.
* **$100M cost of passing on a hit** — opportunity cost of the
  studio not producing a film that would have made 2x or more
  revenue/budget. The brief lists $100M-$200M; we use $100M as the
  conservative-default value. The 1:2 cost ratio (flop:miss) is
  the key asymmetry the entire architecture absorbs.
* **$5K cost of human reader review** — senior development reader
  rate (~$200-$500/hour) × half-day to read and write coverage
  (~$1K-$3K) plus organizational overhead (~$1K-$2K). The brief
  describes human readers as "expensive but limited in throughput."
  $5K is mid-range.

### 3.2 Pre-registered sensitivity sweeps

| Sweep | Variants | Reason |
|---|---|---|
| **Cost asymmetry** | flop:miss = 1:1, 1:2 (default), 1:4, 2:1 (inverted) | Tests robustness to the asymmetry assumption; the 2:1 inverted variant is the diagnostic for "what if studios cared more about not greenlighting flops than not missing hits" |
| **Refer cost** | $0K, $5K (default), $25K, $100K, $1M | Tests the "human reader budget" effect; high refer cost forces the model to commit, low refer cost lets it defer aggressively |
| **Base magnitudes** | scale all costs ×0.1, ×1, ×10 | Sanity check; decision rule is scale-invariant in expected-cost minimization, so this should produce identical actions (Phase 6 verifies this empirically) |
| **Per-genre asymmetries** | default + per-genre flop:miss ratios derived from per-genre median-budget × per-genre median-revenue (when ≥30 films) | Tests whether per-genre tuning materially improves total cost; the per-genre figure from Phase 5 already showed the model's confidence varies by genre, so cost matrices probably should too |

The pre-registered set is locked at four sensitivity dimensions and
not expanded after seeing results.

---

## 4. The decision rule (locked)

For each film with calibrated probability ``p = P(roi_gt_2 = 1)``,
compute expected cost per action:

```
E[cost | Greenlight] = (1 - p) × C_flop_greenlight + p × C_hit_greenlight
                     = (1 - p) × $50M + p × $0
                     = (1 - p) × $50M

E[cost | Pass]       = (1 - p) × C_flop_pass + p × C_hit_pass
                     = (1 - p) × $0 + p × $100M
                     = p × $100M

E[cost | Refer]      = (1 - p) × C_flop_refer + p × C_hit_refer
                     = $5K  (independent of true label by construction)
```

**Choose action minimizing expected cost.** Tie-break to Refer
(conservative; pushes uncertain films to human review).

### 4.1 What the rule produces with the default cost matrix

Crossover thresholds:
* **Greenlight vs Pass break-even**: ``(1-p) × $50M = p × $100M`` →
  ``p = 1/3 ≈ 0.333``. Above this threshold, Greenlight is preferred
  to Pass.
* **Refer vs Greenlight break-even**: ``(1-p) × $50M = $5K`` →
  ``p = 1 - 5K/50M = 0.9999``. Refer is preferred to Greenlight
  unless the model is essentially certain it's a hit.
* **Refer vs Pass break-even**: ``p × $100M = $5K`` →
  ``p = 5K/100M = 5e-5 = 0.00005``. Refer is preferred to Pass
  unless the model is essentially certain it's a flop.

**Implication of the default cost matrix.** Refer dominates the
expected-cost calculation across most of the probability range.
This is the **honest mathematical consequence** of the cost
asymmetry: when getting it wrong costs $50M-$100M and asking a human
costs $5K, deferring is almost always cheaper. The "Refer rate" is
therefore controlled primarily by the **refer cost parameter**, not
by the model's probability output. The sensitivity sweep across
refer costs is therefore the most operationally relevant dimension.

### 4.2 Per-film output schema

For each film, the decision pipeline returns:

```
{
  "imdb_id": ...,
  "calibrated_probability": float,
  "expected_costs": {"Greenlight": float, "Pass": float, "Refer": float},
  "recommended_action": "Greenlight" | "Pass" | "Refer",
  "conformal_set_at_0.9": [0] | [1] | [0, 1] | [],   # diagnostic
  "rationale": <one-sentence string explaining the recommendation>,
  "cost_matrix_version": "default" | "<sensitivity-variant-name>",
}
```

The rationale string follows a template:

* For Greenlight: "Recommended Greenlight: model probability {p} of
  ≥2× ROI yields expected loss ${e_g/1e6}M from Greenlight vs
  ${e_p/1e6}M from Pass."
* For Pass: parallel template, swapping action.
* For Refer: "Recommended Refer to human reader: at probability
  {p}, expected losses from Greenlight (${e_g/1e6}M) and Pass
  (${e_p/1e6}M) both exceed the human-reader cost (${refer_cost/1e3}K).
  Manual review is preferred."

---

## 5. Comparison baselines (locked)

Five baselines on the same 257-film calibration set:

| Baseline | Description |
|---|---|
| **Always Greenlight** | Greenlight every film regardless of features |
| **Always Pass** | Pass every film |
| **Read Everything** | Refer every film to human reader |
| **Random** | Choose Greenlight / Pass / Refer with equal probability per film, seeded |
| **Genre prior** | Greenlight if film's genre has ≥50% positive base rate on the train split (Adventure, Action, Sci-Fi); Pass otherwise (Drama, Romance, Comedy); never refer. A reasonable "studio executive heuristic" baseline. |

For each baseline + the system, report total cost on the calibration
set under the default cost matrix and the cost ratio vs the system.

---

## 6. Per-genre triage diagnostic (locked)

Per Phase 5's per-genre refer figure: report per-genre action
breakdown (% Greenlight / % Pass / % Refer) under the default cost
matrix. Expected pattern: tractable genres (Adventure, Sci-Fi,
Crime, Action) get more Greenlight/Pass commits; intractable
genres (Drama, Romance, Comedy) get more Refer.

If the per-genre cost-matrix sensitivity sweep (Section 3.2 last
row) shows per-genre tuning materially improves total cost (>10%
improvement vs global), the deployed pipeline supports a
``cost_matrix_per_genre`` mode. Otherwise, the global default is
the canonical deployment.

---

## 7. Pre-registered escalation criteria (locked)

Each criterion below triggers a planning-conversation flag (no
unilateral resolution by the executing chat). Phase 6 has **no**
mandatory end-of-phase escalation per the roadmap, but the four
intra-phase triggers below stand.

1. **Decision-rule degeneracy**. Under the default cost matrix, the
   system always picks the same action (e.g., Refer everything).
   Indicates the cost matrix is mis-specified or the calibrated
   probability does not span enough range to differentiate. Surface
   the empirical p distribution and recommend a refer-cost
   adjustment.
2. **Random baseline beats the system**. The system's total cost is
   higher than the Random baseline on the calibration set.
   Indicates a methodology bug.
3. **No genre-tuning lift**. The per-genre cost-matrix variant
   produces total cost within 1% of the global default, even though
   the per-genre AUC and refer rate vary substantially. Indicates
   the genre signal is already captured by the calibrated
   probability and per-genre tuning is not needed.
4. **Refer cost discontinuity**. The system's total cost vs refer
   cost has a sharp jump at any sensitivity point (e.g., total cost
   doubles when refer cost goes from $5K to $25K). Indicates the
   decision rule is fragile in the relevant operating range.

---

## 8. Output artifacts (locked)

### 8.1 Per-film decision artifact

``data/processed/phase6_decision_pipeline_roi_gt_2.joblib`` — a
dict containing:

* ``cost_matrix``: the default cost matrix dict.
* ``calibrated_wrapper``: reference to
  ``phase5_calibrated_model_roi_gt_2.joblib``.
* ``decision_function``: callable (X, cost_matrix) → list of dicts
  with the per-film schema from Section 4.2.
* ``per_genre_cost_matrices``: dict (mode for the genre variant).

### 8.2 Tables

* ``reports/tables/phase6_decisions.csv``: per-film row from the
  default cost matrix (calibration set; 257 rows).
* ``reports/tables/phase6_baselines.csv``: total cost per baseline
  vs the system, per cost-matrix variant.
* ``reports/tables/phase6_sensitivity.csv``: total cost + action
  distribution per cost-matrix variant in the sweep (Section 3.2).
* ``reports/tables/phase6_per_genre_actions.csv``: per-genre action
  breakdown under the default matrix and (if it lifts) the per-genre
  variant.

### 8.3 Figures

* ``phase6_cost_curve.png``: total cost as a function of refer cost
  (the most operationally relevant sensitivity sweep).
* ``phase6_action_distribution.png``: stacked-bar plot of action
  proportion per cost-matrix variant.
* ``phase6_per_genre_actions.png``: per-genre stacked-bar of
  action proportion under the default cost matrix; companion to
  ``phase5_refer_by_genre.png``.
* ``phase6_baselines_comparison.png``: bar chart of total cost per
  baseline + the system.

### 8.4 Documentation

* ``docs/summaries/phase_6_summary.md`` per the Section 7 template.
* ``docs/PROJECT_CONTEXT.md`` Sections 8 (decisions log) and 9
  (phase status) updated.
* ``runs/phase_6/<timestamp>_phase6_decision/`` save_run directory.

---

## 9. Compute budget

Phase 6 is fast. The decision rule is a closed-form expression over
257 films per cost-matrix variant. Estimated wall-clock for the
full Phase 6 run including all sensitivity variants: **under 5
minutes**.

| Step | Estimate |
|---|---|
| Load Phase 5 wrapper + cal-set features | <1 minute |
| Default cost matrix evaluation + per-film decisions | <30 seconds |
| Sensitivity sweep (4 dimensions × ~5 variants each) | <2 minutes |
| Baseline comparisons | <1 minute |
| Figure rendering | <1 minute |

---

## 10. Deviations from the roadmap

One deliberate deviation:

1. **Test-set evaluation explicitly deferred to Phase 8.** The
   roadmap's Phase 6 section mentions "system-level evaluation
   metrics" but does not specify which set. Per
   ``PROJECT_CONTEXT.md`` Section 6, the held-out test set is
   touched once in Phase 8 only. Phase 6 evaluates on the
   calibration set; Phase 8 will re-run the decision pipeline on
   the test set as part of the end-to-end evaluation.

---

## 11. What is locked and what is not

**Locked.** Cost matrix structure (Section 3), decision rule
(Section 4), sensitivity sweep (Section 3.2), baselines (Section 5),
per-genre diagnostic (Section 6), escalation criteria (Section 7),
output artifacts (Section 8).

**Not locked (tactical).** Helper-function structure, plot
styling, per-film rationale string formatting, table column order,
log verbosity, code organization within ``src/decision/``.

**Pre-registration discipline.** The set above is not expanded
after seeing results. If MAPIE / sklearn API surprises surface,
they are documented as Section 11 deviations, not silent revisions.
