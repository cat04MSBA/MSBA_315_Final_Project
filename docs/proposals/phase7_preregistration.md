# Phase 7 Pre-Registration: Layer 4 SHAP Explanations

**Phase:** Phase 7 (Layer 4 of the four-layer triage system)
**Status:** Pre-registered; locked before any SHAP attribution
**Date:** 2026-05-04

This document fixes the Phase 7 attribution methodology, scene-
level approach, stability-validation protocol, output schema, and
escalation criteria **before any SHAP value is computed against the
calibration set**. Same discipline as Phases 3 through 6.

The Phase 4 winners (XGBoost / RF / SVM-RBF), Phase 5 calibrators,
and Phase 6 decision pipeline together establish the inputs.
Phase 7 adds the per-film and scene-level explanation layer that
makes the system actionable to writers.

---

## 1. Purpose and scope

**In scope.** Generate three classes of explanation:

* **Global feature attribution**: which features matter most across
  the calibration set, ranked by mean |SHAP|. Comparable to but
  more rigorous than the Phase 4 importance figure.
* **Per-film attribution**: for each calibration-set film, which
  features pushed the headline-target probability up or down.
  Rendered as both a ranked list and (optionally) a waterfall plot.
* **Scene-level attribution**: for a small set of representative
  example films, which scenes drove the prediction in each
  direction. Computed via per-scene removal counterfactual.

**Out of scope.** Test-set SHAP attribution (Phase 8 only — the
held-out 257 test films are not touched in Phase 7). SHAP for the
SVM-RBF roi_gt_1 winner via KernelSHAP (compute prohibitive at
~hours per attribution; permutation-based importance from
Phase 4 ``importance.py`` is the documented alternative). Any new
model fitting or hyperparameter search. Any modification of the
Phase 5 calibrated wrappers or the Phase 6 decision rule.

**Primary attribution target**: the **headline `roi_gt_2`
XGBoost winner**, fit by Phase 4 finalize on
`standalone_positive_union_mpnet`. Secondary: the `log_roi`
RandomForest winner (TreeSHAP works natively). Tertiary: the
`roi_gt_1` SVM-RBF winner is excluded from per-film SHAP per the
KernelSHAP cost; Phase 4 permutation importance is the
substitute for global ranking.

---

## 2. Strategic decisions inherited (locked)

* **Phase 4 winners are the attribution targets**, loaded from
  `data/processed/phase4_primary_model_*.joblib`.
* **The 257-film calibration set is the SHAP evaluation surface.**
  Already used by Phases 5 and 6; SHAP attribution does not refit
  the model, so re-using these films introduces no leakage.
* **The 257-film test set is reserved for Phase 8.** Phase 7 does
  not touch it. Phase 8 will repeat the SHAP attribution on the
  test set as part of the end-to-end report.
* **`save_run` discipline mandatory** for the canonical Phase 7
  attribution run.

---

## 3. SHAP methodology per family (locked)

### 3.1 XGBoost (`roi_gt_2` headline winner)

**Method**: `shap.TreeExplainer(estimator)` with the
`feature_perturbation="tree_path_dependent"` default. TreeSHAP
is exact (not approximated), fast (sub-second per sample on
gradient-boosted trees), and produces additive attributions
that sum to the model's log-odds output minus the base rate.

**Conversion to probability space**: TreeSHAP returns log-odds
contributions for binary classification. For per-film reports we
convert the SHAP-augmented log-odds to probability via the logistic
function for interpretability ("feature X added 0.07 probability").
Sign and ranking are preserved by the monotonic transform.

### 3.2 RandomForest (`log_roi` regression winner)

**Method**: same `shap.TreeExplainer` API; TreeSHAP applies to
RF regressors directly. SHAP values are in log-revenue-ratio units
(the regression target's natural scale).

### 3.3 SVM-RBF (`roi_gt_1` classifier winner)

**Method**: **excluded from per-film SHAP** (deviation from the
roadmap). KernelSHAP on a 257-film cal set with 92 features would
take 4-8 hours per run and produce only approximate attributions.
The Phase 4 ``importance.py`` permutation importance run gives a
serviceable global ranking; that table is referenced as the
substitute for SVM SHAP in the Phase 7 deliverables. Documented
in the summary's Section 11 deviation list.

---

## 4. Global feature attribution (locked)

For the headline XGBoost winner, the global ranking is computed as:

1. Compute SHAP values for all 257 calibration films.
2. Per feature: `importance = mean(|SHAP value|)` across films.
3. Rank features by this importance.

The top-20 ranking is rendered as the headline figure
``phase7_global_shap.png`` and saved to
``reports/tables/phase7_global_shap_roi_gt_2.csv``. Same logic
applies for the RF on log_roi (top-20 ranking by mean |SHAP|).

The pre-registered comparison: how does the SHAP global ranking
differ from Phase 4's native ``feature_importances_`` ranking? If
ranks correlate at Spearman ≥ 0.8, the two methods agree. If ≤ 0.5,
they disagree (a methodology finding worth documenting).

---

## 5. Per-film attribution (locked)

For each of the 257 calibration films:

1. Compute the 92 SHAP values via TreeExplainer.
2. Map each SHAP value to its film-feature contribution.
3. Rank by absolute SHAP value.
4. Persist the top-5 positive contributors (features that push
   probability up) and top-5 negative contributors (features that
   push probability down).
5. Construct a per-film rationale string concatenating the top
   contributors with their effects (e.g.,
   "Adventure genre (+0.12 probability), embedding component 5
   (+0.07 probability), ...").

The per-film rationale extends the Phase 6 decision rationale: in
Phase 6 the rationale stated only the recommended action and
expected costs; in Phase 7 the rationale also names the features
driving the underlying probability.

---

## 6. Scene-level attribution (locked)

The most ambitious part of the original brief. Approach:
**per-scene removal counterfactual.**

For each example film:

1. Take the original parsed screenplay (N scenes) and compute
   features → original probability `p_orig`.
2. For each scene `s` from 1 to N:
   a. Remove scene `s` from the parsed screenplay.
   b. Re-extract all 92 features on the (N-1)-scene version.
   c. Run the calibrated wrapper to get the new probability `p_minus_s`.
   d. Scene contribution = `p_orig - p_minus_s`.
3. Rank scenes by absolute contribution. Top-3 positive (scenes
   that pushed probability up) and top-3 negative (scenes that
   pulled probability down) are the actionable feedback the
   system reports.

**Compute cost**: one feature re-extraction takes ~5-10 seconds
(structural counts + topic transform + character network rebuild
+ embedding mean-pool over scenes); at 130 scenes per film,
~10-20 minutes per film. Pre-registered: **scene-level
attribution is computed for 5 representative example films
only** (deliberate sampling: 1 high-confidence Greenlight from
Phase 6, 1 Drama referred at high uncertainty, 1 Adventure
high-confidence hit, 1 misclassified prestige film from the
Phase 4 error analysis, 1 misclassified sleeper hit).

The 5-film selection is locked at proposal time and not expanded
after seeing results.

**Per-film rationale extension**: the scene-level top contributors
extend the Phase 7 per-film rationale ("Scene 47 (the protagonist's
confession) increased predicted hit probability by 8 percentage
points; Scene 73 (the climactic chase) decreased it by 5 points").

---

## 7. Stability validation (locked)

TreeSHAP is **deterministic** for a fixed estimator and input
matrix. The pre-registered stability check:

1. Compute SHAP twice with different `feature_perturbation` modes
   (`"tree_path_dependent"` vs `"interventional"`); the latter is
   the unbiased estimator (Lundberg et al. 2020) that requires a
   background dataset.
2. Compare the per-feature global ranking between the two methods.
3. Spearman rank correlation must exceed **0.8**; below that, the
   model's interactions are too dependent on conditioning choices
   to trust.

For scene-level attribution, stability is harder (the feature re-
extraction is deterministic for a fixed parser, but small parsing
variations could produce different scene contributions). The
stability check: re-run the scene removal twice with seed-perturbed
LDA topic transform. The top-3 positive / top-3 negative scene
ranking must overlap by at least **2 of 3** scenes between runs;
if not, scene attributions are too noisy and we report
feature-level only.

---

## 8. Output artifacts (locked)

### 8.1 Explainer + per-film bundles

* `data/processed/phase7_shap_explainer_<target>.joblib` — fitted
  `TreeExplainer` instance + base value + reference to Phase 4
  bundle. One per supported target (xgboost, random_forest).
* `data/processed/phase7_per_film_attribution_<target>.parquet` —
  per-film SHAP values for the 257 calibration films (one row per
  film, columns: imdb_id, recommended_action, top features +
  values).
* `data/processed/phase7_scene_level_examples.json` — scene-level
  contributions for the 5 example films (per scene: text snippet,
  contribution to probability).

### 8.2 Tables

* `reports/tables/phase7_global_shap_roi_gt_2.csv` — top-20
  global feature ranking by mean |SHAP|.
* `reports/tables/phase7_global_shap_log_roi.csv` — same for the
  regression winner.
* `reports/tables/phase7_shap_vs_native_importance.csv` —
  Spearman rank correlation between SHAP global ranking and
  Phase 4 ``feature_importances_``; comparison row per feature.
* `reports/tables/phase7_per_film_rationale.csv` — 257 rows of
  per-film attribution + rationale string + Phase 6 action.
* `reports/tables/phase7_scene_level.csv` — per-scene contribution
  for the 5 example films (5 × ~130 = ~650 rows).
* `reports/tables/phase7_stability.csv` — Spearman rank
  correlations from the stability validation.

### 8.3 Figures

* `phase7_global_shap.png` — top-20 global feature ranking by
  mean |SHAP|, side-by-side for XGBoost (roi_gt_2) and RF
  (log_roi).
* `phase7_shap_vs_native.png` — scatter of SHAP rank vs native
  importance rank; identity line; Spearman ρ printed in the title.
* `phase7_per_film_examples.png` — waterfall plot for 4
  representative films (1 Greenlight, 1 Pass, 1 Refer, 1 surprising).
* `phase7_scene_level_example.png` — per-scene contribution bar
  chart for 1 of the 5 example films, with scene headings on the
  y-axis.

### 8.4 Documentation

* `docs/summaries/phase_7_summary.md` per the Section 7 template.
* `docs/PROJECT_CONTEXT.md` Sections 8 (decisions log) + 9 (status).
* `runs/phase_7/<timestamp>_phase7_attribution/` save_run directory.

---

## 9. Pre-registered escalation criteria (locked)

Phase 7 **has** a mandatory end-of-phase escalation per
`PROJECT_CONTEXT.md` Section 11. The four intra-phase triggers:

1. **SHAP-vs-native rank disagreement**: Spearman correlation
   between SHAP global ranking and Phase 4
   ``feature_importances_`` ranking is below 0.5. Indicates the
   model's interactions are too tangled for global importance to
   be a meaningful summary; would mean per-film SHAP is the
   honest unit of explanation, not global ranking.
2. **TreeSHAP stability failure**: Spearman correlation between
   `tree_path_dependent` and `interventional` SHAP global
   rankings is below 0.8. Indicates the model is too sensitive
   to conditioning assumptions to trust either method.
3. **Scene-level instability**: top-3-scene rank overlap between
   stability re-runs is below 2 of 3. Indicates scene-level
   attributions are noise-dominated; report falls back to
   feature-level only (per the roadmap's end-of-Phase-7 question).
4. **Compute overrun**: scene-level attribution on the 5 example
   films exceeds 90 minutes wall-clock total. Surface for
   methodology adjustment (drop to 3 example films, or use a
   coarser scene-grouping like first/middle/last act).

---

## 10. Compute budget

| Step | Estimate |
|---|---|
| Install + verify `shap` | 1 minute |
| TreeSHAP global on XGBoost (257 films) | ~30 seconds |
| TreeSHAP global on RF (257 films, 92 features) | ~1 minute |
| Per-film attribution + rationale strings | ~2 minutes |
| Stability validation (interventional rerun) | ~5 minutes |
| Scene-level for 5 example films | ~50-90 minutes |
| Figure rendering | ~3 minutes |
| Total | **~1.5-2 hours** |

Reduction levers (in order):
1. Drop scene-level entirely; go feature-level only (saves 50-90
   min). Falls back to the roadmap's "feature-level only" option.
2. Drop scene-level for the 2 misclassified films; keep 3 (saves
   30-50 min).
3. Use a coarser act-level (first / middle / last third of
   scenes) instead of per-scene attribution.

---

## 11. Deviations from the roadmap

Two deliberate deviations:

1. **SHAP applied to the calibration set, not the test set.** The
   roadmap text says "TreeSHAP attributions on the primary model
   for the test set." But ``PROJECT_CONTEXT.md`` Section 6
   prohibits touching the test set before Phase 8. Resolution:
   Phase 7 SHAP runs on the calibration set; Phase 8 repeats the
   attribution on the test set as part of the final evaluation.
2. **SVM-RBF excluded from per-film SHAP**. KernelSHAP cost
   prohibitive; permutation importance from Phase 4
   ``importance.py`` is the documented substitute for global
   ranking. Per-film SVM attributions would require fitting a
   per-film logistic surrogate, which is not pre-registered here.

---

## 12. What is locked and what is not

**Locked.** SHAP methodology per family (Section 3), global
ranking computation (Section 4), per-film schema (Section 5),
scene-level methodology + 5-film selection (Section 6), stability
validation (Section 7), output artifacts (Section 8), escalation
criteria (Section 9), deviations (Section 11).

**Not locked (tactical).** Helper structure, plot styling, the
specific 5 example films chosen for scene-level (will document the
selection rationale at execution time per the criteria in
Section 6), table column ordering, log verbosity, figure binning.

**Pre-registration discipline.** The set above is not expanded
after seeing results.
