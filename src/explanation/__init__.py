"""Phase 7 (Layer 4: SHAP Explanations) modules.

Adds per-film and scene-level explanations to the four-layer
triage system. Reads the Phase 4 winners + Phase 5 calibrators +
Phase 6 decision pipeline and produces:

* Global feature attribution per supported target (TreeSHAP on
  XGBoost / RandomForest).
* Per-film attribution + rationale strings for the 257 calibration
  films.
* Scene-level attribution for 5 representative example films via
  per-scene removal counterfactual.

The held-out 257-film test set is not touched. Phase 8 will repeat
the attribution on the test set as part of the end-to-end report.

Modules:

* :mod:`shap_explainer` — TreeExplainer wrappers per supported family.
* :mod:`global_importance` — global ranking + SHAP-vs-native comparison.
* :mod:`per_film` — per-film attributions + rationale string builder.
* :mod:`scene_level` — per-scene removal counterfactual on example films.
* :mod:`report` — combined Phase 6+7 explanation report template.
* :mod:`pipeline` — end-to-end orchestrator + artifact persistence.
* :mod:`figures` — global SHAP, waterfall examples, scene-level bar.
"""
