"""Phase 8 — End-to-end integration and final test-set evaluation.

This package combines the four layers (Phase 4 prediction,
Phase 5 calibrated probability + conformal interval, Phase 6
asymmetric-cost decision, Phase 7 SHAP attribution) into a single
end-to-end pipeline, evaluates it on the held-out 257-film test
set for the first time across the project, runs the pre-registered
error-analysis cuts, and curates the example-output gallery.

Methodology is locked in ``docs/proposals/phase8_preregistration.md``;
the modules below honor that lock.

Modules
-------
* :mod:`src.evaluation.pipeline` — ``triage_report`` end-to-end
  function and the batch-runner that loads the four-layer
  artifacts once and applies them across all 257 test films.
* :mod:`src.evaluation.test_eval` — Layer-by-layer test-set metrics
  (predictive performance, calibration coverage, decision cost,
  attribution stability) with bootstrap confidence intervals.
* :mod:`src.evaluation.error_analysis` — per-genre, per-decade,
  per-budget-tier, and per-screenplay-length-tier breakdowns plus
  the most-correct / most-wrong galleries.
* :mod:`src.evaluation.example_outputs` — five-film example
  gallery selection and Markdown rendering.
* :mod:`src.evaluation.figures` — the six pre-registered Phase 8
  figures.
"""
