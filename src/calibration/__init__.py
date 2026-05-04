"""Phase 5 (Layer 2: Calibrated Uncertainty) modules.

Wraps Phase 4 winners with two complementary calibration techniques:

* :mod:`probability` applies Platt and isotonic scaling so the
  classifier's reported probabilities match empirical positive
  rates on held-out films.
* :mod:`conformal` applies split conformal prediction (MAPIE 1.4)
  so the prediction sets / intervals empirically cover the true
  label / value at the nominal confidence level.

The pipeline (orchestrator in :mod:`pipeline`) reads the Phase 4
canonical winner artifacts from ``data/processed/phase4_primary_model_*.joblib``,
applies both techniques to each target, evaluates via 5-fold
cross-validation within the 257-film calibration set, persists
calibrated wrapper artifacts to
``data/processed/phase5_calibrated_model_*.joblib`` for Phase 6, and
emits the diagnostic tables and figures pre-registered in
``docs/proposals/phase5_preregistration.md``.

The held-out 257-film test set (``split == "test"`` in
``split_assignments.parquet``) is not touched by anything in this
package; reserved for Phase 8 final evaluation only.
"""
