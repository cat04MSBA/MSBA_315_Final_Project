"""Phase 4 (Layer 1: Core Prediction Model) modules.

The Phase 4 benchmark identifies the strongest model family per target
on the Phase 3 feature matrix. The work is structured as:

* ``families.py`` registers the candidate model families with their
  hyperparameter grids and balancing policies.
* ``matrices.py`` builds the two pre-registered input matrices
  (``all_five`` and ``standalone_positive_union``) from the existing
  Phase 3 feature parquets.
* ``cv.py`` runs repeated stratified 5-fold cross-validation with
  per-fold metric collection so the Bayesian correlated-t-test has the
  per-fold differences it needs.
* ``paired_test.py`` wraps :mod:`baycomp` for the pairwise model
  comparisons.
* ``benchmark.py`` orchestrates the full benchmark over
  (matrix, family, target) and emits the deliverable CSVs and figures.

All choices in these modules are pre-registered in
``docs/proposals/phase4_preregistration.md``. Deviations during
execution are recorded in the Phase 4 summary and the decisions log.
"""
