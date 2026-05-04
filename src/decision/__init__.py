"""Phase 6 (Layer 3: Asymmetric-Cost Decision) modules.

Converts the Phase 5 calibrated probability into one of three
actions — Greenlight, Pass, Refer to human reader — by minimizing
expected cost under a sourced cost matrix. The cost matrix encodes
the brief's asymmetry: greenlighting a flop costs $50M; passing on
a hit costs $100M; human reader costs $5K.

Modules:

* :mod:`cost_matrix` — the cost-matrix dataclass + sweep variants.
* :mod:`rule` — the per-film expected-cost decision rule.
* :mod:`baselines` — naive comparison strategies (Always-Greenlight,
  Always-Pass, Read-Everything, Random, Genre-prior).
* :mod:`sensitivity` — sweep across cost-matrix variants.
* :mod:`evaluation` — total-cost calculation on the calibration set.
* :mod:`pipeline` — orchestrator + canonical artifact persistence.
* :mod:`figures` — cost curves, action distribution, per-genre.

The held-out 257-film test set is not touched by anything in this
package; reserved for Phase 8.
"""
