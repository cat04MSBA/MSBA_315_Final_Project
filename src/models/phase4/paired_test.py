"""Bayesian correlated-t-test for paired model comparison.

Wraps :func:`baycomp.two_on_single` so the Phase 4 benchmark can ask:
"given the per-fold metric values for models A and B over 15 folds (3
repetitions of 5-fold CV), what is the posterior probability that A is
better than B by more than the pre-registered ROPE?"

The pre-registered ROPE half-widths from
``docs/proposals/phase4_preregistration.md`` Section 9 are encoded in
:data:`ROPE_BY_METRIC`. The runs argument to ``baycomp`` is set to
``N_OUTER_REPEATS`` from :mod:`src.models.phase4.cv`, which lets
``baycomp`` infer the correct rho for the repeated-CV correlation
structure rather than assuming plain 5-fold.

The function returns ``(p_a_better, p_rope, p_b_better)`` posteriors
that sum to 1. A pair is declared a winner if the corresponding
posterior is at least 0.95.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.phase4.cv import N_OUTER_REPEATS

# Whether higher metric values are better. RMSE / MSE / MAE / CVRMSE
# and log_loss are minimized; the rest maximized.
HIGHER_IS_BETTER: dict[str, bool] = {
    "auc_roc": True,
    "pr_auc": True,
    "f1": True,
    "log_loss": False,
    "mse": False,
    "rmse": False,
    "mae": False,
    "cvrmse": False,
}

# Pre-registered ROPE half-widths per metric (Section 9).
# For minimized metrics the ROPE is interpreted on the same metric
# scale; the helper flips the sign so ``baycomp`` always sees a
# "higher is better" comparison.
ROPE_BY_METRIC: dict[str, float] = {
    "auc_roc": 0.005,
    "pr_auc": 0.01,
    "f1": 0.01,
    "log_loss": 0.01,
    "mse": 0.01,
    "rmse": 0.01,
    "mae": 0.01,
    "cvrmse": 0.01,
}

# Posterior threshold to declare a pair a winner.
WINNER_THRESHOLD: float = 0.95


@dataclass(frozen=True)
class PairedComparison:
    """Result of one paired Bayesian comparison."""
    family_a: str
    family_b: str
    metric: str
    p_a_better: float
    p_rope: float
    p_b_better: float
    rope: float
    runs: int
    n_folds: int
    winner: str  # "a", "b", or "rope"


def compare_pair(
    family_a: str,
    family_b: str,
    metric: str,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> PairedComparison:
    """Run the Bayesian correlated-t-test for one pair on one metric.

    ``scores_a`` and ``scores_b`` are length-15 per-fold metric arrays
    from :class:`src.models.phase4.cv.CVResult.per_fold_metrics`.

    NaN folds are removed from both series jointly (a fold that yielded
    NaN for either model is dropped from both). If fewer than 4 folds
    survive, the comparison returns ``rope`` as the declared winner with
    NaN posteriors and the caller is expected to interpret this as
    "insufficient evidence."
    """
    if metric not in ROPE_BY_METRIC:
        raise KeyError(
            f"No pre-registered ROPE for metric {metric!r}; "
            f"known: {sorted(ROPE_BY_METRIC)!r}"
        )

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: a={a.shape} b={b.shape}"
        )
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    rope = ROPE_BY_METRIC[metric]

    if len(a) < 4:
        return PairedComparison(
            family_a=family_a, family_b=family_b, metric=metric,
            p_a_better=float("nan"), p_rope=float("nan"),
            p_b_better=float("nan"),
            rope=rope, runs=N_OUTER_REPEATS, n_folds=int(len(a)),
            winner="rope",
        )

    # baycomp's two_on_single is "higher is better"; flip if needed.
    if not HIGHER_IS_BETTER[metric]:
        a = -a
        b = -b

    # Local import so the module imports cleanly when baycomp is absent
    # (e.g. during test discovery before pip install).
    from baycomp import two_on_single  # type: ignore[import-not-found]

    result = two_on_single(a, b, rope=rope, runs=N_OUTER_REPEATS)
    if rope == 0:
        # baycomp returns (p_left, p_right) when rope=0; should not happen here.
        p_left, p_right = result
        p_rope = 0.0
    else:
        p_left, p_rope, p_right = result

    # baycomp's posterior convention (verified empirically with the
    # call ``two_on_single(better, worse) -> (1, 0, 0)``): with a "higher
    # is better" comparison, ``p_left`` is P(first arg is better than
    # second arg) and ``p_right`` is P(second arg is better). Sign-
    # flipping for minimized metrics happened above so this orientation
    # is correct for both directions.
    p_a_better = float(p_left)
    p_b_better = float(p_right)
    p_rope_f = float(p_rope)

    if p_a_better >= WINNER_THRESHOLD:
        winner = "a"
    elif p_b_better >= WINNER_THRESHOLD:
        winner = "b"
    else:
        winner = "rope"

    return PairedComparison(
        family_a=family_a, family_b=family_b, metric=metric,
        p_a_better=p_a_better, p_rope=p_rope_f, p_b_better=p_b_better,
        rope=rope, runs=N_OUTER_REPEATS, n_folds=int(len(a)),
        winner=winner,
    )


def all_pairwise_comparisons(
    per_family_per_fold: dict[str, dict[str, np.ndarray]],
    metrics: list[str],
) -> list[PairedComparison]:
    """Run all pairwise comparisons within a (matrix, target) cell.

    ``per_family_per_fold[family_name][metric_name]`` returns the
    length-15 per-fold metric array. ``metrics`` is the list of metrics
    over which to run the test (typically the four classification
    metrics or the four regression metrics).

    Returns one :class:`PairedComparison` per (family pair, metric).
    Family pairs are ordered alphabetically and not repeated
    (C(N, 2) pairs for N families).
    """
    families = sorted(per_family_per_fold.keys())
    out: list[PairedComparison] = []
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            for metric in metrics:
                scores_a = per_family_per_fold[fa].get(metric)
                scores_b = per_family_per_fold[fb].get(metric)
                if scores_a is None or scores_b is None:
                    continue
                cmp = compare_pair(fa, fb, metric, scores_a, scores_b)
                out.append(cmp)
    return out
