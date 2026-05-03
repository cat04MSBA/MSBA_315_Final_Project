"""Cross-validation + bootstrap metric helpers for Phase 3a baselines.

Each metric is a callable ``(y_true, y_pred_or_score) -> float`` so the
same machinery handles regression and classification. The bootstrap
helper resamples paired ``(y_true, y_pred)`` with replacement, recomputes
the metric on each sample, and returns percentile bounds. Bootstrap
samples that fail (e.g. only one class present for AUC-ROC) are dropped.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# Map of metric name → metric function. Functions take
# ``(y_true, y_pred_or_score)`` and return a float.
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


REGRESSION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "r2": r2_score,
    "mae": mean_absolute_error,
    "rmse": _rmse,
}

CLASSIFICATION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_roc": roc_auc_score,
    "pr_auc": average_precision_score,
}


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iter: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric over ``(y_true, y_pred)``.

    Parameters
    ----------
    metric_fn
        Callable ``(y_true, y_pred) -> float``.
    y_true, y_pred
        Aligned 1-D arrays.
    n_iter
        Number of bootstrap resamples. Project default 1000.
    seed
        RNG seed for reproducibility. Project standard 42.
    alpha
        Significance level. ``0.05`` → 95% CI.

    Returns
    -------
    (lo, hi)
        Lower and upper percentile bounds of the bootstrap
        distribution. Bootstrap samples that raise ``ValueError``
        (e.g. AUC-ROC on a single-class sample) are skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            estimates.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except ValueError:
            continue
    if not estimates:
        return float("nan"), float("nan")
    arr = np.asarray(estimates)
    return (
        float(np.quantile(arr, alpha / 2)),
        float(np.quantile(arr, 1 - alpha / 2)),
    )
