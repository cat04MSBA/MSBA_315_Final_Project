"""Cross-validation and bootstrap metric helpers for Phase 3 baselines.

Metric set committed to as of 2026-05-03 (planning-conversation
directive):

* Regression: MSE, RMSE, MAE, CVRMSE (coefficient of variation of
  RMSE: RMSE divided by the absolute mean of the target). R² is
  removed in favour of these absolute / normalized measures, which
  are more robust on small samples and easier to compare across
  feature configurations.
* Classification: AUC-ROC, PR-AUC, F1, log-loss. F1 is computed at
  the default 0.5 threshold on hard label predictions; log-loss is
  computed on probability predictions. Both threshold-free metrics
  (AUC-ROC, PR-AUC) and threshold-dependent metrics (F1) are
  reported so the reader sees both ranking quality and operating-
  point quality.

Two prediction types are needed for classification:
:data:`CLASSIFICATION_METRICS_PROBA` consume probabilities (or
decision-function scores for SVM) and :data:`CLASSIFICATION_METRICS_HARD`
consume hard predicted labels. Regression metrics consume real-valued
predictions only.

The bootstrap helper resamples paired ``(y_true, y_pred)`` with
replacement, recomputes the metric on each sample, and returns
percentile bounds. Bootstrap samples that fail (single-class for AUC,
single-value-prediction for log-loss) are dropped.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error, in y units."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of variation of the RMSE.

    Defined as ``RMSE / mean(|y_true|)``. Unitless, normalized by
    target scale, easier to compare across regression problems with
    different y-scale magnitudes than RMSE alone. Returns ``inf`` if
    the absolute mean of ``y_true`` is zero.
    """
    rmse = _rmse(y_true, y_pred)
    abs_mean = float(np.mean(np.abs(y_true)))
    if abs_mean == 0.0:
        return float("inf")
    return rmse / abs_mean


REGRESSION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": mean_squared_error,
    "rmse": _rmse,
    "mae": mean_absolute_error,
    "cvrmse": _cvrmse,
}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def _f1_binary(y_true: np.ndarray, y_pred_hard: np.ndarray) -> float:
    """F1 for binary classification (positive class = 1)."""
    return float(f1_score(y_true, y_pred_hard, average="binary"))


def _log_loss_safe(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Binary log-loss (cross-entropy).

    Uses sklearn's :func:`log_loss` with default clipping. Expects
    probability predictions for the positive class (a 1-D array of
    values in ``[0, 1]``). For models that produce decision-function
    scores instead of probabilities (e.g., the SVC family in this
    project), the caller passes ``np.nan`` and this function is
    skipped at the trainer level.
    """
    return float(log_loss(y_true, y_pred_proba, labels=[0, 1]))


# Metrics that consume probability (or decision-function-style score)
# predictions. Threshold-free or score-distribution-aware.
CLASSIFICATION_METRICS_PROBA: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_roc": roc_auc_score,
    "pr_auc": average_precision_score,
    "log_loss": _log_loss_safe,
}


# Metrics that consume hard label predictions. Threshold-dependent.
CLASSIFICATION_METRICS_HARD: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "f1": _f1_binary,
}


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iter: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for a metric over ``(y_true, y_pred)``.

    Bootstrap samples that raise ``ValueError`` (e.g., AUC-ROC or
    log-loss on a single-class sample) are skipped rather than
    propagated as ``nan`` into the CI.
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
