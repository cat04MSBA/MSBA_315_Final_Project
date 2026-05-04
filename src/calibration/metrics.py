"""Calibration metrics: ECE, MCE, Brier, reliability bins.

The standard probability-calibration metrics. ECE is the headline:
a perfectly calibrated classifier outputs probability values that
match the empirical positive rate within each probability bin, so
ECE = 0 means perfect calibration. MCE is the worst-bin disagreement.
Brier score is the squared error between predicted probability and
binary outcome (lower better; combines calibration and resolution).

For evaluation honesty, both metrics are computed on **held-out
predictions**, never on the data the calibrator was fit on. The
:func:`reliability_curve` helper returns the bin centers + empirical
accuracies + bin counts that the reliability-diagram plot consumes.

All functions take 1-D numpy arrays of length ``n_samples`` and
return scalars (or for :func:`reliability_curve`, three arrays).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


BinningStrategy = Literal["equal_width", "equal_size"]


def _bin_edges(
    probs: np.ndarray, n_bins: int, strategy: BinningStrategy,
) -> np.ndarray:
    """Compute bin edges per the chosen binning strategy."""
    if strategy == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    # equal_size: quantile-based; produces bins with roughly equal counts.
    edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0
    # Force monotonicity in the rare case of ties at the boundary.
    edges = np.maximum.accumulate(edges)
    return edges


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_size",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean predicted probability, empirical positive rate, count).

    Returns three arrays of length ``n_bins``. Empty bins return NaN
    in both the mean-prob and accuracy slots; the count is 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = _bin_edges(y_prob, n_bins, strategy)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            mean_pred[b] = y_prob[mask].mean()
            accuracy[b] = y_true[mask].mean()
            counts[b] = int(mask.sum())
    return mean_pred, accuracy, counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_size",
) -> float:
    """Weighted mean per-bin |confidence - accuracy| (lower better)."""
    mean_pred, accuracy, counts = reliability_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy,
    )
    n = counts.sum()
    if n == 0:
        return float("nan")
    valid = counts > 0
    weighted = (counts[valid] / n) * np.abs(mean_pred[valid] - accuracy[valid])
    return float(weighted.sum())


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: BinningStrategy = "equal_size",
) -> float:
    """Worst-bin |confidence - accuracy| (lower better)."""
    mean_pred, accuracy, counts = reliability_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy,
    )
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float(np.max(np.abs(mean_pred[valid] - accuracy[valid])))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome.

    Bounded in [0, 1] (lower better). The Brier score decomposes into
    calibration + resolution + uncertainty; this function returns the
    composite measure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def negative_log_likelihood(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Binary cross-entropy / log-loss; lower better."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob),
    ))
