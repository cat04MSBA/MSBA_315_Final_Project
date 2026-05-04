"""Probability calibration: Platt scaling + isotonic regression.

Wraps sklearn's :class:`CalibratedClassifierCV` with ``cv="prefit"``,
which fits the Platt or isotonic mapping on the calibration set
without re-training the underlying classifier. This is the right
primitive when we already have Phase 4 winners and want to add a
post-hoc probability calibration layer.

For SVM-RBF (whose ``decision_function`` outputs are not in [0, 1]),
``CalibratedClassifierCV`` with ``method="sigmoid"`` reproduces
classical Platt scaling: a single-parameter logistic on the raw
margin scores.

For tree-based winners (XGBoost), ``CalibratedClassifierCV`` reads
``predict_proba`` and adjusts via either Platt (a logistic of the
predicted log-odds) or isotonic (a monotonic step function fit
to bin centers).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.calibration.cv import stratified_cal_folds
from src.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


CalibrationMethod = Literal["sigmoid", "isotonic", "uncalibrated"]


@dataclass
class ProbabilityCalibrationResult:
    """One (target, method) probability-calibration result."""
    target: str
    method: CalibrationMethod
    per_fold_metrics: pd.DataFrame  # rows: fold index; cols: ECE, MCE, Brier, log_loss
    deployed_calibrator: BaseEstimator | None  # None for "uncalibrated"
    aggregate_metrics: dict[str, float]


def _fit_calibrator(
    estimator: BaseEstimator,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    method: CalibrationMethod,
) -> BaseEstimator | None:
    """Return a fitted CalibratedClassifierCV for sigmoid/isotonic; None for uncalibrated.

    sklearn 1.6+ requires wrapping prefit estimators with
    ``FrozenEstimator`` (the older ``cv="prefit"`` API was removed in
    1.8). The wrapped estimator's ``fit`` is a no-op; only the
    calibration mapping is fit on ``X_fit, y_fit``.
    """
    if method == "uncalibrated":
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = CalibratedClassifierCV(
            FrozenEstimator(estimator), method=method,
        )
        cal.fit(X_fit, y_fit)
    return cal


def _predict_proba(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    score_method: str,
) -> np.ndarray:
    """Return P(positive class) regardless of underlying score type.

    Used for the "uncalibrated" baseline so we can ECE the raw
    scores. SVM decision_function values are mapped to [0, 1] via
    a logistic for the ECE calculation; that monotonic transform
    does not change AUC but lets us put SVM and predict_proba
    families on the same calibration scale.
    """
    if score_method == "predict_proba":
        return estimator.predict_proba(X)[:, 1]
    raw = estimator.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def evaluate_calibration_method(
    estimator: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: CalibrationMethod,
    target: str,
    score_method: str,
    n_bins: int = 10,
) -> ProbabilityCalibrationResult:
    """5-fold within-cal evaluation of one calibration method.

    For each fold: fit Platt/isotonic on (cal \\ fold), predict on
    fold, compute ECE/MCE/Brier/log-loss. Aggregate across folds.

    The deployed calibrator (returned in the result) is fit on
    the full calibration set.
    """
    fold_rows = []
    for fold_i, (fit_idx, eval_idx) in enumerate(
        stratified_cal_folds(y_cal, task="classification"),
    ):
        X_fit, y_fit = X_cal.iloc[fit_idx], y_cal[fit_idx]
        X_eval, y_eval = X_cal.iloc[eval_idx], y_cal[eval_idx]
        cal = _fit_calibrator(estimator, X_fit, y_fit, method)
        if cal is None:
            probs = _predict_proba(estimator, X_eval, score_method)
        else:
            probs = cal.predict_proba(X_eval)[:, 1]
        fold_rows.append({
            "fold": fold_i,
            "ece": expected_calibration_error(y_eval, probs, n_bins=n_bins),
            "mce": maximum_calibration_error(y_eval, probs, n_bins=n_bins),
            "brier": brier_score(y_eval, probs),
            "log_loss": negative_log_likelihood(y_eval, probs),
            "n_eval": int(len(y_eval)),
        })

    per_fold = pd.DataFrame(fold_rows)
    agg = {
        f"{m}_mean": float(per_fold[m].mean()) for m in ("ece", "mce", "brier", "log_loss")
    }
    agg.update({
        f"{m}_std": float(per_fold[m].std(ddof=1)) for m in ("ece", "mce", "brier", "log_loss")
    })

    # Deployed calibrator fits on the full calibration set.
    deployed = _fit_calibrator(estimator, X_cal, y_cal, method)
    logger.info(
        "Probability calib | %s | %s | mean ECE=%.4f Brier=%.4f log_loss=%.4f",
        target, method, agg["ece_mean"], agg["brier_mean"], agg["log_loss_mean"],
    )
    return ProbabilityCalibrationResult(
        target=target,
        method=method,
        per_fold_metrics=per_fold,
        deployed_calibrator=deployed,
        aggregate_metrics=agg,
    )


def select_best_method(
    results: list[ProbabilityCalibrationResult],
) -> ProbabilityCalibrationResult:
    """Pick the method with lowest mean ECE.

    Tie-break to the simpler method (sigmoid > isotonic > uncalibrated).
    """
    ranked = sorted(
        results,
        key=lambda r: (
            r.aggregate_metrics["ece_mean"],
            {"sigmoid": 0, "isotonic": 1, "uncalibrated": 2}[r.method],
        ),
    )
    return ranked[0]
