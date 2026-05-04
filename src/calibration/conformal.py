"""Conformal prediction for Phase 5.

Two wrappers, one per task type:

* :func:`evaluate_conformal_classifier` — split conformal binary
  classification using the LAC (Least Ambiguous Classifier)
  conformity score, hand-implemented. We do not use MAPIE 1.4 for
  classification because MAPIE's internal numpy conversion breaks
  prefit sklearn Pipelines that use ``ColumnTransformer`` with
  string column names (which all our Phase 4 winners do). The LAC
  procedure is trivial to implement and the resulting empirical
  coverage matches MAPIE's on toy data within finite-sample noise.
* :func:`evaluate_conformal_regressor` — split conformal regression
  via :class:`mapie.regression.SplitConformalRegressor`. MAPIE's
  regression path does not hit the column-name issue (it does not
  call predict_proba) so we keep it.

Both wrappers run 5-fold cross-validation within the calibration set
to estimate empirical coverage honestly: for each fold, fit
conformity scores on (cal \\ fold) and evaluate coverage on the
fold. Aggregate.

The deployed wrappers (returned in the result) are conformalized on
all 257 calibration films, which is what the canonical Phase 5
artifact uses.

For classification, the prediction set ``{0, 1}`` is the "Refer to
human reader" indicator for Phase 6: the model cannot distinguish
flop from hit at the chosen confidence level.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.calibration.cv import SEED, stratified_cal_folds
from src.utils.logging import get_logger

logger = get_logger(__name__)


CONFIDENCE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]
ClassificationConformityScore = Literal["lac", "aps"]
RegressionConformityScore = Literal["absolute", "gamma"]


@dataclass
class LACConformalClassifier:
    """Hand-rolled split-conformal LAC wrapper for binary classification.

    Stores the conformity-score quantiles per confidence level (computed
    from the calibration set's true-class probabilities) and exposes
    ``predict_set(X)`` to produce per-row prediction sets.

    The wrapped estimator must have ``predict_proba``. For SVM-RBF
    families (no native ``predict_proba``), the caller passes a
    Platt/isotonic-wrapped version.
    """

    estimator: BaseEstimator
    confidence_levels: list[float]
    quantiles: dict[float, float]  # {confidence_level: quantile of conformity}
    n_cal: int

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Forward to the underlying estimator's predict_proba."""
        return self.estimator.predict_proba(X)

    def predict_set(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_pred, y_pss) matching the MAPIE shape convention.

        ``y_pred`` is the argmax class. ``y_pss`` is shape
        ``(n, 2, n_levels)`` boolean mask: ``y_pss[i, c, l]`` is True
        iff class ``c`` is in the prediction set for sample ``i`` at
        confidence level index ``l``.
        """
        proba = self.estimator.predict_proba(X)  # (n, 2)
        y_pred = proba.argmax(axis=1)
        # Conformity score for each (sample, class): 1 - p(class)
        score_per_class = 1.0 - proba  # (n, 2)
        y_pss = np.zeros((len(X), 2, len(self.confidence_levels)), dtype=bool)
        for li, level in enumerate(self.confidence_levels):
            q = self.quantiles[level]
            y_pss[:, :, li] = score_per_class <= q
        return y_pred, y_pss


def _fit_lac_quantiles(
    estimator: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    confidence_levels: list[float],
) -> dict[float, float]:
    """Compute LAC conformity quantiles from cal-set true-class probabilities.

    Conformity score for sample ``i`` is ``1 - p(true_class_i)``. The
    quantile at confidence level ``c`` is the
    ``ceil((n + 1) * c) / n``-th order statistic of conformity scores
    (the standard finite-sample correction).
    """
    proba = estimator.predict_proba(X_cal)  # (n_cal, 2)
    n = len(y_cal)
    cal_scores = 1.0 - proba[np.arange(n), y_cal.astype(int)]
    sorted_scores = np.sort(cal_scores)
    quantiles: dict[float, float] = {}
    for level in confidence_levels:
        # k = ceil((n + 1) * level), clipped to n
        k = int(np.ceil((n + 1) * level))
        k = min(max(k, 1), n)
        quantiles[level] = float(sorted_scores[k - 1])
    return quantiles


@dataclass
class ConformalClassificationResult:
    """Per-fold + aggregated coverage / set-size for one (target, score)."""
    target: str
    conformity_score: ClassificationConformityScore
    per_fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, float]
    deployed_wrapper: LACConformalClassifier
    deployed_predictions: dict[str, np.ndarray]  # 'y_pred', 'y_pss' on full cal predict-on-self


@dataclass
class ConformalRegressionResult:
    """Per-fold + aggregated coverage / interval-width for one (target, score)."""
    target: str
    conformity_score: RegressionConformityScore
    per_fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, float]
    deployed_wrapper: SplitConformalRegressor
    deployed_predictions: dict[str, np.ndarray]  # 'y_pred', 'y_pis' on full cal predict-on-self


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _wrap_for_predict_proba(
    estimator: BaseEstimator,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    calibration_method: str | None,
) -> BaseEstimator:
    """Return an estimator with ``predict_proba`` for MAPIE.

    If the base estimator already has ``predict_proba`` and
    ``calibration_method`` is None, return it as-is. Otherwise wrap
    via ``CalibratedClassifierCV(FrozenEstimator(estimator), method=...)``,
    fit on (X_fit, y_fit). This preserves no-leakage discipline because
    each conformal fold re-fits its own Platt/isotonic mapping.
    """
    if calibration_method is None and hasattr(estimator, "predict_proba"):
        return estimator
    method = calibration_method or "sigmoid"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapped = CalibratedClassifierCV(
            FrozenEstimator(estimator), method=method,
        )
        wrapped.fit(X_fit, y_fit)
    return wrapped


def evaluate_conformal_classifier(
    estimator: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    target: str,
    conformity_score: ClassificationConformityScore = "lac",
    confidence_levels: list[float] | None = None,
    calibration_method: str | None = None,
) -> ConformalClassificationResult:
    """5-fold within-cal split-conformal evaluation for binary classification.

    If ``calibration_method`` is set (``"sigmoid"`` or ``"isotonic"``),
    each fold first re-fits a Platt/isotonic wrapper on the fold's
    fit-side data, then conformalizes the wrapped estimator. This
    preserves the no-leakage discipline. If ``calibration_method`` is
    None, the base estimator is conformalized directly (requires
    ``predict_proba``; SVM-RBF families must use a calibration_method).
    """
    confidence_levels = confidence_levels or CONFIDENCE_LEVELS
    fold_rows = []

    for fold_i, (fit_idx, eval_idx) in enumerate(
        stratified_cal_folds(y_cal, task="classification"),
    ):
        X_fit, y_fit = X_cal.iloc[fit_idx], y_cal[fit_idx]
        X_eval, y_eval = X_cal.iloc[eval_idx], y_cal[eval_idx]
        # Per-fold calibration wrapping for honest no-leakage CV.
        wrapped = _wrap_for_predict_proba(
            estimator, X_fit, y_fit, calibration_method,
        )
        # Hand-rolled LAC split conformal (avoids MAPIE's prefit
        # column-name issue on sklearn Pipelines with ColumnTransformer).
        quantiles = _fit_lac_quantiles(wrapped, X_fit, y_fit, confidence_levels)
        fold_wrapper = LACConformalClassifier(
            estimator=wrapped,
            confidence_levels=confidence_levels,
            quantiles=quantiles,
            n_cal=int(len(X_fit)),
        )
        _, y_pss = fold_wrapper.predict_set(X_eval)

        for level_i, level in enumerate(confidence_levels):
            sets = y_pss[:, :, level_i]  # (n_eval, n_classes) bool
            in_set = sets[np.arange(len(y_eval)), y_eval.astype(int)]
            set_sizes = sets.sum(axis=1)
            fold_rows.append({
                "fold": fold_i,
                "level": level,
                "empirical_coverage": float(in_set.mean()),
                "mean_set_size": float(set_sizes.mean()),
                "singleton_rate": float((set_sizes == 1).mean()),
                "refer_rate": float((set_sizes == 2).mean()),
                "empty_rate": float((set_sizes == 0).mean()),
                "n_eval": int(len(y_eval)),
            })

    per_fold = pd.DataFrame(fold_rows)
    agg = {}
    for level in confidence_levels:
        sub = per_fold[per_fold["level"] == level]
        for col in ("empirical_coverage", "mean_set_size", "singleton_rate", "refer_rate"):
            agg[f"{col}_at_{level}_mean"] = float(sub[col].mean())
            agg[f"{col}_at_{level}_std"] = float(sub[col].std(ddof=1))

    # Deployed wrapper conformalized on all 257 films. Uses the same
    # hand-rolled LAC procedure as the per-fold runs; the deployed
    # quantiles are estimated from the full calibration set.
    deployed_wrapped = _wrap_for_predict_proba(
        estimator, X_cal, y_cal, calibration_method,
    )
    deployed_quantiles = _fit_lac_quantiles(
        deployed_wrapped, X_cal, y_cal, confidence_levels,
    )
    deployed = LACConformalClassifier(
        estimator=deployed_wrapped,
        confidence_levels=confidence_levels,
        quantiles=deployed_quantiles,
        n_cal=int(len(X_cal)),
    )

    logger.info(
        "Conformal classif | %s | %s | coverage@0.9=%.3f singleton@0.9=%.3f",
        target, conformity_score,
        agg["empirical_coverage_at_0.9_mean"],
        agg["singleton_rate_at_0.9_mean"],
    )
    return ConformalClassificationResult(
        target=target,
        conformity_score=conformity_score,
        per_fold_metrics=per_fold,
        aggregate_metrics=agg,
        deployed_wrapper=deployed,
        deployed_predictions={},  # populated by caller as needed
    )


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def evaluate_conformal_regressor(
    estimator: BaseEstimator,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    target: str,
    conformity_score: RegressionConformityScore = "absolute",
    confidence_levels: list[float] | None = None,
) -> ConformalRegressionResult:
    """5-fold within-cal split-conformal evaluation for regression."""
    confidence_levels = confidence_levels or CONFIDENCE_LEVELS
    fold_rows = []

    for fold_i, (fit_idx, eval_idx) in enumerate(
        stratified_cal_folds(y_cal, task="regression"),
    ):
        X_fit, y_fit = X_cal.iloc[fit_idx], y_cal[fit_idx]
        X_eval, y_eval = X_cal.iloc[eval_idx], y_cal[eval_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mapie = SplitConformalRegressor(
                estimator=estimator,
                confidence_level=confidence_levels,
                conformity_score=conformity_score,
                prefit=True,
            )
            mapie.conformalize(X_fit, y_fit)
            _, y_pis = mapie.predict_interval(X_eval)

        for level_i, level in enumerate(confidence_levels):
            lower = y_pis[:, 0, level_i]
            upper = y_pis[:, 1, level_i]
            in_int = (y_eval >= lower) & (y_eval <= upper)
            widths = upper - lower
            fold_rows.append({
                "fold": fold_i,
                "level": level,
                "empirical_coverage": float(in_int.mean()),
                "mean_width": float(widths.mean()),
                "median_width": float(np.median(widths)),
                "n_eval": int(len(y_eval)),
            })

    per_fold = pd.DataFrame(fold_rows)
    agg = {}
    for level in confidence_levels:
        sub = per_fold[per_fold["level"] == level]
        for col in ("empirical_coverage", "mean_width", "median_width"):
            agg[f"{col}_at_{level}_mean"] = float(sub[col].mean())
            agg[f"{col}_at_{level}_std"] = float(sub[col].std(ddof=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deployed = SplitConformalRegressor(
            estimator=estimator,
            confidence_level=confidence_levels,
            conformity_score=conformity_score,
            prefit=True,
        )
        deployed.conformalize(X_cal, y_cal)

    logger.info(
        "Conformal reg | %s | %s | coverage@0.9=%.3f mean_width@0.9=%.3f",
        target, conformity_score,
        agg["empirical_coverage_at_0.9_mean"],
        agg["mean_width_at_0.9_mean"],
    )
    return ConformalRegressionResult(
        target=target,
        conformity_score=conformity_score,
        per_fold_metrics=per_fold,
        aggregate_metrics=agg,
        deployed_wrapper=deployed,
        deployed_predictions={},
    )
