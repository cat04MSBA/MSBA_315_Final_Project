"""Repeated cross-validation harness for the Phase 4 benchmark.

Operates on the train split only. Two responsibilities:

* Run a hyperparameter search (``GridSearchCV``, 5-fold inner stratified
  CV) to identify the best hyperparameter cell per (family, matrix,
  target) combination.
* Re-fit the best estimator across the 15 outer folds of repeated 5-fold
  CV (3 repetitions, seed 42) to produce per-fold OOF metric values
  for the Bayesian correlated-t-test in :mod:`paired_test`.

The 15 per-fold metric values are the unit of variation the paired test
operates on. Per-fold OOF predictions are also collected so the
benchmark can plot calibration diagrams and compute global OOF metrics
(mean and bootstrap CI).

The calibration set (15% of films) and the held-out test set (15%) are
not touched by anything in this module. Only the train split is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.baseline_features import GENRE_PREFIX
from src.models.baseline.metrics import (
    CLASSIFICATION_METRICS_HARD,
    CLASSIFICATION_METRICS_PROBA,
    REGRESSION_METRICS,
    bootstrap_ci,
)
from src.models.phase4.families import FamilySpec
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Pre-registered constants (Section 5 of phase4_preregistration.md).
SEED: int = 42
N_INNER_FOLDS: int = 5
N_OUTER_FOLDS: int = 5
N_OUTER_REPEATS: int = 3


# Scoring strings for ``GridSearchCV``. Pre-registered: AUC-ROC for
# classification, neg-RMSE for regression. PR-AUC, F1, log-loss are
# reported but not the search-time selection metric.
INNER_SCORING_REGRESSION: str = "neg_root_mean_squared_error"
INNER_SCORING_CLASSIFICATION: str = "roc_auc"


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class FoldRecord:
    """Per-outer-fold record from repeated CV.

    ``score`` is the score-style prediction (probability for predict_proba
    families, raw decision_function value for SVC). ``hard`` is the
    binary-class hard prediction (classification only; ``None`` for
    regression).
    """
    test_idx: np.ndarray
    y_true: np.ndarray
    score: np.ndarray
    hard: np.ndarray | None
    metrics: dict[str, float]


@dataclass
class CVResult:
    """Aggregate result of repeated CV for one (family, matrix, target).

    Attributes
    ----------
    best_params
        The hyperparameter cell selected by ``GridSearchCV`` on the
        inner 5-fold CV.
    inner_score
        The inner search's best CV score (AUC for classification,
        neg-RMSE for regression).
    fold_records
        Per-outer-fold records (length n_outer_folds * n_outer_repeats).
        Each record carries that fold's test indices, predictions, and
        metric values.
    per_fold_metrics
        ``dict[metric_name, np.ndarray]`` with arrays of length
        n_outer_folds * n_outer_repeats. Input to the paired Bayesian
        test.
    oof_score, oof_hard
        Length-n score / hard predictions, averaged across outer
        repetitions for the score and majority-voted across repetitions
        for the hard label. Used for global OOF metrics and
        calibration plotting.
    n_repeats_per_row
        Number of outer-CV repetitions a row appeared in test (always
        equal to n_outer_repeats for repeated K-fold without holdouts).
    in_sample_score, in_sample_hard
        Score and hard predictions from re-fitting the best estimator
        on the full train split. Yields the in-sample (train-eval-set)
        metrics reported alongside OOF.
    fitted_estimator
        The best estimator fit on the full train split. Saved for the
        per-target winner artifact.
    """
    best_params: dict[str, object]
    inner_score: float
    fold_records: list[FoldRecord]
    per_fold_metrics: dict[str, np.ndarray]
    oof_score: np.ndarray
    oof_hard: np.ndarray | None
    n_repeats_per_row: int
    in_sample_score: np.ndarray
    in_sample_hard: np.ndarray | None
    fitted_estimator: BaseEstimator
    in_sample_metrics: dict[str, float] = field(default_factory=dict)
    oof_metrics_global: dict[str, float] = field(default_factory=dict)
    oof_metrics_global_ci: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition columns into (numeric, one-hot-genre) for ColumnTransformer."""
    one_hot = [c for c in X.columns if c.startswith(GENRE_PREFIX)]
    numeric = [c for c in X.columns if c not in one_hot]
    return numeric, one_hot


def build_pipeline(
    spec: FamilySpec,
    task: Literal["regression", "classification"],
    X: pd.DataFrame,
) -> Pipeline:
    """Construct the (preprocessing + estimator) pipeline for a family.

    The numeric branch uses median imputation and standard scaling for
    families that need them (``spec.needs_scaling=True``: linear, SVMs).
    Tree models pass numeric features through unchanged. The genre
    one-hot dummies always pass through unchanged.
    """
    numeric, one_hot = _split_columns(X)

    if spec.needs_scaling:
        num_branch = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        # HistGB and RF accept NaN natively for HistGB; RF needs imputation
        # but no scaling.
        if spec.name == "random_forest":
            num_branch = Pipeline([("impute", SimpleImputer(strategy="median"))])
        else:
            num_branch = "passthrough"

    pre = ColumnTransformer([
        ("num", num_branch, numeric),
        ("oh", "passthrough", one_hot),
    ])

    if task == "regression":
        est = spec.regressor_factory()
    else:
        est = spec.classifier_factory()

    return Pipeline([("pre", pre), ("model", est)])


# ---------------------------------------------------------------------------
# Hyperparameter search (inner CV)
# ---------------------------------------------------------------------------


def _prefix_grid(grid: dict[str, list]) -> dict[str, list]:
    """Prefix grid keys with ``model__`` for use inside Pipeline GridSearchCV."""
    return {f"model__{k}": v for k, v in grid.items()}


def _compute_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for binary classification.

    Equivalent to ``class_weight="balanced"`` for families that accept
    only ``sample_weight`` (HistGB). Yields weight = n / (2 * n_class)
    per row, matching sklearn's ``compute_sample_weight("balanced", y)``.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.ones(n)
    w = np.empty(n, dtype=float)
    w[y == 1] = n / (2.0 * n_pos)
    w[y == 0] = n / (2.0 * n_neg)
    return w


def run_inner_search(
    spec: FamilySpec,
    task: Literal["regression", "classification"],
    X: pd.DataFrame,
    y: pd.Series,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Run ``GridSearchCV`` (inner 5-fold) and return the fitted searcher.

    Sample-weight balancing for HistGB classification is handled by
    passing ``model__sample_weight`` through ``fit_params``. The weight
    routing applies at training time; scoring on held-out folds remains
    unweighted, which is the apples-to-apples comparison the
    pre-registration intends.
    """
    pipe = build_pipeline(spec, task, X)
    grid = (
        spec.regression_grid if task == "regression" else spec.classification_grid
    )
    prefixed = _prefix_grid(grid)

    if task == "regression":
        cv = RepeatedKFold(
            n_splits=N_INNER_FOLDS, n_repeats=1, random_state=SEED,
        )
        scoring = INNER_SCORING_REGRESSION
    else:
        cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED,
        )
        scoring = INNER_SCORING_CLASSIFICATION

    search = GridSearchCV(
        pipe,
        param_grid=prefixed,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )

    fit_params: dict[str, np.ndarray] = {}
    if (
        task == "classification"
        and spec.balancing == "sample_weight"
    ):
        fit_params["model__sample_weight"] = _compute_sample_weight(y.values)

    search.fit(X, y, **fit_params)
    return search


# ---------------------------------------------------------------------------
# Repeated outer CV
# ---------------------------------------------------------------------------


def _predict_classification(
    estimator: BaseEstimator, X: pd.DataFrame, score_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hard, score) for a fitted classification estimator."""
    hard = estimator.predict(X)
    if score_method == "predict_proba":
        score = estimator.predict_proba(X)[:, 1]
    elif score_method == "decision_function":
        score = estimator.decision_function(X)
    else:
        raise ValueError(f"Unknown score_method: {score_method!r}")
    return hard, score


def _compute_fold_metrics_classification(
    y_true: np.ndarray,
    score: np.ndarray,
    hard: np.ndarray,
    score_method: str,
) -> dict[str, float]:
    """Compute the four classification metrics on one fold."""
    out: dict[str, float] = {}
    for name, fn in CLASSIFICATION_METRICS_PROBA.items():
        if name == "log_loss" and score_method != "predict_proba":
            out[name] = float("nan")
            continue
        try:
            out[name] = float(fn(y_true, score))
        except ValueError:
            out[name] = float("nan")
    for name, fn in CLASSIFICATION_METRICS_HARD.items():
        try:
            out[name] = float(fn(y_true, hard))
        except ValueError:
            out[name] = float("nan")
    return out


def _compute_fold_metrics_regression(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute the four regression metrics on one fold."""
    return {name: float(fn(y_true, y_pred)) for name, fn in REGRESSION_METRICS.items()}


def evaluate_repeated_cv(
    spec: FamilySpec,
    task: Literal["regression", "classification"],
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, object],
    n_jobs: int = -1,
) -> tuple[
    list[FoldRecord],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray | None,
]:
    """Run repeated stratified 5-fold CV with the chosen hyperparameters.

    Returns the per-fold records, the per-fold metric arrays (15-length
    each, indexed by metric name), and the OOF score / hard predictions
    averaged across the 3 outer repetitions.

    The chosen hyperparameters from ``run_inner_search`` are applied
    via ``Pipeline.set_params``; the resulting estimator is re-fit
    independently on each of the 15 outer folds.
    """
    pipe = build_pipeline(spec, task, X)
    pipe = pipe.set_params(**best_params)

    if task == "regression":
        cv = RepeatedKFold(
            n_splits=N_OUTER_FOLDS,
            n_repeats=N_OUTER_REPEATS,
            random_state=SEED,
        )
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=N_OUTER_FOLDS,
            n_repeats=N_OUTER_REPEATS,
            random_state=SEED,
        )

    n = len(y)
    score_accum = np.zeros(n)
    hard_accum: np.ndarray | None
    if task == "classification":
        # Average score across repetitions; majority-vote hard label.
        hard_accum = np.zeros(n)
    else:
        hard_accum = None
    counts = np.zeros(n, dtype=int)

    fold_records: list[FoldRecord] = []
    metric_acc: dict[str, list[float]] = {}
    y_arr = y.values

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_arr)):
        pipe_fit = clone(pipe)

        # HistGB sample_weight handling: same routing as inner search.
        # The fold's training subset gets the corresponding weight subset.
        fit_params: dict[str, np.ndarray] = {}
        if (
            task == "classification"
            and spec.balancing == "sample_weight"
        ):
            full_weights = _compute_sample_weight(y_arr)
            fit_params["model__sample_weight"] = full_weights[train_idx]

        pipe_fit.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)

        if task == "regression":
            pred = pipe_fit.predict(X.iloc[test_idx])
            score_accum[test_idx] += pred
            counts[test_idx] += 1
            metrics = _compute_fold_metrics_regression(y_arr[test_idx], pred)
            fold_records.append(FoldRecord(
                test_idx=test_idx, y_true=y_arr[test_idx],
                score=pred, hard=None, metrics=metrics,
            ))
        else:
            hard, score = _predict_classification(
                pipe_fit, X.iloc[test_idx], spec.score_method,
            )
            score_accum[test_idx] += score
            assert hard_accum is not None
            hard_accum[test_idx] += hard
            counts[test_idx] += 1
            metrics = _compute_fold_metrics_classification(
                y_arr[test_idx], score, hard, spec.score_method,
            )
            fold_records.append(FoldRecord(
                test_idx=test_idx, y_true=y_arr[test_idx],
                score=score, hard=hard, metrics=metrics,
            ))

        for name, value in metrics.items():
            metric_acc.setdefault(name, []).append(value)

        logger.debug(
            "Fold %d/%d done (%s, %s): %s",
            fold_idx + 1, cv.get_n_splits(), spec.name, task,
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not np.isnan(v)),
        )

    # Average the per-row score across repetitions.
    safe_counts = np.where(counts == 0, 1, counts)
    oof_score = score_accum / safe_counts
    if hard_accum is not None:
        # Majority vote: round the average (1 if >0.5).
        oof_hard = (hard_accum / safe_counts >= 0.5).astype(int)
    else:
        oof_hard = None

    per_fold = {name: np.asarray(values) for name, values in metric_acc.items()}
    return fold_records, per_fold, oof_score, oof_hard


# ---------------------------------------------------------------------------
# In-sample (full-train) refit
# ---------------------------------------------------------------------------


def fit_full_train(
    spec: FamilySpec,
    task: Literal["regression", "classification"],
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, object],
) -> tuple[BaseEstimator, np.ndarray, np.ndarray | None]:
    """Fit the chosen pipeline on the full train split.

    Returns the fitted pipeline plus its in-sample score and hard
    predictions (hard is ``None`` for regression). Used both to record
    in-sample metrics and to produce the saved model artifact.
    """
    pipe = build_pipeline(spec, task, X)
    pipe = pipe.set_params(**best_params)

    fit_params: dict[str, np.ndarray] = {}
    if (
        task == "classification"
        and spec.balancing == "sample_weight"
    ):
        fit_params["model__sample_weight"] = _compute_sample_weight(y.values)

    pipe.fit(X, y, **fit_params)

    if task == "regression":
        return pipe, pipe.predict(X), None
    hard, score = _predict_classification(pipe, X, spec.score_method)
    return pipe, score, hard


# ---------------------------------------------------------------------------
# Top-level entry point per (family, matrix, target)
# ---------------------------------------------------------------------------


def evaluate_family_target(
    spec: FamilySpec,
    task: Literal["regression", "classification"],
    X: pd.DataFrame,
    y: pd.Series,
    n_jobs: int = -1,
) -> CVResult:
    """End-to-end evaluation for one (family, matrix, target).

    Pipeline:

    1. Inner ``GridSearchCV`` (5-fold) selects the best hyperparameter
       cell.
    2. The best cell is re-fit across 15 outer folds (3x5 repeated CV)
       to produce per-fold metric arrays (input to paired Bayesian
       comparison) and aggregated OOF predictions (input to global OOF
       metrics and calibration plot).
    3. The best cell is also re-fit on the full train split to produce
       in-sample metrics and the saved model artifact.
    """
    logger.info(
        "Inner GridSearchCV: family=%s task=%s X.shape=%s",
        spec.name, task, X.shape,
    )
    search = run_inner_search(spec, task, X, y, n_jobs=n_jobs)
    best_params = search.best_params_
    inner_score = float(search.best_score_)
    logger.info(
        "Inner CV best: family=%s task=%s score=%.4f params=%s",
        spec.name, task, inner_score, best_params,
    )

    logger.info(
        "Repeated outer CV (%dx%d): family=%s task=%s",
        N_OUTER_REPEATS, N_OUTER_FOLDS, spec.name, task,
    )
    fold_records, per_fold, oof_score, oof_hard = evaluate_repeated_cv(
        spec, task, X, y, best_params, n_jobs=n_jobs,
    )

    fitted, in_sample_score, in_sample_hard = fit_full_train(
        spec, task, X, y, best_params,
    )

    result = CVResult(
        best_params=dict(best_params),
        inner_score=inner_score,
        fold_records=fold_records,
        per_fold_metrics=per_fold,
        oof_score=oof_score,
        oof_hard=oof_hard,
        n_repeats_per_row=N_OUTER_REPEATS,
        in_sample_score=in_sample_score,
        in_sample_hard=in_sample_hard,
        fitted_estimator=fitted,
    )

    # Compute in-sample and global-OOF metrics with bootstrap CIs.
    y_arr = y.values
    if task == "regression":
        result.in_sample_metrics = _compute_fold_metrics_regression(
            y_arr, in_sample_score,
        )
        result.oof_metrics_global = _compute_fold_metrics_regression(
            y_arr, oof_score,
        )
        for name, fn in REGRESSION_METRICS.items():
            ci = bootstrap_ci(fn, y_arr, oof_score, n_iter=1000, seed=SEED)
            result.oof_metrics_global_ci[name] = ci
    else:
        result.in_sample_metrics = _compute_fold_metrics_classification(
            y_arr, in_sample_score, in_sample_hard, spec.score_method,
        )
        result.oof_metrics_global = _compute_fold_metrics_classification(
            y_arr, oof_score, oof_hard, spec.score_method,
        )
        for name, fn in CLASSIFICATION_METRICS_PROBA.items():
            if name == "log_loss" and spec.score_method != "predict_proba":
                result.oof_metrics_global_ci[name] = (float("nan"), float("nan"))
                continue
            ci = bootstrap_ci(fn, y_arr, oof_score, n_iter=1000, seed=SEED)
            result.oof_metrics_global_ci[name] = ci
        for name, fn in CLASSIFICATION_METRICS_HARD.items():
            ci = bootstrap_ci(fn, y_arr, oof_hard, n_iter=1000, seed=SEED)
            result.oof_metrics_global_ci[name] = ci

    return result
