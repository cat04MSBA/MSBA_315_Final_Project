"""Baseline trainer for Phase 3 across four model families.

Trains the same feature configurations under four model families with
distinct inductive biases so that an ablation row's lift number can be
compared across paradigms. The four families:

* ``linear``: L2-regularized linear (RidgeCV for regression,
  LogisticRegressionCV for classification). Captures linear additive
  signal. Needs median imputation and z-score scaling.
* ``histgb``: histogram-based gradient-boosted trees
  (HistGradientBoostingRegressor / Classifier). Captures non-linear
  global signal with feature interactions. Handles NaN natively and is
  invariant to monotonic transforms; no preprocessing needed in the
  numeric branch.
* ``knn``: k-nearest-neighbours (KNeighborsRegressor / Classifier).
  Captures local-neighbourhood structure. Non-parametric. Needs both
  imputation and scaling because distances are computed on standardized
  features.
* ``svm``: support-vector machine with RBF kernel
  (SVR / SVC). Captures non-linear kernel-induced signal. Needs both
  imputation and scaling. SVC produces decision-function scores
  rather than calibrated probabilities, which means log-loss is not
  computed for the SVM family (left as ``NaN``); AUC-ROC, PR-AUC,
  and F1 are unaffected (AUC and PR-AUC depend only on the ordering
  of scores; F1 uses the sign-of-decision hard prediction).

Two evaluation sets are reported per (feature_set, model_family,
target, metric):

* ``train``: in-sample fit on the entire train split. Captures how
  well the model can fit the training data.
* ``oof``: out-of-fold predictions from 5-fold CV on the train split.
  Captures generalization error within train.

The gap between the two indicates overfitting. The held-out 15%
calibration set and 15% test set are not touched by this trainer
(reserved for Phase 5 and Phase 8 respectively).

The output CSV schema is one row per
``(feature_set, model_family, eval_set, target, metric)`` combination.

Run from the project root::

    python -m src.models.baseline.train

Idempotent.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.features.baseline_features import (
    BaselineFeatureConfig,
    GENRE_PREFIX,
    build_baseline_features,
)
from src.features.targets import (
    CLASSIFICATION_TARGETS,
    LOG_ROI_COL,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.baseline.metrics import (
    CLASSIFICATION_METRICS_HARD,
    CLASSIFICATION_METRICS_PROBA,
    REGRESSION_METRICS,
    bootstrap_ci,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# All four model families, in the order they appear in the output CSV.
MODEL_FAMILIES: tuple[str, ...] = ("linear", "histgb", "knn", "svm")

# Evaluation set labels: in-sample fit on full train, plus OOF CV on train.
EVAL_SETS: tuple[str, ...] = ("train", "oof")


@dataclass(frozen=True)
class BaselineTrainConfig:
    """Knobs for the multi-family baseline trainer."""

    n_cv_folds: int = 5
    bootstrap_iter: int = 1000
    bootstrap_alpha: float = 0.05
    seed: int = 42

    # Linear (Ridge / Logistic L2) hyperparameters.
    ridge_alphas: tuple[float, ...] = tuple(np.logspace(-3, 3, 13))
    logistic_Cs: tuple[float, ...] = tuple(np.logspace(-3, 3, 13))

    # Histogram gradient boosting hyperparameters.
    histgb_max_iter: int = 300
    histgb_max_depth: int = 4
    histgb_learning_rate: float = 0.05

    # KNN hyperparameters.
    knn_n_neighbors: int = 20
    knn_weights: str = "distance"

    # SVM-RBF hyperparameters.
    svm_C: float = 1.0
    svm_gamma: str = "scale"

    in_corpus: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    )
    in_splits: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    )
    out_table: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR / "phase3a_baseline.csv"
    )


# ---------------------------------------------------------------------------
# Per-family pipeline construction
# ---------------------------------------------------------------------------


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition feature columns into (numeric, one-hot)."""
    one_hot = [c for c in X.columns if c.startswith(GENRE_PREFIX)]
    numeric = [c for c in X.columns if c not in one_hot]
    return numeric, one_hot


def _numeric_branch(family: str):
    """Return the numeric-branch transformer for ``family``."""
    if family == "histgb":
        return "passthrough"
    if family in ("linear", "knn", "svm"):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    raise ValueError(f"Unknown model family: {family!r}")


def _build_regression_pipeline(
    family: str,
    numeric: list[str],
    one_hot: list[str],
    cfg: BaselineTrainConfig,
) -> Pipeline:
    pre = ColumnTransformer([
        ("num", _numeric_branch(family), numeric),
        ("oh", "passthrough", one_hot),
    ])
    if family == "linear":
        model = RidgeCV(alphas=list(cfg.ridge_alphas))
    elif family == "histgb":
        model = HistGradientBoostingRegressor(
            max_iter=cfg.histgb_max_iter,
            max_depth=cfg.histgb_max_depth,
            learning_rate=cfg.histgb_learning_rate,
            early_stopping=True,
            random_state=cfg.seed,
        )
    elif family == "knn":
        model = KNeighborsRegressor(
            n_neighbors=cfg.knn_n_neighbors,
            weights=cfg.knn_weights,
        )
    elif family == "svm":
        model = SVR(kernel="rbf", C=cfg.svm_C, gamma=cfg.svm_gamma)
    else:
        raise ValueError(f"Unknown model family: {family!r}")
    return Pipeline([("pre", pre), ("model", model)])


def _build_classification_pipeline(
    family: str,
    numeric: list[str],
    one_hot: list[str],
    cfg: BaselineTrainConfig,
) -> tuple[Pipeline, str]:
    """Build (pipeline, score_method) for a classification family.

    ``score_method`` is the sklearn method name used to obtain
    probability-style scores (``"predict_proba"`` for the families
    that produce probabilities; ``"decision_function"`` for SVC).
    Hard predictions are always obtained via ``predict``.
    """
    pre = ColumnTransformer([
        ("num", _numeric_branch(family), numeric),
        ("oh", "passthrough", one_hot),
    ])
    if family == "linear":
        inner_cv = StratifiedKFold(
            n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.seed,
        )
        model = LogisticRegressionCV(
            Cs=list(cfg.logistic_Cs),
            cv=inner_cv,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            scoring="roc_auc",
            random_state=cfg.seed,
        )
        score_method = "predict_proba"
    elif family == "histgb":
        model = HistGradientBoostingClassifier(
            max_iter=cfg.histgb_max_iter,
            max_depth=cfg.histgb_max_depth,
            learning_rate=cfg.histgb_learning_rate,
            early_stopping=True,
            random_state=cfg.seed,
        )
        score_method = "predict_proba"
    elif family == "knn":
        model = KNeighborsClassifier(
            n_neighbors=cfg.knn_n_neighbors,
            weights=cfg.knn_weights,
        )
        score_method = "predict_proba"
    elif family == "svm":
        model = SVC(kernel="rbf", C=cfg.svm_C, gamma=cfg.svm_gamma)
        score_method = "decision_function"
    else:
        raise ValueError(f"Unknown model family: {family!r}")
    return Pipeline([("pre", pre), ("model", model)]), score_method


# ---------------------------------------------------------------------------
# Manual CV helpers (return both hard and score predictions per fold)
# ---------------------------------------------------------------------------


def _cv_predict_regression(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv,
) -> np.ndarray:
    """Manual K-fold CV for regression. Returns OOF predictions."""
    n = len(y)
    oof = np.zeros(n)
    for train_idx, test_idx in cv.split(X, y):
        pipe_fit = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = pipe_fit.predict(X.iloc[test_idx])
    return oof


def _cv_predict_classification(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv, score_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual K-fold CV for classification. Returns (hard, score) OOF predictions.

    ``score`` is probability-of-positive-class for ``predict_proba``
    families and the raw decision-function value for SVC. Both are
    valid inputs to AUC-style metrics; only ``predict_proba`` outputs
    are valid for log-loss (see :func:`_compute_classification_metrics`).
    """
    n = len(y)
    hard = np.zeros(n, dtype=int)
    score = np.zeros(n)
    for train_idx, test_idx in cv.split(X, y):
        pipe_fit = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        hard[test_idx] = pipe_fit.predict(X.iloc[test_idx])
        if score_method == "predict_proba":
            score[test_idx] = pipe_fit.predict_proba(X.iloc[test_idx])[:, 1]
        elif score_method == "decision_function":
            score[test_idx] = pipe_fit.decision_function(X.iloc[test_idx])
        else:
            raise ValueError(f"Unknown score_method: {score_method!r}")
    return hard, score


def _full_train_predict_classification(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, score_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on full train, predict on full train. Returns (hard, score)."""
    pipe_fit = clone(pipe).fit(X, y)
    hard = pipe_fit.predict(X)
    if score_method == "predict_proba":
        score = pipe_fit.predict_proba(X)[:, 1]
    elif score_method == "decision_function":
        score = pipe_fit.decision_function(X)
    else:
        raise ValueError(f"Unknown score_method: {score_method!r}")
    return hard, score


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: BaselineTrainConfig,
) -> list[dict]:
    rows: list[dict] = []
    for name, fn in REGRESSION_METRICS.items():
        value = float(fn(y_true, y_pred))
        ci_lo, ci_hi = bootstrap_ci(
            fn, y_true, y_pred,
            n_iter=cfg.bootstrap_iter, seed=cfg.seed, alpha=cfg.bootstrap_alpha,
        )
        rows.append({
            "metric": name,
            "value": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_alpha": cfg.bootstrap_alpha,
            "bootstrap_iter": cfg.bootstrap_iter,
            "n": len(y_true),
        })
    return rows


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_hard: np.ndarray,
    y_pred_score: np.ndarray,
    score_method: str,
    cfg: BaselineTrainConfig,
) -> list[dict]:
    rows: list[dict] = []
    # Probability-style metrics. log_loss only valid for predict_proba scores
    # (decision_function values from SVC are not in [0, 1]).
    for name, fn in CLASSIFICATION_METRICS_PROBA.items():
        if name == "log_loss" and score_method != "predict_proba":
            rows.append({
                "metric": name,
                "value": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "ci_alpha": cfg.bootstrap_alpha,
                "bootstrap_iter": cfg.bootstrap_iter,
                "n": len(y_true),
                "note": "skipped: score_method=decision_function (no probabilities)",
            })
            continue
        value = float(fn(y_true, y_pred_score))
        ci_lo, ci_hi = bootstrap_ci(
            fn, y_true, y_pred_score,
            n_iter=cfg.bootstrap_iter, seed=cfg.seed, alpha=cfg.bootstrap_alpha,
        )
        rows.append({
            "metric": name,
            "value": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_alpha": cfg.bootstrap_alpha,
            "bootstrap_iter": cfg.bootstrap_iter,
            "n": len(y_true),
        })
    # Hard-prediction metrics.
    for name, fn in CLASSIFICATION_METRICS_HARD.items():
        value = float(fn(y_true, y_pred_hard))
        ci_lo, ci_hi = bootstrap_ci(
            fn, y_true, y_pred_hard,
            n_iter=cfg.bootstrap_iter, seed=cfg.seed, alpha=cfg.bootstrap_alpha,
        )
        rows.append({
            "metric": name,
            "value": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_alpha": cfg.bootstrap_alpha,
            "bootstrap_iter": cfg.bootstrap_iter,
            "n": len(y_true),
        })
    return rows


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_feature_set(
    df_train: pd.DataFrame,
    feature_cfg: BaselineFeatureConfig,
    train_cfg: BaselineTrainConfig,
    *,
    set_name: str,
    families: tuple[str, ...] = MODEL_FAMILIES,
) -> list[dict]:
    """Train and evaluate baselines for one feature configuration across families.

    For each (family, target) combination, trains a model under 5-fold
    CV on the train split and reports BOTH in-sample (full-train fit)
    and out-of-fold metrics. Returns one row per
    (model_family, eval_set, target, metric) combination.
    """
    X = build_baseline_features(df_train, feature_cfg)
    numeric, one_hot = _split_columns(X)
    rows: list[dict] = []

    for family in families:
        for target in REGRESSION_TARGETS:
            pipe = _build_regression_pipeline(family, numeric, one_hot, train_cfg)
            cv = KFold(
                n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed,
            )
            y = df_train[target]
            y_true = y.values

            # In-sample (train).
            pipe_full = clone(pipe).fit(X, y)
            in_sample = pipe_full.predict(X)
            for r in _compute_regression_metrics(y_true, in_sample, train_cfg):
                r.update({
                    "feature_set": set_name, "model_family": family,
                    "eval_set": "train", "target": target, "task": "regression",
                })
                rows.append(r)
                logger.info(
                    "%s | %s | train | %s | %s = %.4f",
                    set_name, family, target, r["metric"], r["value"],
                )

            # OOF (validation).
            oof = _cv_predict_regression(pipe, X, y, cv)
            for r in _compute_regression_metrics(y_true, oof, train_cfg):
                r.update({
                    "feature_set": set_name, "model_family": family,
                    "eval_set": "oof", "target": target, "task": "regression",
                })
                rows.append(r)
                logger.info(
                    "%s | %s | oof   | %s | %s = %.4f [CI %.4f, %.4f]",
                    set_name, family, target, r["metric"], r["value"],
                    r["ci_lo"], r["ci_hi"],
                )

        for target in CLASSIFICATION_TARGETS:
            pipe, score_method = _build_classification_pipeline(
                family, numeric, one_hot, train_cfg,
            )
            cv = StratifiedKFold(
                n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed,
            )
            y = df_train[target].astype(int)
            y_true = y.values

            # In-sample (train).
            hard_train, score_train = _full_train_predict_classification(
                pipe, X, y, score_method,
            )
            for r in _compute_classification_metrics(
                y_true, hard_train, score_train, score_method, train_cfg,
            ):
                r.update({
                    "feature_set": set_name, "model_family": family,
                    "eval_set": "train", "target": target, "task": "classification",
                })
                rows.append(r)

            # OOF (validation).
            hard_oof, score_oof = _cv_predict_classification(
                pipe, X, y, cv, score_method,
            )
            for r in _compute_classification_metrics(
                y_true, hard_oof, score_oof, score_method, train_cfg,
            ):
                r.update({
                    "feature_set": set_name, "model_family": family,
                    "eval_set": "oof", "target": target, "task": "classification",
                })
                rows.append(r)

            # Log only OOF metrics to avoid doubling the log output.
            for r in rows[-len(CLASSIFICATION_METRICS_PROBA) - len(CLASSIFICATION_METRICS_HARD):]:
                if not np.isnan(r["value"]):
                    logger.info(
                        "%s | %s | %s   | %s | %s = %.4f [CI %.4f, %.4f]",
                        set_name, family, r["eval_set"], target, r["metric"],
                        r["value"], r["ci_lo"], r["ci_hi"],
                    )

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint: train baselines across 4 families, save table.

    The original Phase 3 brief's R²-based escalation threshold is no
    longer applied because R² has been removed from the reported
    metric set (planning conversation 2026-05-03). The classification
    AUC-ROC threshold (0.55) is also no longer gating; per-group
    ablation lifts against the floor are the primary signal.
    """
    train_cfg = BaselineTrainConfig()
    paths.ensure_dirs()

    logger.info("Loading master corpus and split assignments")
    df_full = pd.read_parquet(train_cfg.in_corpus)
    splits = pd.read_parquet(train_cfg.in_splits)

    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)
    logger.info(
        "Training on %d films (train split); families: %s",
        len(df_train), ", ".join(MODEL_FAMILIES),
    )

    rows: list[dict] = []
    configs = [
        ("dialogue_only", BaselineFeatureConfig(include_log_budget=False)),
        ("with_budget",   BaselineFeatureConfig(include_log_budget=True)),
        ("dialogue_only_logged", BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
        )),
        ("with_budget_logged",   BaselineFeatureConfig(
            include_log_budget=True,
            log_transform_structural=True,
            include_log_runtime=True,
        )),
    ]
    for name, fc in configs:
        logger.info("Evaluating feature config: %s", name)
        rows.extend(evaluate_feature_set(df_train, fc, train_cfg, set_name=name))

    out = pd.DataFrame(rows)[
        [
            "feature_set",
            "model_family",
            "eval_set",
            "target",
            "task",
            "metric",
            "value",
            "ci_lo",
            "ci_hi",
            "ci_alpha",
            "bootstrap_iter",
            "n",
        ]
    ]
    out.to_csv(train_cfg.out_table, index=False)
    logger.info("Saved baseline table (%d rows)", len(out))


if __name__ == "__main__":
    main()
