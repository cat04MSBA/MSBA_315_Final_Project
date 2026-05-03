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
  imputation and scaling. SVC uses ``decision_function`` rather than
  ``predict_proba`` so AUC is computed without the slow internal Platt
  calibration that ``probability=True`` triggers.

The four families are run on the same feature matrix per feature
configuration so the comparison is direct. The output CSV has one row
per ``(feature_set, model_family, target, metric)`` combination.

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
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
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    bootstrap_ci,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# All four model families, in the order they appear in the output CSV.
MODEL_FAMILIES: tuple[str, ...] = ("linear", "histgb", "knn", "svm")

# The historical reference family used for the brief's escalation
# threshold check (Phase 3 brief Section 3 Task 2). Other families'
# numbers are reported but do not gate the escalation.
THRESHOLD_REFERENCE_FAMILY: str = "linear"


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

    # Histogram gradient boosting hyperparameters. Conservative defaults
    # for n=1,199: shallow trees, modest learning rate, internal
    # early-stopping to prevent overfitting without explicit tuning.
    histgb_max_iter: int = 300
    histgb_max_depth: int = 4
    histgb_learning_rate: float = 0.05

    # KNN hyperparameters. n_neighbors near sqrt(n_train) (~35 for the
    # 1,199-film train split) is the textbook starting point; 20 with
    # distance-weighting strikes a balance between local structure and
    # variance reduction.
    knn_n_neighbors: int = 20
    knn_weights: str = "distance"

    # SVM-RBF hyperparameters. Default C=1.0 and gamma="scale" are
    # sklearn's recommended starting values; both are stable on n=1,199
    # without explicit tuning.
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
    """Return the numeric-branch transformer for ``family``.

    ``histgb`` requires no preprocessing because the algorithm handles
    NaN natively and is invariant to monotonic transforms. Other
    families need median imputation followed by z-score scaling: KNN
    and SVM-RBF both compute distances on standardized features, and
    L2 linear models need scaling for the regularization to operate
    on comparable feature magnitudes.
    """
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
) -> tuple[Pipeline, str]:
    """Build (pipeline, cross_val_predict_method) for a regression family."""
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
    return Pipeline([("pre", pre), ("model", model)]), "predict"


def _build_classification_pipeline(
    family: str,
    numeric: list[str],
    one_hot: list[str],
    cfg: BaselineTrainConfig,
) -> tuple[Pipeline, str]:
    """Build (pipeline, cross_val_predict_method) for a classification family.

    SVM uses ``decision_function`` rather than ``predict_proba`` because
    AUC and PR-AUC depend only on the relative ordering of scores, and
    SVC's ``probability=True`` would trigger an internal 5-fold
    Platt-scaling CV that multiplies training time five-fold without
    changing the score ordering.
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
        method = "predict_proba"
    elif family == "histgb":
        model = HistGradientBoostingClassifier(
            max_iter=cfg.histgb_max_iter,
            max_depth=cfg.histgb_max_depth,
            learning_rate=cfg.histgb_learning_rate,
            early_stopping=True,
            random_state=cfg.seed,
        )
        method = "predict_proba"
    elif family == "knn":
        model = KNeighborsClassifier(
            n_neighbors=cfg.knn_n_neighbors,
            weights=cfg.knn_weights,
        )
        method = "predict_proba"
    elif family == "svm":
        model = SVC(kernel="rbf", C=cfg.svm_C, gamma=cfg.svm_gamma)
        method = "decision_function"
    else:
        raise ValueError(f"Unknown model family: {family!r}")
    return Pipeline([("pre", pre), ("model", model)]), method


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    cv,
    cfg: BaselineTrainConfig,
    *,
    method: str,
) -> tuple[np.ndarray, list[dict]]:
    """Run CV, compute every metric in ``metrics``, return rows.

    ``method`` is one of ``"predict"`` (regression), ``"predict_proba"``
    (most classifiers), or ``"decision_function"`` (SVC). For
    ``predict_proba`` the positive-class column is sliced off; for the
    other two the raw output is used directly.
    """
    raw = cross_val_predict(pipe, X, y, cv=cv, method=method)
    if method == "predict_proba":
        y_pred = raw[:, 1]
    else:
        y_pred = raw
    y_true = y.values

    rows: list[dict] = []
    for name, fn in metrics.items():
        value = float(fn(y_true, y_pred))
        ci_lo, ci_hi = bootstrap_ci(
            fn, y_true, y_pred,
            n_iter=cfg.bootstrap_iter,
            seed=cfg.seed,
            alpha=cfg.bootstrap_alpha,
        )
        rows.append({
            "metric": name,
            "value": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_alpha": cfg.bootstrap_alpha,
            "bootstrap_iter": cfg.bootstrap_iter,
            "n_train": len(y_true),
        })
    return y_pred, rows


def evaluate_feature_set(
    df_train: pd.DataFrame,
    feature_cfg: BaselineFeatureConfig,
    train_cfg: BaselineTrainConfig,
    *,
    set_name: str,
    families: tuple[str, ...] = MODEL_FAMILIES,
) -> list[dict]:
    """Train and evaluate baselines for one feature configuration across families.

    Iterates over the requested model families and the three targets,
    fits each combination under 5-fold CV on the training split, and
    returns one row per (model_family, target, metric) combination.
    """
    X = build_baseline_features(df_train, feature_cfg)
    numeric, one_hot = _split_columns(X)
    rows: list[dict] = []

    for family in families:
        for target in REGRESSION_TARGETS:
            pipe, method = _build_regression_pipeline(family, numeric, one_hot, train_cfg)
            cv = KFold(
                n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed,
            )
            _, target_rows = _evaluate(
                pipe, X, df_train[target], REGRESSION_METRICS, cv, train_cfg, method=method,
            )
            for r in target_rows:
                r.update({
                    "feature_set": set_name,
                    "model_family": family,
                    "target": target,
                    "task": "regression",
                })
                rows.append(r)
                logger.info(
                    "%s | %s | %s | %s = %.4f [CI %.4f, %.4f]",
                    set_name, family, target, r["metric"], r["value"], r["ci_lo"], r["ci_hi"],
                )

        for target in CLASSIFICATION_TARGETS:
            pipe, method = _build_classification_pipeline(family, numeric, one_hot, train_cfg)
            cv = StratifiedKFold(
                n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed,
            )
            _, target_rows = _evaluate(
                pipe, X, df_train[target].astype(int),
                CLASSIFICATION_METRICS, cv, train_cfg, method=method,
            )
            for r in target_rows:
                r.update({
                    "feature_set": set_name,
                    "model_family": family,
                    "target": target,
                    "task": "classification",
                })
                rows.append(r)
                logger.info(
                    "%s | %s | %s | %s = %.4f [CI %.4f, %.4f]",
                    set_name, family, target, r["metric"], r["value"], r["ci_lo"], r["ci_hi"],
                )

    return rows


# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------


DEPLOYABLE_FEATURE_SETS: tuple[str, ...] = ("dialogue_only", "dialogue_only_logged")


def _check_thresholds(rows: list[dict]) -> dict[str, bool]:
    """Apply the brief's escalation thresholds to the linear reference family.

    R² < 0.05 (regression) or AUC-ROC < 0.55 (classification) on a
    deployable (no-budget) feature set under the linear family triggers
    escalation per Phase 3 brief Section 3 Task 2. Other families'
    numbers are reported alongside but do not gate the escalation: the
    linear family is the historical reference the threshold was set
    against.
    """
    triggers: dict[str, bool] = {}
    for r in rows:
        if r["feature_set"] not in DEPLOYABLE_FEATURE_SETS:
            continue
        if r["model_family"] != THRESHOLD_REFERENCE_FAMILY:
            continue
        key_prefix = r["feature_set"]
        if r["target"] == LOG_ROI_COL and r["metric"] == "r2":
            triggers[f"{key_prefix}_regression_below_floor"] = r["value"] < 0.05
        if r["task"] == "classification" and r["metric"] == "auc_roc":
            triggers[f"{key_prefix}_{r['target']}_below_floor"] = r["value"] < 0.55
    return triggers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint: train baselines across 4 families, save table, log threshold check."""
    train_cfg = BaselineTrainConfig()
    paths.ensure_dirs()

    logger.info("Loading master corpus and split assignments")
    df_full = pd.read_parquet(train_cfg.in_corpus)
    splits = pd.read_parquet(train_cfg.in_splits)

    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)
    logger.info("Training on %d films (train split); families: %s",
                len(df_train), ", ".join(MODEL_FAMILIES))

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
            "target",
            "task",
            "metric",
            "value",
            "ci_lo",
            "ci_hi",
            "ci_alpha",
            "bootstrap_iter",
            "n_train",
        ]
    ]
    out.to_csv(train_cfg.out_table, index=False)
    logger.info("Saved baseline table (%d rows)", len(out))

    triggers = _check_thresholds(rows)
    if any(triggers.values()):
        flagged = ", ".join(name for name, hit in triggers.items() if hit)
        logger.warning("Phase 3a thresholds (linear reference) tripped: %s", flagged)
    else:
        logger.info(
            "Phase 3a thresholds OK on linear reference (R^2 >= 0.05, AUC-ROC >= 0.55)"
        )


if __name__ == "__main__":
    main()
