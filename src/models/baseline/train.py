"""Phase 3a baseline trainer.

Trains a simple linear baseline on each of the three Phase 3 targets,
under four feature configurations:

* ``dialogue_only`` — original deployable baseline. Raw structural
  counts z-scored without log transform; no ``log_runtime``.
* ``with_budget`` — original sanity-check; ``dialogue_only`` plus
  ``log_budget``.
* ``dialogue_only_logged`` — revised deployable baseline (planning
  conversation 2026-05-03). ``log1p`` applied to the heavy-tailed
  structural counts before z-scoring, and ``log_runtime`` added.
* ``with_budget_logged`` — revised sanity-check; same revision plus
  ``log_budget``.

The original (un-logged) configurations are kept in the output table
so the final report can show the floor before and after the revision.

Targets:

* ``log_roi`` (regression) — :class:`sklearn.linear_model.RidgeCV` over
  a log-spaced alpha grid; alpha selection by leave-one-out GCV.
* ``roi_gt_1`` (classification) — :class:`sklearn.linear_model.LogisticRegressionCV`
  with L2 penalty over a log-spaced C grid; C selection by inner
  5-fold stratified CV optimizing AUC-ROC.
* ``roi_gt_2`` (classification) — same as ``roi_gt_1``.

Cross-validation: 5-fold (stratified by target for classification,
plain KFold for regression) on the **training split only**. Out-of-fold
predictions are concatenated across folds and used for both the
headline metric and the bootstrap CI.

The test set is not touched. The calibration set is not touched
(reserved for Phase 5).

Output: ``reports/tables/phase3a_baseline.csv``, one row per
``[feature_set, target, metric]`` combination.

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
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


@dataclass(frozen=True)
class BaselineTrainConfig:
    """Knobs for the Phase 3a baseline trainer."""
    n_cv_folds: int = 5
    bootstrap_iter: int = 1000
    bootstrap_alpha: float = 0.05

    # Regularization grids, log-spaced over six decades.
    ridge_alphas: tuple[float, ...] = tuple(np.logspace(-3, 3, 13))
    logistic_Cs: tuple[float, ...] = tuple(np.logspace(-3, 3, 13))

    seed: int = 42

    in_corpus: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    )
    in_splits: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    )
    out_table: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR / "phase3a_baseline.csv"
    )


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition feature columns into (numeric, one-hot)."""
    one_hot = [c for c in X.columns if c.startswith(GENRE_PREFIX)]
    numeric = [c for c in X.columns if c not in one_hot]
    return numeric, one_hot


def _build_regression_pipeline(
    numeric: list[str], one_hot: list[str], cfg: BaselineTrainConfig
) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("oh", "passthrough", one_hot),
        ]
    )
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=list(cfg.ridge_alphas)))])


def _build_classification_pipeline(
    numeric: list[str], one_hot: list[str], cfg: BaselineTrainConfig
) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("oh", "passthrough", one_hot),
        ]
    )
    inner_cv = StratifiedKFold(
        n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.seed
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
    return Pipeline([("pre", pre), ("model", model)])


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

    ``method`` is passed through to :func:`cross_val_predict` —
    ``"predict"`` for regression, ``"predict_proba"`` for classification
    (the positive-class column is sliced off).
    """
    raw = cross_val_predict(pipe, X, y, cv=cv, method=method)
    y_pred = raw[:, 1] if method == "predict_proba" else raw
    y_true = y.values

    rows: list[dict] = []
    for name, fn in metrics.items():
        value = float(fn(y_true, y_pred))
        ci_lo, ci_hi = bootstrap_ci(
            fn,
            y_true,
            y_pred,
            n_iter=cfg.bootstrap_iter,
            seed=cfg.seed,
            alpha=cfg.bootstrap_alpha,
        )
        rows.append(
            {
                "metric": name,
                "value": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "ci_alpha": cfg.bootstrap_alpha,
                "bootstrap_iter": cfg.bootstrap_iter,
                "n_train": len(y_true),
            }
        )
    return y_pred, rows


def evaluate_feature_set(
    df_train: pd.DataFrame,
    feature_cfg: BaselineFeatureConfig,
    train_cfg: BaselineTrainConfig,
    *,
    set_name: str,
) -> list[dict]:
    """Train + evaluate baselines for one feature configuration.

    Iterates over the three targets, fits the appropriate model under
    5-fold CV on the training split, and returns one row per
    ``[target, metric]`` combination.
    """
    X = build_baseline_features(df_train, feature_cfg)
    numeric, one_hot = _split_columns(X)
    rows: list[dict] = []

    for target in REGRESSION_TARGETS:
        pipe = _build_regression_pipeline(numeric, one_hot, train_cfg)
        cv = KFold(n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed)
        _, target_rows = _evaluate(
            pipe, X, df_train[target], REGRESSION_METRICS, cv, train_cfg, method="predict"
        )
        for r in target_rows:
            r.update({"feature_set": set_name, "target": target, "task": "regression"})
            rows.append(r)
            logger.info(
                "%s | %s | %s = %.4f [CI %.4f, %.4f]",
                set_name, target, r["metric"], r["value"], r["ci_lo"], r["ci_hi"],
            )

    for target in CLASSIFICATION_TARGETS:
        pipe = _build_classification_pipeline(numeric, one_hot, train_cfg)
        cv = StratifiedKFold(
            n_splits=train_cfg.n_cv_folds, shuffle=True, random_state=train_cfg.seed
        )
        _, target_rows = _evaluate(
            pipe,
            X,
            df_train[target].astype(int),
            CLASSIFICATION_METRICS,
            cv,
            train_cfg,
            method="predict_proba",
        )
        for r in target_rows:
            r.update({"feature_set": set_name, "target": target, "task": "classification"})
            rows.append(r)
            logger.info(
                "%s | %s | %s = %.4f [CI %.4f, %.4f]",
                set_name, target, r["metric"], r["value"], r["ci_lo"], r["ci_hi"],
            )

    return rows


DEPLOYABLE_FEATURE_SETS: tuple[str, ...] = ("dialogue_only", "dialogue_only_logged")


def _check_thresholds(rows: list[dict]) -> dict[str, bool]:
    """Apply the brief's escalation thresholds.

    R² < 0.05 (regression) or AUC-ROC < 0.55 (classification) on a
    deployable (no-budget) feature set triggers escalation per Phase 3
    brief Section 3 Task 2 ("Decision point at end of Task 2"). Both
    the original and revised dialogue-only sets are checked so the
    revision is held to the same floor as the original.
    """
    triggers: dict[str, bool] = {}
    for r in rows:
        if r["feature_set"] not in DEPLOYABLE_FEATURE_SETS:
            continue
        key_prefix = r["feature_set"]
        if r["target"] == LOG_ROI_COL and r["metric"] == "r2":
            triggers[f"{key_prefix}_regression_below_floor"] = r["value"] < 0.05
        if r["task"] == "classification" and r["metric"] == "auc_roc":
            triggers[f"{key_prefix}_{r['target']}_below_floor"] = r["value"] < 0.55
    return triggers


def main() -> None:
    """CLI entrypoint: train baselines, save table, log threshold check."""
    train_cfg = BaselineTrainConfig()
    paths.ensure_dirs()

    logger.info("Loading master corpus and split assignments")
    df_full = pd.read_parquet(train_cfg.in_corpus)
    splits = pd.read_parquet(train_cfg.in_splits)

    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)
    logger.info("Training on %d films (train split)", len(df_train))

    rows: list[dict] = []
    logger.info("Evaluating dialogue-only baseline (original)")
    rows.extend(
        evaluate_feature_set(
            df_train,
            BaselineFeatureConfig(include_log_budget=False),
            train_cfg,
            set_name="dialogue_only",
        )
    )
    logger.info("Evaluating with-budget sanity-check baseline (original)")
    rows.extend(
        evaluate_feature_set(
            df_train,
            BaselineFeatureConfig(include_log_budget=True),
            train_cfg,
            set_name="with_budget",
        )
    )
    logger.info("Evaluating dialogue-only baseline (logged + log_runtime)")
    rows.extend(
        evaluate_feature_set(
            df_train,
            BaselineFeatureConfig(
                include_log_budget=False,
                log_transform_structural=True,
                include_log_runtime=True,
            ),
            train_cfg,
            set_name="dialogue_only_logged",
        )
    )
    logger.info("Evaluating with-budget sanity-check baseline (logged + log_runtime)")
    rows.extend(
        evaluate_feature_set(
            df_train,
            BaselineFeatureConfig(
                include_log_budget=True,
                log_transform_structural=True,
                include_log_runtime=True,
            ),
            train_cfg,
            set_name="with_budget_logged",
        )
    )

    out = pd.DataFrame(rows)[
        [
            "feature_set",
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
        logger.warning("Phase 3a escalation thresholds tripped: %s", flagged)
    else:
        logger.info("Phase 3a thresholds OK (R^2 >= 0.05, AUC-ROC >= 0.55)")


if __name__ == "__main__":
    main()
