"""Phase 4 feature importance for the per-target winners.

Reads the winner artifacts persisted by ``finalize.py`` and produces
a per-feature importance ranking using the appropriate method for
each family:

* **Linear families** (Ridge, Logistic-L2, Lasso, L1-Logistic,
  LinearSVR, LinearSVC): standardized coefficient magnitudes.
  Features are already z-scored inside the pipeline (StandardScaler
  on the numeric branch) so the raw ``model.coef_`` values are
  comparable across features.

* **Tree-based families** (HistGB, RandomForest, LightGBM, XGBoost):
  the model's native ``feature_importances_`` attribute. For
  HistGradientBoosting we fall back to permutation importance because
  the native API does not expose per-feature importance directly.

* **Kernel-based families** (SVM-RBF, KNN): permutation importance
  via :func:`sklearn.inspection.permutation_importance`. The model
  has no per-feature weight that translates to importance, so we
  measure each feature's effect on AUC/RMSE by shuffling its values
  and observing the metric drop. More compute-expensive but
  model-agnostic.

The output is one CSV per target plus a summary figure showing the
top-20 features per target side-by-side.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)

ALL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

PERMUTATION_FAMILIES: frozenset[str] = frozenset({
    "svm_rbf", "linear_svm", "histgb",
})


def _load_train_split() -> pd.DataFrame:
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df = add_targets(df)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    return df[df["imdb_id"].isin(train_ids)].reset_index(drop=True)


def _coefficient_importance(
    pipeline, feature_columns: list[str],
) -> pd.DataFrame:
    """Importance from coefficients for linear / linear-SVM models.

    The model's ``coef_`` is in the post-preprocessing feature space,
    which (for the linear families that need scaling) means the
    coefficients are on z-scored features and directly comparable.
    """
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_).ravel()
    return pd.DataFrame({
        "feature": feature_columns,
        "importance": np.abs(coef),
        "signed_coef": coef,
        "method": "abs(coef)",
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _tree_importance(
    pipeline, feature_columns: list[str],
) -> pd.DataFrame:
    """Importance from the model's ``feature_importances_`` attribute."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None  # type: ignore[return-value]
    imp = np.asarray(model.feature_importances_).ravel()
    return pd.DataFrame({
        "feature": feature_columns,
        "importance": imp,
        "signed_coef": np.full(len(imp), np.nan),
        "method": "feature_importances_",
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_repeats: int = 10,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on a (sub)sample of the train set.

    Permutation importance quantifies how much the metric drops when
    a feature's values are randomly shuffled. Model-agnostic; works
    for any estimator including SVM-RBF. We sample to keep compute
    bounded (1199 films x 92 features x 10 repeats can be slow under
    SVM-RBF with kernel evaluations).
    """
    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size < len(X):
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_eval = X.iloc[idx]
        y_eval = y.iloc[idx]
    else:
        X_eval = X
        y_eval = y

    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = permutation_importance(
            pipeline,
            X_eval, y_eval,
            n_repeats=n_repeats,
            random_state=seed,
            scoring=scoring,
            n_jobs=1,  # safer with sklearn pipelines
        )
    return pd.DataFrame({
        "feature": list(X.columns),
        "importance": result.importances_mean,
        "signed_coef": np.full(len(X.columns), np.nan),
        "importance_std": result.importances_std,
        "method": f"permutation (n_repeats={n_repeats})",
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def importance_for_winner(
    target: str,
    df_train: pd.DataFrame,
    permutation_sample_size: int = 600,
) -> pd.DataFrame:
    """Compute the appropriate importance ranking for one target's winner."""
    artifact = paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{target}.joblib"
    if not artifact.is_file():
        raise RuntimeError(f"Missing winner artifact: {artifact}")
    bundle = joblib.load(artifact)
    pipeline = bundle["estimator"]
    family = bundle["family"]
    matrix_name = bundle["matrix"]
    feature_columns = bundle["feature_columns"]
    task = bundle["task"]

    matrix_spec = MATRICES[matrix_name]
    X = build_matrix(matrix_spec, df_train)
    if task == "classification":
        y = df_train[target].astype(int)
    else:
        y = df_train[target]
    # Defensive column alignment; should already match.
    X = X[feature_columns]

    logger.info(
        "Importance for %s: family=%s matrix=%s n_features=%d",
        target, family, matrix_name, len(feature_columns),
    )

    if family in {"linear", "lasso", "linear_svm"}:
        df = _coefficient_importance(pipeline, feature_columns)
    elif family in {"random_forest", "lightgbm", "xgboost"}:
        df = _tree_importance(pipeline, feature_columns)
        if df is None:
            df = _permutation_importance(
                pipeline, X, y, task,
                sample_size=permutation_sample_size,
            )
    else:
        # SVM-RBF, HistGB (no native importance), KNN, etc.
        df = _permutation_importance(
            pipeline, X, y, task,
            sample_size=permutation_sample_size,
        )

    df["target"] = target
    df["family"] = family
    df["matrix"] = matrix_name
    return df


def plot_top_features(
    importances: dict[str, pd.DataFrame],
    out_path: Path,
    top_k: int = 20,
) -> Path:
    """Render side-by-side bar charts of top-k features per target."""
    targets = list(importances.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 8))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        df = importances[target].head(top_k).iloc[::-1]
        bundle_family = df["family"].iloc[0] if not df.empty else "?"
        method = df["method"].iloc[0] if not df.empty else "?"
        ax.barh(df["feature"], df["importance"], color="steelblue")
        ax.set_title(
            f"{target}\n({bundle_family}, top {top_k} by {method})",
            fontsize=10,
        )
        ax.set_xlabel("importance")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Phase 4 winner feature importance (post-Tier-A models)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--permutation-sample", type=int, default=600,
        help="Subsample size for permutation importance (default 600).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    set_log_level(args.log_level)
    paths.ensure_dirs()
    df_train = _load_train_split()

    importances: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for target in ALL_TARGETS:
        try:
            df = importance_for_winner(
                target, df_train,
                permutation_sample_size=args.permutation_sample,
            )
        except RuntimeError as exc:
            logger.warning("Skip %s: %s", target, exc)
            continue
        importances[target] = df
        rows.append({"target": target, "n_features": len(df)})

        out_csv = paths.REPORTS_TABLES_DIR / f"phase4_importance_{target}.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Wrote %s (%d rows)", out_csv, len(df))
        print()
        print(f"=== Top {args.top_k} for {target} ({df['family'].iloc[0]}, {df['matrix'].iloc[0]}) ===")
        print(df.head(args.top_k).to_string(index=False))

    plot_top_features(
        importances,
        paths.REPORTS_FIGURES_DIR / "phase4_feature_importance.png",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
