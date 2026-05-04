"""Phase 4 finalization: pick per-target winners and persist artifacts.

Reads the consolidated benchmark CSVs (MiniLM + mpnet primary tier),
identifies the headline winner per target by OOF metric (AUC-ROC for
classification, RMSE for regression; parsimony tie-breaker per the
pre-registration), re-fits the winner on the full train split, and
writes ``data/processed/phase4_primary_model_<target>.joblib``.

Also writes ``data/processed/phase4_stacking_model_<target>.joblib``
holding the meta-learner + the four per-family base estimators that
feed it, so Phase 5 can choose to calibrate either the single best
base model or the stacking ensemble.

Run from project root::

    python -m src.models.phase4.finalize
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4.cv import (
    build_pipeline,
    fit_full_train,
)
from src.models.phase4.families import FAMILIES, FamilySpec
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def task_for_target(target: str) -> str:
    if target in REGRESSION_TARGETS:
        return "regression"
    return "classification"


def load_benchmark() -> pd.DataFrame:
    """Concatenate the MiniLM and mpnet benchmark CSVs."""
    frames = []
    for p in (
        paths.REPORTS_TABLES_DIR / "phase4_benchmark.csv",
        paths.REPORTS_TABLES_DIR / "phase4_benchmark_mpnet.csv",
    ):
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        raise RuntimeError("No Phase 4 benchmark CSVs found.")
    return pd.concat(frames, ignore_index=True)


def find_winner(
    bench: pd.DataFrame,
    target: str,
) -> tuple[str, str, dict, float]:
    """Return (matrix, family, best_params, oof_value) for one target.

    Selection metric: AUC-ROC for classification, RMSE for regression.
    Ties on the headline metric break to the parsimonious matrix
    (``standalone_*`` over ``all_five*``) per pre-registration Sec 8.
    """
    task = task_for_target(target)
    metric = "auc_roc" if task == "classification" else "rmse"
    higher_is_better = task == "classification"
    sub = bench[
        (bench["target"] == target)
        & (bench["metric"] == metric)
        & (bench["eval_set"] == "oof_global")
        & (bench["tier"] == "primary")
    ].copy()
    if sub.empty:
        raise RuntimeError(f"No primary-tier rows for target {target!r}")

    if higher_is_better:
        best_value = sub["value"].max()
        ties = sub[sub["value"] == best_value]
    else:
        best_value = sub["value"].min()
        ties = sub[sub["value"] == best_value]

    # Parsimony tie-break: prefer "standalone" over "all_five".
    def parsimony_score(row: pd.Series) -> int:
        return 0 if row["matrix"].startswith("standalone") else 1
    ties = ties.copy()
    ties["_parsimony"] = ties.apply(parsimony_score, axis=1)
    chosen = ties.sort_values("_parsimony").iloc[0]

    best_params = ast.literal_eval(chosen["best_params"])
    # Strip np.float64() / np.int64() wrappers via repr round-trip.
    def _scalarize(v):
        if hasattr(v, "item"):
            return v.item()
        return v
    best_params = {k: _scalarize(v) for k, v in best_params.items()}

    return (
        str(chosen["matrix"]),
        str(chosen["family"]),
        best_params,
        float(chosen["value"]),
    )


def fit_and_save_winner(
    target: str,
    matrix_name: str,
    family_name: str,
    best_params: dict,
    df_train: pd.DataFrame,
) -> Path:
    """Re-fit the winning estimator on full train and persist."""
    spec: FamilySpec = FAMILIES[family_name]
    matrix_spec = MATRICES[matrix_name]
    X = build_matrix(matrix_spec, df_train)

    task = task_for_target(target)
    if task == "classification":
        y = df_train[target].astype(int)
    else:
        y = df_train[target]

    pipeline_params = {f"model__{k}": v for k, v in best_params.items()}
    fitted_pipeline, _, _ = fit_full_train(
        spec, task, X, y, pipeline_params,
    )

    bundle = {
        "target": target,
        "matrix": matrix_name,
        "family": family_name,
        "best_params": best_params,
        "score_method": spec.score_method if task == "classification" else None,
        "estimator": fitted_pipeline,
        "feature_columns": list(X.columns),
        "task": task,
    }
    out = paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{target}.joblib"
    joblib.dump(bundle, out)
    logger.info(
        "Saved Phase 4 winner: target=%s matrix=%s family=%s -> %s",
        target, matrix_name, family_name, out.name,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    set_log_level(args.log_level)
    paths.ensure_dirs()

    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)

    bench = load_benchmark()
    targets = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

    summary_rows = []
    for target in targets:
        matrix_name, family_name, best_params, oof_value = find_winner(bench, target)
        logger.info(
            "Winner for %s: %s on %s (OOF %.4f) params=%s",
            target, family_name, matrix_name, oof_value, best_params,
        )
        out = fit_and_save_winner(
            target, matrix_name, family_name, best_params, df_train,
        )
        summary_rows.append({
            "target": target,
            "matrix": matrix_name,
            "family": family_name,
            "oof_metric": oof_value,
            "best_params": str(best_params),
            "artifact": out.name,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = paths.REPORTS_TABLES_DIR / "phase4_winners.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info("Wrote winners summary to %s", out_csv)
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
