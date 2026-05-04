"""Phase 4 sensitivity-analysis runner: unweighted vs class-balanced.

Pre-registered class-balancing policy is ``class_weight="balanced"``
for SVM and Logistic, ``sample_weight`` for HistGB. The Phase 3c
benchmark used unweighted training and reported SVM-RBF reaching
``roi_gt_2`` AUC 0.665 OOF on ``all_five``. If the Phase 4 benchmark
under the weighted policy lands materially below that band, the gap
could be the policy choice rather than the corpus ceiling.

This sensitivity runner is the diagnostic. It re-runs the primary
tier on a single (matrix, target) under the *unweighted* training
policy, holding everything else equal (same outer CV scheme, same
hyperparameter grid, same metric vocabulary). The output goes to a
separate CSV (``phase4_sensitivity_unweighted.csv``) and a separate
``runs/phase_4/<timestamp>_unweighted_*/`` set so the headline
benchmark is unaffected.

The runner is a documented diagnostic, not a methodology pivot. The
pre-registration discipline still requires the headline benchmark to
report under the weighted policy. The sensitivity result lets the
planning conversation see the gap explicitly when deciding whether
the headline number is corpus-ceiling or policy-induced.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import warnings
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

# Silence sklearn 1.8 deprecation noise.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.experiments.save_run import save_run
from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4 import families as families_module
from src.models.phase4.benchmark import cell_to_rows, task_for_target
from src.models.phase4.cv import evaluate_family_target
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Unweighted family overrides
# ---------------------------------------------------------------------------


def unweighted_primary_specs() -> list[families_module.FamilySpec]:
    """Return primary-tier specs with ``class_weight=None``.

    Patches each FamilySpec's classifier_factory to omit
    ``class_weight``. Regressor factories are unchanged (no class
    weight semantics for regression). Hyperparameter grids unchanged.
    """
    seed = families_module.SEED
    specs = [
        families_module.FamilySpec(
            name="linear_unweighted",
            tier="primary",
            needs_scaling=True,
            balancing="none",
            regressor_factory=lambda: Ridge(random_state=seed),
            classifier_factory=lambda: LogisticRegression(
                solver="lbfgs", max_iter=2000, random_state=seed,
            ),
            regression_grid=families_module.FAMILIES["linear"].regression_grid,
            classification_grid=families_module.FAMILIES["linear"].classification_grid,
            score_method="predict_proba",
        ),
        families_module.FamilySpec(
            name="histgb_unweighted",
            tier="primary",
            needs_scaling=False,
            balancing="none",
            regressor_factory=lambda: HistGradientBoostingRegressor(
                max_iter=200, early_stopping=True, random_state=seed,
            ),
            classifier_factory=lambda: HistGradientBoostingClassifier(
                max_iter=200, early_stopping=True, random_state=seed,
            ),
            regression_grid=families_module.FAMILIES["histgb"].regression_grid,
            classification_grid=families_module.FAMILIES["histgb"].classification_grid,
            score_method="predict_proba",
        ),
        families_module.FamilySpec(
            name="random_forest_unweighted",
            tier="primary",
            needs_scaling=False,
            balancing="none",
            regressor_factory=lambda: RandomForestRegressor(
                random_state=seed, n_jobs=1,
            ),
            classifier_factory=lambda: RandomForestClassifier(
                random_state=seed, n_jobs=1,
            ),
            regression_grid=families_module.FAMILIES["random_forest"].regression_grid,
            classification_grid=families_module.FAMILIES["random_forest"].classification_grid,
            score_method="predict_proba",
        ),
        families_module.FamilySpec(
            name="svm_rbf_unweighted",
            tier="primary",
            needs_scaling=True,
            balancing="none",
            regressor_factory=lambda: SVR(kernel="rbf"),
            classifier_factory=lambda: SVC(
                kernel="rbf", random_state=seed,
            ),
            regression_grid=families_module.FAMILIES["svm_rbf"].regression_grid,
            classification_grid=families_module.FAMILIES["svm_rbf"].classification_grid,
            score_method="decision_function",
        ),
    ]
    return specs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_sensitivity(
    matrix_name: str = "all_five",
    targets: tuple[str, ...] = ("log_roi", "roi_gt_1", "roi_gt_2"),
    n_jobs: int = -1,
    out_table: Path | None = None,
) -> pd.DataFrame:
    """Run unweighted sensitivity comparison on one matrix, saving rows."""
    out_table = out_table or paths.REPORTS_TABLES_DIR / "phase4_sensitivity_unweighted.csv"
    paths.ensure_dirs()

    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)

    matrix_spec = MATRICES[matrix_name]
    X = build_matrix(matrix_spec, df_train)
    rows: list[dict] = []

    for spec in unweighted_primary_specs():
        logger.info("Sensitivity: family=%s matrix=%s X=%s", spec.name, matrix_name, X.shape)
        with save_run(
            phase="phase_4",
            name=f"sensitivity_{matrix_name}_{spec.name}",
            params={
                "family": spec.name,
                "matrix": matrix_name,
                "policy": "unweighted (class_weight=None)",
                "mode": "sensitivity",
            },
            preprocessing={"matrix_description": matrix_spec.description},
            features=list(X.columns),
        ) as run:
            per_target_payload: dict = {}
            cv_results = {}
            for target in targets:
                task = task_for_target(target)
                y = df_train[target].astype(int) if task == "classification" else df_train[target]
                cv_result = evaluate_family_target(spec, task, X, y, n_jobs=n_jobs)
                cv_results[target] = cv_result
                logger.info(
                    "Sensitivity Done: family=%s target=%s OOF=%s",
                    spec.name, target,
                    {k: round(v, 4) for k, v in cv_result.oof_metrics_global.items()
                     if isinstance(v, float) and not np.isnan(v)},
                )
                per_target_payload[target] = {
                    "task": task,
                    "best_params": {
                        k.removeprefix("model__"): (v.item() if hasattr(v, "item") else v)
                        for k, v in cv_result.best_params.items()
                    },
                    "in_sample_metrics": cv_result.in_sample_metrics,
                    "oof_metrics_global": cv_result.oof_metrics_global,
                    "per_fold_metrics": {
                        k: v.tolist() for k, v in cv_result.per_fold_metrics.items()
                    },
                    "oof_score": cv_result.oof_score.tolist(),
                    "oof_hard": (
                        cv_result.oof_hard.tolist() if cv_result.oof_hard is not None else None
                    ),
                    "y_true": y.values.tolist(),
                }
            run.record_metrics({"per_target": per_target_payload})

        # Flatten to CSV rows.
        from src.models.phase4.benchmark import CellResult
        cell = CellResult(matrix_name=matrix_name, family_name=spec.name)
        cell.per_target = cv_results
        rows.extend(cell_to_rows(cell, spec))

    df = pd.DataFrame(rows)
    df.to_csv(out_table, index=False)
    logger.info("Wrote sensitivity table %s (%d rows)", out_table, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="all_five",
                        choices=sorted(MATRICES))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--targets", nargs="+",
        default=["log_roi", "roi_gt_1", "roi_gt_2"],
        help="Subset of targets to evaluate.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_log_level(args.log_level)
    run_sensitivity(
        matrix_name=args.matrix,
        targets=tuple(args.targets),
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
