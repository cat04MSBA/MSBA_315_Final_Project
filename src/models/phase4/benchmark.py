"""Phase 4 benchmark orchestrator.

Iterates over (matrix, family, target) combinations, calling the inner
``GridSearchCV`` and the outer repeated-CV evaluator from
:mod:`src.models.phase4.cv`. For each combination it writes a
``save_run`` directory, a row to the consolidated benchmark CSV, and
(once all primary-tier rows are in) the paired Bayesian comparisons
CSV. After all primary-tier work is in, the per-target winner is
identified by OOF AUC-ROC for classification and OOF RMSE for
regression, and the winning fitted estimator is persisted to
``data/processed/phase4_primary_model_<target>.joblib``.

Two CLI modes (selected via :class:`BenchmarkConfig.mode`):

* ``"smoke"``: tiny grid + 1 family + 1 target + 1 matrix. Verifies
  the harness end-to-end before committing to the full sweep. Saves
  to ``runs/phase_4/smoke_*/`` and to a separate ``smoke`` CSV that
  is not the deliverable.
* ``"full"``: the pre-registered benchmark per Section 4 of
  ``phase4_preregistration.md``. Both matrices for the primary tier;
  ``all_five`` only for the secondary tier.

The orchestrator is deliberately a script-friendly module: import
``run_full_benchmark`` for use in a notebook, or invoke the entry-point
:mod:`src.experiments.run_phase4_benchmark` from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4.cv import (
    CVResult,
    N_OUTER_FOLDS,
    N_OUTER_REPEATS,
    evaluate_family_target,
)
from src.models.phase4.families import (
    FAMILIES,
    FamilySpec,
    primary_tier,
    secondary_tier,
)
from src.models.phase4.matrices import MATRICES, MatrixSpec, build_matrix
from src.models.phase4.paired_test import (
    PairedComparison,
    all_pairwise_comparisons,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Targets are stable strings the rest of the project uses.
ALL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

# Headline target for the primary-outcome decision (informally already
# committed per Phase 3c evidence; formal commit at end of phase).
HEADLINE_TARGET: str = "roi_gt_2"

# Metrics over which the paired tests are run.
CLASSIFICATION_TEST_METRICS: list[str] = ["auc_roc", "pr_auc", "f1", "log_loss"]
REGRESSION_TEST_METRICS: list[str] = ["rmse", "mse", "mae", "cvrmse"]


@dataclass
class BenchmarkConfig:
    """Top-level configuration for one benchmark invocation."""

    mode: str = "full"  # "smoke" or "full"
    families: tuple[str, ...] = tuple(f.name for f in primary_tier())
    secondary_families: tuple[str, ...] = tuple(f.name for f in secondary_tier())
    matrices: tuple[str, ...] = ("all_five", "standalone_positive_union")
    secondary_matrices: tuple[str, ...] = ("all_five",)
    targets: tuple[str, ...] = ALL_TARGETS
    n_jobs: int = -1
    save_models: bool = True
    out_table: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR / "phase4_benchmark.csv"
    )
    out_paired_table: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR / "phase4_paired_tests.csv"
    )
    model_artifact_pattern: str = "phase4_primary_model_{target}.joblib"


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------


def load_train_split() -> pd.DataFrame:
    """Load the master corpus, attach targets, filter to the train split.

    Returns a DataFrame with the train rows only. The calibration set
    and the held-out test set are not loaded into the benchmark; the
    Phase 4 brief and ``CLAUDE_CODE_GUIDELINES.md`` Section 4 forbid
    touching either before the relevant downstream phase.
    """
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)
    logger.info(
        "Loaded train split: %d films (calibration and test held out)",
        len(df_train),
    )
    return df_train


def task_for_target(target: str) -> str:
    """Return ``"regression"`` or ``"classification"`` for a target name."""
    if target in REGRESSION_TARGETS:
        return "regression"
    if target in CLASSIFICATION_TARGETS:
        return "classification"
    raise KeyError(f"Unknown target {target!r}")


# ---------------------------------------------------------------------------
# Per-cell evaluation (one matrix x one family x all targets)
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """All evaluation results for one (matrix, family) cell across targets."""
    matrix_name: str
    family_name: str
    per_target: dict[str, CVResult] = field(default_factory=dict)


def evaluate_cell(
    matrix_spec: MatrixSpec,
    family_spec: FamilySpec,
    df_train: pd.DataFrame,
    targets: tuple[str, ...],
    cfg: BenchmarkConfig,
) -> CellResult:
    """Evaluate one family on one input matrix across the requested targets.

    Wraps the work in a ``save_run`` block so the per-target metrics,
    chosen hyperparameters, and per-fold scores are persisted under
    ``runs/phase_4/<timestamp>_<matrix>_<family>/``.
    """
    X = build_matrix(matrix_spec, df_train)
    n_features = X.shape[1]
    logger.info(
        "Cell start: matrix=%s family=%s X=%s",
        matrix_spec.name, family_spec.name, X.shape,
    )

    cell = CellResult(matrix_name=matrix_spec.name, family_name=family_spec.name)
    run_name = f"{matrix_spec.name}_{family_spec.name}"

    imdb_ids = list(df_train["imdb_id"].astype(str).values)
    with save_run(
        phase="phase_4",
        name=run_name,
        params={
            "family": family_spec.name,
            "matrix": matrix_spec.name,
            "regression_grid": _serializable_grid(family_spec.regression_grid),
            "classification_grid": _serializable_grid(family_spec.classification_grid),
            "balancing": family_spec.balancing,
            "score_method": family_spec.score_method,
            "n_outer_folds": N_OUTER_FOLDS,
            "n_outer_repeats": N_OUTER_REPEATS,
            "n_features": int(n_features),
            "n_train": int(len(df_train)),
            "mode": cfg.mode,
        },
        preprocessing={
            "matrix_description": matrix_spec.description,
            "needs_scaling": family_spec.needs_scaling,
        },
        features=list(X.columns),
    ) as run:
        per_target_for_metrics: dict[str, dict[str, Any]] = {}
        for target in targets:
            task = task_for_target(target)
            y = df_train[target].astype(int) if task == "classification" else df_train[target]
            logger.info(
                "Eval: matrix=%s family=%s target=%s task=%s",
                matrix_spec.name, family_spec.name, target, task,
            )
            cv_result = evaluate_family_target(
                family_spec, task, X, y, n_jobs=cfg.n_jobs,
            )
            cell.per_target[target] = cv_result
            per_target_for_metrics[target] = {
                "task": task,
                "best_params": _stringify_params(cv_result.best_params),
                "inner_score": cv_result.inner_score,
                "in_sample_metrics": cv_result.in_sample_metrics,
                "oof_metrics_global": cv_result.oof_metrics_global,
                "oof_metrics_global_ci": {
                    k: list(v) for k, v in cv_result.oof_metrics_global_ci.items()
                },
                "per_fold_metrics": {
                    k: v.tolist() for k, v in cv_result.per_fold_metrics.items()
                },
                # OOF predictions per row, averaged across the 3 outer
                # repetitions. Used by the calibration plotter
                # (Phase 4 deliverable phase4_calibration_pre.png) and by
                # any post-hoc rerank of the OOF metric set.
                "oof_score": cv_result.oof_score.tolist(),
                "oof_hard": (
                    cv_result.oof_hard.tolist()
                    if cv_result.oof_hard is not None else None
                ),
                "y_true": y.values.tolist(),
                "in_sample_score": cv_result.in_sample_score.tolist(),
                "in_sample_hard": (
                    cv_result.in_sample_hard.tolist()
                    if cv_result.in_sample_hard is not None else None
                ),
            }
            logger.info(
                "Done: matrix=%s family=%s target=%s OOF=%s inner_best_params=%s",
                matrix_spec.name, family_spec.name, target,
                _format_global(cv_result.oof_metrics_global),
                cv_result.best_params,
            )
        per_target_for_metrics["_imdb_ids"] = imdb_ids

        run.record_metrics({"per_target": per_target_for_metrics})

        # Persist a {target: fitted_estimator} dict; lets a single run
        # carry all three targets so we don't need 3x the run directories.
        if cfg.save_models and family_spec.tier == "primary":
            estimators = {
                tgt: cell.per_target[tgt].fitted_estimator for tgt in cell.per_target
            }
            run.save_model(estimators)
            run.append_to_runs_md(
                model_family=family_spec.name,
                features_group=matrix_spec.name,
                key_metric=_format_run_summary(cell),
                notes="Phase 4 primary-tier benchmark cell",
            )
        elif family_spec.tier == "secondary":
            run.append_to_runs_md(
                model_family=family_spec.name,
                features_group=matrix_spec.name,
                key_metric=_format_run_summary(cell),
                notes="Phase 4 secondary-tier benchmark cell",
            )

    return cell


def _serializable_grid(grid: dict[str, list]) -> dict[str, list]:
    """Make a grid JSON-friendly (numpy floats -> Python floats)."""
    out: dict[str, list] = {}
    for k, vs in grid.items():
        out[k] = [v.item() if hasattr(v, "item") else v for v in vs]
    return out


def _stringify_params(params: dict[str, object]) -> dict[str, object]:
    """Convert ``model__<key>`` pipeline-prefixed keys to plain ``<key>``."""
    out: dict[str, object] = {}
    for k, v in params.items():
        bare = k.removeprefix("model__")
        out[bare] = v.item() if hasattr(v, "item") else v
    return out


def _format_global(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(metrics.items())
        if isinstance(v, float) and not np.isnan(v)
    )


def _format_run_summary(cell: CellResult) -> str:
    """One-line metric summary for the runs/RUNS.md row."""
    parts = []
    for tgt in cell.per_target:
        m = cell.per_target[tgt].oof_metrics_global
        if "auc_roc" in m and not np.isnan(m["auc_roc"]):
            parts.append(f"{tgt} AUC={m['auc_roc']:.3f}")
        elif "rmse" in m and not np.isnan(m["rmse"]):
            parts.append(f"{tgt} RMSE={m['rmse']:.3f}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------


def cell_to_rows(cell: CellResult, family_spec: FamilySpec) -> list[dict]:
    """Flatten one CellResult into rows for ``phase4_benchmark.csv``.

    Schema: ``matrix, family, tier, target, task, eval_set, metric,
    value, ci_lo, ci_hi, n_folds, best_params``. One row per
    (target, eval_set, metric).
    """
    rows: list[dict] = []
    for target, cv_result in cell.per_target.items():
        task = task_for_target(target)
        # In-sample metrics.
        for metric, value in cv_result.in_sample_metrics.items():
            rows.append({
                "matrix": cell.matrix_name,
                "family": cell.family_name,
                "tier": family_spec.tier,
                "target": target,
                "task": task,
                "eval_set": "train",
                "metric": metric,
                "value": value,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_folds": 0,
                "best_params": str(_stringify_params(cv_result.best_params)),
            })
        # Global OOF metrics with bootstrap CIs.
        for metric, value in cv_result.oof_metrics_global.items():
            ci_lo, ci_hi = cv_result.oof_metrics_global_ci.get(
                metric, (float("nan"), float("nan")),
            )
            rows.append({
                "matrix": cell.matrix_name,
                "family": cell.family_name,
                "tier": family_spec.tier,
                "target": target,
                "task": task,
                "eval_set": "oof_global",
                "metric": metric,
                "value": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_folds": int(N_OUTER_FOLDS * N_OUTER_REPEATS),
                "best_params": str(_stringify_params(cv_result.best_params)),
            })
        # Per-fold metric mean and std (the input to the paired Bayesian test).
        for metric, arr in cv_result.per_fold_metrics.items():
            valid = arr[np.isfinite(arr)]
            mean = float(np.mean(valid)) if len(valid) else float("nan")
            std = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
            rows.append({
                "matrix": cell.matrix_name,
                "family": cell.family_name,
                "tier": family_spec.tier,
                "target": target,
                "task": task,
                "eval_set": "oof_perfold_mean",
                "metric": metric,
                "value": mean,
                "ci_lo": mean - std,
                "ci_hi": mean + std,
                "n_folds": int(len(valid)),
                "best_params": str(_stringify_params(cv_result.best_params)),
            })
    return rows


def comparisons_to_rows(
    matrix: str,
    target: str,
    cmps: list[PairedComparison],
) -> list[dict]:
    """Flatten paired comparisons into rows for ``phase4_paired_tests.csv``."""
    rows = []
    for c in cmps:
        rows.append({
            "matrix": matrix,
            "target": target,
            "family_a": c.family_a,
            "family_b": c.family_b,
            "metric": c.metric,
            "p_a_better": c.p_a_better,
            "p_rope": c.p_rope,
            "p_b_better": c.p_b_better,
            "rope_halfwidth": c.rope,
            "runs": c.runs,
            "n_folds": c.n_folds,
            "winner": c.winner,
        })
    return rows


# ---------------------------------------------------------------------------
# Winner identification and artifact saving
# ---------------------------------------------------------------------------


def select_winner(
    cells: list[CellResult],
    target: str,
) -> tuple[str, str, CVResult]:
    """Pick the (matrix, family) winner for a target on the global OOF metric.

    Selection metric: ``auc_roc`` for classification, ``rmse`` (lower
    is better) for regression. Ties on the headline metric break to the
    standalone_positive_union matrix per the parsimony rule
    (Section 8 of phase4_preregistration.md).
    """
    task = task_for_target(target)
    best_score = -np.inf if task == "classification" else np.inf
    best: tuple[str, str, CVResult] | None = None

    for cell in cells:
        if target not in cell.per_target:
            continue
        cv_result = cell.per_target[target]
        if task == "classification":
            score = cv_result.oof_metrics_global.get("auc_roc", float("nan"))
            if np.isnan(score):
                continue
            if score > best_score or (
                score == best_score and best is not None
                and cell.matrix_name == "standalone_positive_union"
            ):
                best_score = score
                best = (cell.matrix_name, cell.family_name, cv_result)
        else:
            score = cv_result.oof_metrics_global.get("rmse", float("nan"))
            if np.isnan(score):
                continue
            if score < best_score or (
                score == best_score and best is not None
                and cell.matrix_name == "standalone_positive_union"
            ):
                best_score = score
                best = (cell.matrix_name, cell.family_name, cv_result)

    if best is None:
        raise RuntimeError(f"No valid cell for target {target!r}")
    return best


def save_winner_artifact(
    target: str,
    matrix_name: str,
    family_name: str,
    cv_result: CVResult,
    pattern: str,
) -> Path:
    """Persist the winner's fitted estimator to ``data/processed/``."""
    out = paths.DATA_PROCESSED_DIR / pattern.format(target=target)
    bundle = {
        "target": target,
        "matrix": matrix_name,
        "family": family_name,
        "best_params": cv_result.best_params,
        "oof_metrics_global": cv_result.oof_metrics_global,
        "estimator": cv_result.fitted_estimator,
    }
    joblib.dump(bundle, out)
    logger.info("Saved Phase 4 winner for %s: %s", target, out)
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_full_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Run the full Phase 4 benchmark per the pre-registered config.

    Returns a dict with the aggregated results, also written to disk:

    * ``cells``: list of ``CellResult``
    * ``winners``: dict ``target -> (matrix, family, oof_score)``
    * ``benchmark_table``: ``pd.DataFrame`` corresponding to
      ``phase4_benchmark.csv``
    * ``paired_table``: ``pd.DataFrame`` corresponding to
      ``phase4_paired_tests.csv``
    """
    paths.ensure_dirs()
    df_train = load_train_split()

    cells: list[CellResult] = []
    benchmark_rows: list[dict] = []

    # Primary tier: every (matrix in cfg.matrices, family in cfg.families).
    for matrix_name in cfg.matrices:
        matrix_spec = MATRICES[matrix_name]
        for family_name in cfg.families:
            family_spec = FAMILIES[family_name]
            cell = evaluate_cell(matrix_spec, family_spec, df_train, cfg.targets, cfg)
            cells.append(cell)
            benchmark_rows.extend(cell_to_rows(cell, family_spec))

    # Secondary tier: only on cfg.secondary_matrices.
    for matrix_name in cfg.secondary_matrices:
        matrix_spec = MATRICES[matrix_name]
        for family_name in cfg.secondary_families:
            family_spec = FAMILIES[family_name]
            cell = evaluate_cell(matrix_spec, family_spec, df_train, cfg.targets, cfg)
            cells.append(cell)
            benchmark_rows.extend(cell_to_rows(cell, family_spec))

    # Paired Bayesian comparisons, primary tier only.
    paired_rows: list[dict] = []
    for matrix_name in cfg.matrices:
        for target in cfg.targets:
            task = task_for_target(target)
            metrics = (
                CLASSIFICATION_TEST_METRICS if task == "classification"
                else REGRESSION_TEST_METRICS
            )
            per_family: dict[str, dict[str, np.ndarray]] = {}
            for cell in cells:
                if cell.matrix_name != matrix_name:
                    continue
                family_spec = FAMILIES[cell.family_name]
                if family_spec.tier != "primary":
                    continue
                if target not in cell.per_target:
                    continue
                per_family[cell.family_name] = cell.per_target[target].per_fold_metrics
            if len(per_family) >= 2:
                cmps = all_pairwise_comparisons(per_family, metrics)
                paired_rows.extend(comparisons_to_rows(matrix_name, target, cmps))

    benchmark_table = pd.DataFrame(benchmark_rows)
    paired_table = pd.DataFrame(paired_rows)
    benchmark_table.to_csv(cfg.out_table, index=False)
    paired_table.to_csv(cfg.out_paired_table, index=False)
    logger.info(
        "Wrote benchmark table (%d rows) and paired table (%d rows)",
        len(benchmark_table), len(paired_table),
    )

    # Winners + artifacts.
    winners: dict[str, dict[str, Any]] = {}
    if cfg.save_models:
        for target in cfg.targets:
            try:
                matrix_name, family_name, cv_result = select_winner(cells, target)
            except RuntimeError as exc:
                logger.warning("Skipping winner for %s: %s", target, exc)
                continue
            artifact = save_winner_artifact(
                target, matrix_name, family_name, cv_result,
                cfg.model_artifact_pattern,
            )
            winners[target] = {
                "matrix": matrix_name,
                "family": family_name,
                "artifact": str(artifact),
                "oof_metrics_global": cv_result.oof_metrics_global,
            }

    return {
        "cells": cells,
        "winners": winners,
        "benchmark_table": benchmark_table,
        "paired_table": paired_table,
    }
