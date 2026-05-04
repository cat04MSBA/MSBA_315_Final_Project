"""Phase 5 orchestrator: end-to-end calibration of the per-target winners.

For each target:

1. Load the Phase 4 winner artifact from
   ``data/processed/phase4_primary_model_<target>.joblib``.
2. Build the matching feature matrix on the calibration set
   (``standalone_positive_union_mpnet`` for all three current
   winners; the matrix spec is read from the bundle).
3. For classification targets: evaluate Platt + isotonic +
   uncalibrated; pick the winner by mean ECE. Run conformal
   classification under both ``"lac"`` and ``"aps"`` conformity
   scores; pick the conformity score with empirical coverage
   closest to nominal at 0.90.
4. For the regression target: run conformal regression under
   both ``"absolute"`` and ``"gamma"`` conformity scores; pick
   the score with empirical coverage closest to nominal at 0.90.
5. Persist the calibrated wrapper bundle to
   ``data/processed/phase5_calibrated_model_<target>.joblib``.
6. Aggregate results into the deliverable CSVs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.calibration.conformal import (
    CONFIDENCE_LEVELS,
    ConformalClassificationResult,
    ConformalRegressionResult,
    evaluate_conformal_classifier,
    evaluate_conformal_regressor,
)
from src.calibration.cv import load_cal_assignments
from src.calibration.metrics import expected_calibration_error
from src.calibration.probability import (
    ProbabilityCalibrationResult,
    evaluate_calibration_method,
    select_best_method,
)
from src.experiments.save_run import save_run
from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

ALL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

CLASSIFICATION_METHODS: tuple[str, ...] = ("uncalibrated", "sigmoid", "isotonic")
# MAPIE 1.4 enforces ``lac`` as the only valid score for binary
# classification; ``aps``/``raps``/``top_k`` are multi-class only.
# Pre-registration Section 9 trigger #4 (conformity-score
# sensitivity) is therefore not testable on binary; documented in
# the Phase 5 summary.
CLASSIFICATION_CONFORMITY_SCORES: tuple[str, ...] = ("lac",)
# ``gamma`` requires strictly positive targets; ``log_roi`` is
# signed (negative for flops). Only ``absolute`` applies to the
# regression target. Documented as a Section 11 deviation in the
# Phase 5 summary.
REGRESSION_CONFORMITY_SCORES: tuple[str, ...] = ("absolute",)


def _task_for(target: str) -> str:
    if target in REGRESSION_TARGETS:
        return "regression"
    return "classification"


def _load_cal_features(matrix_name: str, target: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (X_cal, y_cal, imdb_ids) for the calibration split."""
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    cal_ids = splits.loc[splits["split"] == "cal", "imdb_id"].tolist()
    df_cal = df_full[df_full["imdb_id"].isin(cal_ids)].reset_index(drop=True)
    matrix_spec = MATRICES[matrix_name]
    X_cal = build_matrix(matrix_spec, df_cal)
    if _task_for(target) == "classification":
        y_cal = df_cal[target].astype(int).values
    else:
        y_cal = df_cal[target].values
    imdb_ids = df_cal["imdb_id"].astype(str).tolist()
    return X_cal, y_cal, imdb_ids


def _select_best_conformal_score_classification(
    results: list[ConformalClassificationResult],
    target_level: float = 0.90,
) -> ConformalClassificationResult:
    """Pick the conformity score whose empirical coverage at the target level
    is closest to nominal."""
    def gap(r: ConformalClassificationResult) -> float:
        return abs(r.aggregate_metrics[f"empirical_coverage_at_{target_level}_mean"] - target_level)
    return sorted(results, key=gap)[0]


def _select_best_conformal_score_regression(
    results: list[ConformalRegressionResult],
    target_level: float = 0.90,
) -> ConformalRegressionResult:
    def gap(r: ConformalRegressionResult) -> float:
        return abs(r.aggregate_metrics[f"empirical_coverage_at_{target_level}_mean"] - target_level)
    return sorted(results, key=gap)[0]


def calibrate_target(target: str, run_dir_name: str | None = None) -> dict[str, Any]:
    """Run the full Phase 5 calibration for one target.

    Returns a dict with keys ``probability_results``,
    ``conformal_results``, ``best_probability``, ``best_conformal``,
    ``deployed_calibrated_estimator``, ``deployed_conformal_wrapper``,
    ``calibration_metrics``, ``set_size_metrics``, ``imdb_ids``,
    plus the per-fold DataFrames.
    """
    bundle_path = paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{target}.joblib"
    if not bundle_path.is_file():
        raise RuntimeError(f"Missing Phase 4 winner for {target}: {bundle_path}")
    bundle = joblib.load(bundle_path)
    matrix_name = bundle["matrix"]
    family = bundle["family"]
    estimator = bundle["estimator"]
    score_method = bundle.get("score_method", "predict_proba")
    task = _task_for(target)

    logger.info(
        "Phase 5 calibration | target=%s family=%s matrix=%s task=%s",
        target, family, matrix_name, task,
    )

    X_cal, y_cal, imdb_ids = _load_cal_features(matrix_name, target)
    logger.info("Calibration set: %d films, %d features", len(X_cal), X_cal.shape[1])

    out: dict[str, Any] = {
        "target": target,
        "matrix": matrix_name,
        "family": family,
        "task": task,
        "imdb_ids": imdb_ids,
        "phase4_oof_metric": bundle.get("oof_metrics_global"),
    }

    name = run_dir_name or f"phase5_{target}"
    with save_run(
        phase="phase_5",
        name=name,
        params={
            "target": target,
            "matrix": matrix_name,
            "family": family,
            "task": task,
            "confidence_levels": CONFIDENCE_LEVELS,
            "n_cal_films": int(len(X_cal)),
            "n_features": int(X_cal.shape[1]),
        },
        preprocessing={
            "phase4_winner_artifact": str(bundle_path.name),
            "score_method": score_method,
        },
        features=list(X_cal.columns),
    ) as run:
        if task == "classification":
            # 1. Probability calibration
            prob_results = []
            for method in CLASSIFICATION_METHODS:
                res = evaluate_calibration_method(
                    estimator, X_cal, y_cal, method=method,
                    target=target, score_method=score_method,
                )
                prob_results.append(res)
            best_prob = select_best_method(prob_results)

            # 2. Conformal classification — uses the BEST probability
            # calibration method picked above, re-fit per fold inside
            # the conformal CV for honest no-leakage evaluation.
            # ``calibration_method=None`` uses the base estimator
            # directly (only valid for families with native
            # predict_proba like XGBoost when "uncalibrated" wins).
            conformal_calibration_method = (
                None if best_prob.method == "uncalibrated"
                else best_prob.method
            )
            # SVM-RBF guard: "uncalibrated" should never win for SVM
            # because it has no predict_proba; if it does (shouldn't),
            # fall back to sigmoid Platt for the conformal wrapper.
            if (
                conformal_calibration_method is None
                and not hasattr(estimator, "predict_proba")
            ):
                conformal_calibration_method = "sigmoid"

            conf_results = []
            for cs in CLASSIFICATION_CONFORMITY_SCORES:
                res = evaluate_conformal_classifier(
                    estimator, X_cal, y_cal,
                    target=target, conformity_score=cs,
                    calibration_method=conformal_calibration_method,
                )
                conf_results.append(res)
            best_conf = _select_best_conformal_score_classification(conf_results)

            out["probability_results"] = prob_results
            out["conformal_results"] = conf_results
            out["best_probability"] = best_prob
            out["best_conformal"] = best_conf
        else:
            # Regression: no probability calibration; only conformal intervals.
            conf_results = []
            for cs in REGRESSION_CONFORMITY_SCORES:
                res = evaluate_conformal_regressor(
                    estimator, X_cal, y_cal,
                    target=target, conformity_score=cs,
                )
                conf_results.append(res)
            best_conf = _select_best_conformal_score_regression(conf_results)
            out["probability_results"] = []
            out["conformal_results"] = conf_results
            out["best_probability"] = None
            out["best_conformal"] = best_conf

        # Save the calibrated wrapper bundle.
        wrapper_bundle: dict[str, Any] = {
            "target": target,
            "matrix": matrix_name,
            "family": family,
            "task": task,
            "phase4_winner_path": str(bundle_path),
            "phase4_oof_metric": bundle.get("oof_metrics_global"),
            "feature_columns": list(X_cal.columns),
            "score_method": score_method,
            "deployed_confidence_levels": CONFIDENCE_LEVELS,
            "best_probability_method": (
                out["best_probability"].method
                if out["best_probability"] else None
            ),
            "probability_calibrator": (
                out["best_probability"].deployed_calibrator
                if out["best_probability"] else None
            ),
            "best_conformal_score": out["best_conformal"].conformity_score,
            "conformal_wrapper": out["best_conformal"].deployed_wrapper,
            "calibration_metrics": {
                **(out["best_probability"].aggregate_metrics
                   if out["best_probability"] else {}),
                **out["best_conformal"].aggregate_metrics,
            },
        }
        out_path = paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{target}.joblib"
        joblib.dump(wrapper_bundle, out_path)
        run.save_model(wrapper_bundle)
        logger.info("Saved Phase 5 calibrated wrapper for %s -> %s", target, out_path.name)

        # Persist metrics to run.
        metrics_payload = {
            "target": target,
            "task": task,
            "best_probability_method": wrapper_bundle["best_probability_method"],
            "best_conformal_score": wrapper_bundle["best_conformal_score"],
            "calibration_metrics": wrapper_bundle["calibration_metrics"],
            "all_probability_methods": {
                r.method: r.aggregate_metrics for r in out["probability_results"]
            },
            "all_conformal_scores": {
                r.conformity_score: r.aggregate_metrics for r in out["conformal_results"]
            },
        }
        run.record_metrics(metrics_payload)
        run.append_to_runs_md(
            model_family=f"phase5_calib_{family}",
            features_group=matrix_name,
            key_metric=_format_run_summary(out, task),
            notes="Phase 5 Layer 2 calibration",
        )

    return out


def _format_run_summary(out: dict[str, Any], task: str) -> str:
    parts = []
    if task == "classification":
        bp = out["best_probability"]
        bc = out["best_conformal"]
        parts.append(f"prob={bp.method} ECE={bp.aggregate_metrics['ece_mean']:.3f}")
        parts.append(
            f"conf={bc.conformity_score} cov@0.9={bc.aggregate_metrics['empirical_coverage_at_0.9_mean']:.3f} "
            f"singleton@0.9={bc.aggregate_metrics['singleton_rate_at_0.9_mean']:.3f}"
        )
    else:
        bc = out["best_conformal"]
        parts.append(
            f"conf={bc.conformity_score} cov@0.9={bc.aggregate_metrics['empirical_coverage_at_0.9_mean']:.3f} "
            f"width@0.9={bc.aggregate_metrics['mean_width_at_0.9_mean']:.3f}"
        )
    return "; ".join(parts)


def run_phase5(targets: tuple[str, ...] = ALL_TARGETS) -> dict[str, dict]:
    """Top-level orchestrator. Calibrate every target; emit deliverable CSVs."""
    paths.ensure_dirs()

    results = {}
    for target in targets:
        results[target] = calibrate_target(target)

    # Build the consolidated CSVs.
    coverage_rows = []
    set_size_rows = []
    width_rows = []
    cal_metric_rows = []

    for target, res in results.items():
        for cr in res["conformal_results"]:
            for _, row in cr.per_fold_metrics.iterrows():
                base = {
                    "target": target,
                    "conformity_score": cr.conformity_score,
                    "fold": int(row["fold"]),
                    "level": float(row["level"]),
                    "empirical_coverage": float(row["empirical_coverage"]),
                    "n_eval": int(row["n_eval"]),
                }
                coverage_rows.append(base)
                if res["task"] == "classification":
                    set_size_rows.append({
                        **base,
                        "mean_set_size": float(row["mean_set_size"]),
                        "singleton_rate": float(row["singleton_rate"]),
                        "refer_rate": float(row["refer_rate"]),
                        "empty_rate": float(row["empty_rate"]),
                    })
                else:
                    width_rows.append({
                        **base,
                        "mean_width": float(row["mean_width"]),
                        "median_width": float(row["median_width"]),
                    })
        for pr in res["probability_results"]:
            for _, row in pr.per_fold_metrics.iterrows():
                cal_metric_rows.append({
                    "target": target,
                    "method": pr.method,
                    "fold": int(row["fold"]),
                    "ece": float(row["ece"]),
                    "mce": float(row["mce"]),
                    "brier": float(row["brier"]),
                    "log_loss": float(row["log_loss"]),
                    "n_eval": int(row["n_eval"]),
                })

    pd.DataFrame(coverage_rows).to_csv(
        paths.REPORTS_TABLES_DIR / "phase5_coverage.csv", index=False,
    )
    if set_size_rows:
        pd.DataFrame(set_size_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase5_set_sizes.csv", index=False,
        )
    if width_rows:
        pd.DataFrame(width_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase5_interval_widths.csv", index=False,
        )
    if cal_metric_rows:
        pd.DataFrame(cal_metric_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase5_calibration_metrics.csv", index=False,
        )

    logger.info(
        "Phase 5 complete: wrote phase5_coverage.csv, phase5_set_sizes.csv, "
        "phase5_interval_widths.csv, phase5_calibration_metrics.csv",
    )
    return results
