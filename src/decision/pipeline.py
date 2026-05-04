"""Phase 6 orchestrator: end-to-end decision pipeline + sensitivity sweep.

For the headline target ``roi_gt_2``:

1. Load the Phase 5 calibrated wrapper from
   ``data/processed/phase5_calibrated_model_roi_gt_2.joblib``.
2. Compute calibrated probabilities for the 257 calibration films.
3. For each pre-registered cost-matrix variant, apply the decision
   rule and compute total cost + action distribution.
4. Compute baseline costs (Always-Greenlight, Always-Pass,
   Read-Everything, Random, Genre-prior).
5. Per-genre breakdowns under the default cost matrix.
6. Persist a deployable decision-pipeline artifact.

Outputs to ``reports/tables/phase6_*.csv`` and a save_run directory
under ``runs/phase_6/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# sklearn 1.8 deprecation noise from CalibratedClassifierCV's Platt fit.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.decision.baselines import BASELINES, _compute_genre_priors
from src.decision.cost_matrix import (
    CostMatrix,
    DEFAULT_COST_MATRIX,
    all_sensitivity_variants,
    asymmetry_variants,
    base_magnitude_variants,
    per_genre_matrices,
    refer_cost_variants,
)
from src.decision.evaluation import EvaluationResult, evaluate
from src.decision.rule import DecisionResult, decide_batch
from src.experiments.save_run import save_run
from src.features.targets import add_targets
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


HEADLINE_TARGET: str = "roi_gt_2"


def load_cal_inputs(target: str = HEADLINE_TARGET) -> tuple[
    list[str], np.ndarray, np.ndarray, list[str],
]:
    """Return (imdb_ids, calibrated_probabilities, true_labels, genres) for the cal set."""
    bundle = joblib.load(paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{target}.joblib")
    matrix_name = bundle["matrix"]

    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    cal_ids = splits.loc[splits["split"] == "cal", "imdb_id"]
    df_cal = df_full[df_full["imdb_id"].isin(cal_ids)].reset_index(drop=True)
    matrix_spec = MATRICES[matrix_name]
    X_cal = build_matrix(matrix_spec, df_cal)

    calibrator = bundle["probability_calibrator"]
    if calibrator is not None:
        probs = calibrator.predict_proba(X_cal)[:, 1]
    else:
        # Regression target or natively-uncalibrated estimator: fall back to
        # the underlying estimator's predict_proba via the saved phase4 path.
        phase4 = joblib.load(bundle["phase4_winner_path"])
        probs = phase4["estimator"].predict_proba(X_cal)[:, 1]

    return (
        df_cal["imdb_id"].astype(str).tolist(),
        np.asarray(probs, dtype=float),
        df_cal[target].astype(int).values,
        df_cal["primary_genre_bucketed"].astype(str).tolist(),
    )


def _per_genre_financials() -> tuple[dict[str, float], dict[str, float]]:
    """Median budget and revenue per primary genre on the train split."""
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df[df["imdb_id"].isin(train_ids)]
    counts = df_train.groupby("primary_genre_bucketed").size()
    keep_genres = counts[counts >= 30].index.tolist()
    df_train = df_train[df_train["primary_genre_bucketed"].isin(keep_genres)]
    budgets = df_train.groupby("primary_genre_bucketed")["budget"].median().to_dict()
    revenues = df_train.groupby("primary_genre_bucketed")["revenue"].median().to_dict()
    return (
        {str(k): float(v) for k, v in budgets.items()},
        {str(k): float(v) for k, v in revenues.items()},
    )


def _evaluate_baselines(
    imdb_ids: list[str],
    true_labels: np.ndarray,
    genres: list[str],
    cost_matrix: CostMatrix,
) -> list[EvaluationResult]:
    """Evaluate all five baselines under one cost matrix."""
    out: list[EvaluationResult] = []
    for name, fn in BASELINES.items():
        actions = fn(imdb_ids, genres=genres)
        out.append(evaluate(name, actions, true_labels.tolist(), cost_matrix))
    return out


def _evaluate_system(
    imdb_ids: list[str],
    probs: np.ndarray,
    true_labels: np.ndarray,
    cost_matrix: CostMatrix,
) -> tuple[EvaluationResult, list[DecisionResult]]:
    decisions = decide_batch(imdb_ids, probs.tolist(), cost_matrix)
    actions = [d.recommended_action for d in decisions]
    eval_result = evaluate("system", actions, true_labels.tolist(), cost_matrix)
    return eval_result, decisions


def run_phase6() -> dict[str, Any]:
    """End-to-end Phase 6 orchestrator. Persists all deliverables."""
    paths.ensure_dirs()

    imdb_ids, probs, true_labels, genres = load_cal_inputs(HEADLINE_TARGET)
    logger.info(
        "Phase 6 inputs: %d cal films, mean prob=%.3f, positive rate=%.3f",
        len(imdb_ids), float(probs.mean()), float(true_labels.mean()),
    )

    # 1. Default cost matrix: per-film decisions + system evaluation.
    system_default, decisions_default = _evaluate_system(
        imdb_ids, probs, true_labels, DEFAULT_COST_MATRIX,
    )
    logger.info(
        "System under default cost matrix: total_cost=$%dM action_dist=%s",
        int(system_default.total_cost / 1e6), system_default.action_proportions,
    )

    # 2. Baselines under default cost matrix.
    baseline_results = _evaluate_baselines(
        imdb_ids, true_labels, genres, DEFAULT_COST_MATRIX,
    )
    for r in baseline_results:
        logger.info(
            "Baseline %s under default: total_cost=$%dM action_dist=%s",
            r.strategy_name, int(r.total_cost / 1e6), r.action_proportions,
        )

    # 3. Sensitivity sweep across all pre-registered cost-matrix variants.
    sweep_rows: list[dict] = []
    for variant in all_sensitivity_variants():
        sys_result, _ = _evaluate_system(imdb_ids, probs, true_labels, variant)
        sweep_rows.append({
            "cost_matrix_name": variant.name,
            "cost_greenlight_flop": variant.cost_greenlight_flop,
            "cost_pass_hit": variant.cost_pass_hit,
            "cost_refer_flop": variant.cost_refer_flop,
            "total_cost_system": sys_result.total_cost,
            "n_greenlight": sys_result.action_counts["Greenlight"],
            "n_pass": sys_result.action_counts["Pass"],
            "n_refer": sys_result.action_counts["Refer"],
            "p_greenlight": sys_result.action_proportions["Greenlight"],
            "p_pass": sys_result.action_proportions["Pass"],
            "p_refer": sys_result.action_proportions["Refer"],
            "n_decisions": sys_result.n_decisions,
        })

    # 4. Per-genre cost-matrix variant.
    budgets, revenues = _per_genre_financials()
    per_genre_cms = per_genre_matrices(budgets, revenues, DEFAULT_COST_MATRIX)
    per_genre_actions: list[str] = []
    per_genre_total_cost = 0.0
    for iid, p, label, genre in zip(imdb_ids, probs, true_labels, genres):
        cm = per_genre_cms.get(genre, DEFAULT_COST_MATRIX)
        d = decide_batch([iid], [float(p)], cm)[0]
        per_genre_actions.append(d.recommended_action)
        per_genre_total_cost += cm.realized_cost(d.recommended_action, int(label))
    sweep_rows.append({
        "cost_matrix_name": "per_genre_default",
        "cost_greenlight_flop": float("nan"),
        "cost_pass_hit": float("nan"),
        "cost_refer_flop": DEFAULT_COST_MATRIX.cost_refer_flop,
        "total_cost_system": per_genre_total_cost,
        "n_greenlight": int(sum(1 for a in per_genre_actions if a == "Greenlight")),
        "n_pass": int(sum(1 for a in per_genre_actions if a == "Pass")),
        "n_refer": int(sum(1 for a in per_genre_actions if a == "Refer")),
        "p_greenlight": float(sum(1 for a in per_genre_actions if a == "Greenlight") / len(per_genre_actions)),
        "p_pass": float(sum(1 for a in per_genre_actions if a == "Pass") / len(per_genre_actions)),
        "p_refer": float(sum(1 for a in per_genre_actions if a == "Refer") / len(per_genre_actions)),
        "n_decisions": len(per_genre_actions),
    })

    # 5. Per-genre action breakdown (default cost matrix).
    per_genre_rows: list[dict] = []
    actions_default = [d.recommended_action for d in decisions_default]
    df_pg = pd.DataFrame({
        "genre": genres,
        "action_default": actions_default,
        "action_per_genre_cm": per_genre_actions,
        "true_label": true_labels,
        "probability": probs,
    })
    for genre, sub in df_pg.groupby("genre"):
        n = len(sub)
        per_genre_rows.append({
            "genre": genre,
            "n": n,
            "n_pos": int(sub["true_label"].sum()),
            "default_p_greenlight": float((sub["action_default"] == "Greenlight").mean()),
            "default_p_pass": float((sub["action_default"] == "Pass").mean()),
            "default_p_refer": float((sub["action_default"] == "Refer").mean()),
            "per_genre_p_greenlight": float((sub["action_per_genre_cm"] == "Greenlight").mean()),
            "per_genre_p_pass": float((sub["action_per_genre_cm"] == "Pass").mean()),
            "per_genre_p_refer": float((sub["action_per_genre_cm"] == "Refer").mean()),
        })

    # ---- Persist tables ----
    decisions_df = pd.DataFrame([
        {
            "imdb_id": d.imdb_id,
            "calibrated_probability": d.calibrated_probability,
            "expected_cost_greenlight": d.expected_cost_greenlight,
            "expected_cost_pass": d.expected_cost_pass,
            "expected_cost_refer": d.expected_cost_refer,
            "recommended_action": d.recommended_action,
            "rationale": d.rationale,
            "true_label": int(true_labels[i]),
            "genre": genres[i],
        }
        for i, d in enumerate(decisions_default)
    ])
    decisions_df.to_csv(paths.REPORTS_TABLES_DIR / "phase6_decisions.csv", index=False)
    logger.info("Wrote phase6_decisions.csv (%d rows)", len(decisions_df))

    baseline_rows = [
        {
            "strategy": r.strategy_name,
            "cost_matrix": r.cost_matrix_name,
            "total_cost": r.total_cost,
            "p_greenlight": r.action_proportions["Greenlight"],
            "p_pass": r.action_proportions["Pass"],
            "p_refer": r.action_proportions["Refer"],
            "cost_per_film_M": r.total_cost / max(r.n_decisions, 1) / 1_000_000,
        }
        for r in (baseline_results + [system_default])
    ]
    pd.DataFrame(baseline_rows).to_csv(
        paths.REPORTS_TABLES_DIR / "phase6_baselines.csv", index=False,
    )
    logger.info("Wrote phase6_baselines.csv")

    pd.DataFrame(sweep_rows).to_csv(
        paths.REPORTS_TABLES_DIR / "phase6_sensitivity.csv", index=False,
    )
    logger.info("Wrote phase6_sensitivity.csv (%d rows)", len(sweep_rows))

    pd.DataFrame(per_genre_rows).to_csv(
        paths.REPORTS_TABLES_DIR / "phase6_per_genre_actions.csv", index=False,
    )
    logger.info("Wrote phase6_per_genre_actions.csv")

    # ---- save_run + deploy artifact ----
    with save_run(
        phase="phase_6",
        name="phase6_decision",
        params={
            "target": HEADLINE_TARGET,
            "default_cost_matrix": {
                "greenlight_flop": DEFAULT_COST_MATRIX.cost_greenlight_flop,
                "pass_hit": DEFAULT_COST_MATRIX.cost_pass_hit,
                "refer": DEFAULT_COST_MATRIX.cost_refer_flop,
            },
            "n_cal_films": len(imdb_ids),
            "sensitivity_variants": [v.name for v in all_sensitivity_variants()] + ["per_genre_default"],
        },
        preprocessing={
            "phase5_wrapper_artifact": str(
                paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{HEADLINE_TARGET}.joblib"
            ),
            "decision_rule": "expected_cost_minimization_tie_break_refer",
        },
        features=["calibrated_probability"],
    ) as run:
        bundle = {
            "target": HEADLINE_TARGET,
            "phase5_wrapper_path": str(
                paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{HEADLINE_TARGET}.joblib"
            ),
            "default_cost_matrix": DEFAULT_COST_MATRIX,
            "per_genre_cost_matrices": per_genre_cms,
            "default_total_cost": system_default.total_cost,
            "default_action_proportions": system_default.action_proportions,
        }
        out_path = paths.DATA_PROCESSED_DIR / f"phase6_decision_pipeline_{HEADLINE_TARGET}.joblib"
        joblib.dump(bundle, out_path)
        run.save_model(bundle)
        run.record_metrics({
            "default_cost_matrix": {
                "greenlight_flop": DEFAULT_COST_MATRIX.cost_greenlight_flop,
                "pass_hit": DEFAULT_COST_MATRIX.cost_pass_hit,
                "refer": DEFAULT_COST_MATRIX.cost_refer_flop,
            },
            "system_default_total_cost": system_default.total_cost,
            "system_default_action_proportions": system_default.action_proportions,
            "baselines_total_cost": {
                r.strategy_name: r.total_cost for r in baseline_results
            },
            "per_genre_default_total_cost": per_genre_total_cost,
        })
        run.append_to_runs_md(
            model_family="phase6_decision_rule",
            features_group="cost_matrix_default",
            key_metric=(
                f"system_total=${int(system_default.total_cost / 1e6)}M; "
                f"refer={system_default.action_proportions['Refer']:.2%}; "
                f"per_genre_total=${int(per_genre_total_cost / 1e6)}M"
            ),
            notes="Phase 6 Layer 3 cost-decision rule",
        )

    return {
        "system_default": system_default,
        "baseline_results": baseline_results,
        "decisions_default": decisions_default,
        "sweep_rows": sweep_rows,
        "per_genre_rows": per_genre_rows,
        "per_genre_total_cost": per_genre_total_cost,
    }
