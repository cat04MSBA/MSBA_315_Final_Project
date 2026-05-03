"""Phase 3c combinations sub-phase runner.

Evaluates four pre-specified feature-group combinations against the
Phase 3a revised dialogue-only floor, using the same 4-family
multi-family harness as the Phase 3b standalone ablations. The
combinations are locked at proposal time
(`docs/proposals/phase3c_combinations_proposal.md`) and not expanded
after seeing results; this preserves the pre-registration discipline
at the combinations level.

Output appended to `reports/tables/phase3c_combinations.csv` (separate
table from `phase3_ablation.csv` so the Phase 3b standalone-group
ablation table stays clean).

Run from the project root::

    python -m src.experiments.run_combinations_ablation
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_features
from src.features.targets import LOG_ROI_COL, add_targets
from src.models.baseline.train import BaselineTrainConfig, MODEL_FAMILIES, evaluate_feature_set
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pre-registered combinations (locked; matches proposal Section 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinationSpec:
    """One pre-specified feature-group combination."""
    name: str
    description: str
    include_lexical: bool
    include_sentiment: bool
    include_topic: bool
    include_character_network: bool
    include_embedding: bool


COMBINATIONS: tuple[CombinationSpec, ...] = (
    CombinationSpec(
        name="all_five",
        description="All five Phase 3b feature groups joined onto the structural baseline",
        include_lexical=True,
        include_sentiment=True,
        include_topic=True,
        include_character_network=True,
        include_embedding=True,
    ),
    CombinationSpec(
        name="partial_positives",
        description="The three Phase 3b partial-positive groups (topic + character_network + embedding)",
        include_lexical=False,
        include_sentiment=False,
        include_topic=True,
        include_character_network=True,
        include_embedding=True,
    ),
    CombinationSpec(
        name="topic_plus_cn",
        description="Topic + character_network (complementary classification targets)",
        include_lexical=False,
        include_sentiment=False,
        include_topic=True,
        include_character_network=True,
        include_embedding=False,
    ),
    CombinationSpec(
        name="semantic_trio",
        description="Sentiment + topic + embedding (planning-conversation 'semantic' hypothesis)",
        include_lexical=False,
        include_sentiment=True,
        include_topic=True,
        include_character_network=False,
        include_embedding=True,
    ),
)


# Pre-registered linear-family OOF lift bands per combination, per
# (target, metric). Lower-is-better metrics use a (low, high) band where
# both bounds are negative or near zero. Higher-is-better metrics use a
# (low, high) band of positive numbers. See proposal Section 3 for the
# mechanism behind each band.
PREDICTED_LIFT: dict[tuple[str, str, str], tuple[float, float]] = {
    # (combination_name, target, metric) -> (lift_low, lift_high)
    ("all_five", "log_roi", "rmse"): (-0.025, -0.005),
    ("all_five", "roi_gt_1", "auc_roc"): (0.020, 0.040),
    ("all_five", "roi_gt_2", "auc_roc"): (0.010, 0.030),

    ("partial_positives", "log_roi", "rmse"): (-0.025, -0.010),
    ("partial_positives", "roi_gt_1", "auc_roc"): (0.025, 0.045),
    ("partial_positives", "roi_gt_2", "auc_roc"): (0.015, 0.035),

    ("topic_plus_cn", "log_roi", "rmse"): (-0.005, 0.010),
    ("topic_plus_cn", "roi_gt_1", "auc_roc"): (0.025, 0.040),
    ("topic_plus_cn", "roi_gt_2", "auc_roc"): (0.015, 0.030),

    ("semantic_trio", "log_roi", "rmse"): (-0.015, 0.000),
    ("semantic_trio", "roi_gt_1", "auc_roc"): (0.020, 0.035),
    ("semantic_trio", "roi_gt_2", "auc_roc"): (0.000, 0.020),
}


# ---------------------------------------------------------------------------
# Floor loading + lift computation
# ---------------------------------------------------------------------------


def _load_phase_3a_floor() -> pd.DataFrame:
    """Load the Phase 3a revised dialogue-only floor for all families and eval sets.

    Returns a DataFrame indexed by (model_family, eval_set, target,
    metric) with a single ``floor`` column.
    """
    table = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3a_baseline.csv")
    floor = table[table["feature_set"] == "dialogue_only_logged"][
        ["model_family", "eval_set", "target", "metric", "value"]
    ].rename(columns={"value": "floor"})
    return floor.set_index(["model_family", "eval_set", "target", "metric"])


def _build_ablation_rows(
    metric_rows: list[dict],
    floor: pd.DataFrame,
    combination_name: str,
) -> list[dict]:
    """Build per-(family, eval_set, metric) ablation rows with predicted vs actual lift.

    Mirrors the Phase 3b runner's row schema. ``in_predicted_band`` is
    populated only for the linear family OOF rows where a band is
    pre-registered for that (combination, target, metric).
    """
    out: list[dict] = []
    for r in metric_rows:
        key = (r["model_family"], r["eval_set"], r["target"], r["metric"])
        if key not in floor.index:
            continue
        floor_value = float(floor.loc[key, "floor"])
        actual = float(r["value"])
        if np.isnan(floor_value) or np.isnan(actual):
            lift = float("nan")
        else:
            lift = actual - floor_value
        predicted = PREDICTED_LIFT.get(
            (combination_name, r["target"], r["metric"])
        )
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": combination_name,
            "model_family": r["model_family"],
            "eval_set": r["eval_set"],
            "target": r["target"],
            "metric": r["metric"],
            "task": r["task"],
            "phase_3a_floor": round(floor_value, 4) if not np.isnan(floor_value) else None,
            "phase_3c_actual": round(actual, 4) if not np.isnan(actual) else None,
            "lift": round(lift, 4) if not np.isnan(lift) else None,
            "predicted_lift_low": predicted[0] if predicted else None,
            "predicted_lift_high": predicted[1] if predicted else None,
            "in_predicted_band": in_band,
            "ci_lo": (
                round(float(r["ci_lo"]), 4)
                if not np.isnan(r.get("ci_lo", float("nan")))
                else None
            ),
            "ci_hi": (
                round(float(r["ci_hi"]), 4)
                if not np.isnan(r.get("ci_hi", float("nan")))
                else None
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the four pre-specified combinations through the multi-family harness."""
    paths.ensure_dirs()

    # Load corpus + splits, attach targets, restrict to training films.
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_with_targets = add_targets(df)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = (
        df_with_targets[df_with_targets["imdb_id"].isin(train_ids)]
        .reset_index(drop=True)
    )

    train_cfg = BaselineTrainConfig()
    floor = _load_phase_3a_floor()
    ablation_path = paths.REPORTS_TABLES_DIR / "phase3c_combinations.csv"

    # Build the ablation table fresh (drop any prior Phase 3c rows).
    ablation_rows: list[dict] = []

    for spec in COMBINATIONS:
        feature_cfg = BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
            include_lexical=spec.include_lexical,
            include_sentiment=spec.include_sentiment,
            include_topic=spec.include_topic,
            include_character_network=spec.include_character_network,
            include_embedding=spec.include_embedding,
        )

        # Build the matrix once outside save_run so we can capture the
        # feature column list for the run's metadata.
        X = build_baseline_features(df_train, feature_cfg)
        feature_names = list(X.columns)

        run_name = f"combinations_{spec.name}"
        with save_run(
            phase="phase_3",
            name=run_name,
            params={
                "model_families": list(MODEL_FAMILIES),
                "linear_alpha_grid": list(train_cfg.ridge_alphas),
                "linear_C_grid": list(train_cfg.logistic_Cs),
                "histgb_max_iter": train_cfg.histgb_max_iter,
                "histgb_max_depth": train_cfg.histgb_max_depth,
                "histgb_learning_rate": train_cfg.histgb_learning_rate,
                "knn_n_neighbors": train_cfg.knn_n_neighbors,
                "knn_weights": train_cfg.knn_weights,
                "svm_C": train_cfg.svm_C,
                "svm_gamma": train_cfg.svm_gamma,
                "n_cv_folds": train_cfg.n_cv_folds,
                "bootstrap_iter": train_cfg.bootstrap_iter,
                "bootstrap_alpha": train_cfg.bootstrap_alpha,
                "seed": train_cfg.seed,
                "combination_name": spec.name,
                "combination_description": spec.description,
                "include_lexical": spec.include_lexical,
                "include_sentiment": spec.include_sentiment,
                "include_topic": spec.include_topic,
                "include_character_network": spec.include_character_network,
                "include_embedding": spec.include_embedding,
            },
            preprocessing={
                "split": "70_15_15_seed_42",
                "stratification": "(primary_genre_bucketed, decade_bucket)",
                "rare_cell_threshold": 5,
                "structural_features_count": 7,
                "log_transform_structural": True,
                "include_log_runtime": True,
                "include_log_budget": False,
                "feature_count_total": len(feature_names),
                "linear_imputer": "SimpleImputer(strategy='median')",
                "linear_scaler": "StandardScaler",
                "histgb_preprocessing": "passthrough (native NaN handling, no scaling)",
                "knn_preprocessing": "SimpleImputer(median) + StandardScaler",
                "svm_preprocessing": "SimpleImputer(median) + StandardScaler",
            },
            features=feature_names,
            notes=(
                f"Phase 3c combination '{spec.name}': {spec.description}. "
                f"{len(feature_names)} features. Pre-registered linear-OOF lift bands "
                f"in proposal Section 3."
            ),
        ) as run:
            logger.info(
                "Phase 3c combination '%s': %d features (%d total columns)",
                spec.name, len(feature_names), len(feature_names),
            )
            rows = evaluate_feature_set(
                df_train,
                feature_cfg,
                train_cfg,
                set_name=f"phase_3c_{spec.name}",
            )
            metrics: dict[str, float] = {}
            for r in rows:
                key = f"{r['model_family']}_{r['eval_set']}_{r['target']}_{r['metric']}"
                metrics[key] = round(float(r["value"]), 4)
                metrics[f"{key}_ci_lo"] = round(float(r["ci_lo"]), 4)
                metrics[f"{key}_ci_hi"] = round(float(r["ci_hi"]), 4)
            run.record_metrics(metrics)

            # Append ablation rows for this combination.
            ablation_rows.extend(_build_ablation_rows(rows, floor, spec.name))

            # Headline RUNS.md row from linear OOF.
            linear_log_roi_rmse = next(
                r for r in rows
                if r["model_family"] == "linear"
                and r["eval_set"] == "oof"
                and r["target"] == LOG_ROI_COL
                and r["metric"] == "rmse"
            )
            linear_roi_gt_2_auc = next(
                r for r in rows
                if r["model_family"] == "linear"
                and r["eval_set"] == "oof"
                and r["target"] == "roi_gt_2"
                and r["metric"] == "auc_roc"
            )
            histgb_log_roi_rmse = next(
                r for r in rows
                if r["model_family"] == "histgb"
                and r["eval_set"] == "oof"
                and r["target"] == LOG_ROI_COL
                and r["metric"] == "rmse"
            )
            run.append_to_runs_md(
                model_family="4-family (linear, histgb, knn, svm)",
                features_group=f"combination: {spec.name}",
                key_metric=(
                    f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                    f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                    f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}"
                ),
                notes=(
                    f"Phase 3c combination ({len(feature_names)} features); "
                    f"{spec.description[:80]}"
                ),
            )

    # Write the combinations table fresh (overwrite if exists).
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(
        "Phase 3c combinations table written: %d rows (%d combinations)",
        len(ablation_df), len(COMBINATIONS),
    )


if __name__ == "__main__":
    main()
