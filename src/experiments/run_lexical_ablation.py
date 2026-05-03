"""Phase 3b lexical-feature ablation runner.

Wraps the multi-family baseline trainer in a ``save_run`` block so the
lexical ablation row produces a complete `runs/phase_3/<timestamp>_lexical/`
directory: the params and feature list at entry, the metrics on
completion, and the full INFO/DEBUG trace in ``run.log``.

The ablation runs the lexical-augmented matrix through all four model
families (linear, histgb, knn, svm). Each family produces its own
floor (Phase 3a `dialogue_only_logged` numbers under that family) and
its own actual-with-lexical numbers; the lift is family-specific. The
linear-family lift remains the "official" ablation row for the
linear-baseline ablation methodology, while the other families'
lifts disambiguate "feature issue" from "model family issue".

Run from the project root::

    python -m src.experiments.run_lexical_ablation

The output is appended to ``reports/tables/phase3_ablation.csv`` (or
the file is created if it does not exist) with one row per
(model_family, target, metric) combination carrying the
predicted-vs-actual lift over the corresponding family's Phase 3a
revised dialogue-only floor.
"""

from __future__ import annotations

# Allow running by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_features
from src.features.targets import LOG_ROI_COL, add_targets
from src.models.baseline.train import BaselineTrainConfig, MODEL_FAMILIES, evaluate_feature_set
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_phase_3a_floor() -> pd.DataFrame:
    """Load the Phase 3a revised dialogue-only floor for all families and eval sets.

    Returns a DataFrame indexed by (model_family, eval_set, target,
    metric) with a single ``floor`` column. Used to compute
    family-specific and eval-set-specific lift.
    """
    table = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3a_baseline.csv")
    floor = table[table["feature_set"] == "dialogue_only_logged"][
        ["model_family", "eval_set", "target", "metric", "value"]
    ].rename(columns={"value": "floor"})
    return floor.set_index(["model_family", "eval_set", "target", "metric"])


# Pre-registered lift bands from proposal v2 Section 3. The original
# pre-registration was made in R² for the regression target; with R²
# removed from the metric set, the regression band is translated to
# RMSE on the linear family's OOF-evaluation floor (the historical
# reference). The translation: R² lift band of +0.010 to +0.025 on a
# floor of R² = 0.052 with var(log_roi) ~ 1.8 corresponds to RMSE
# lift band of approximately -0.020 to -0.010 (lower is better for
# RMSE). Classification AUC bands are unchanged.
LEXICAL_PREDICTED_LIFT: dict[tuple[str, str], tuple[float, float]] = {
    ("log_roi", "rmse"): (-0.020, -0.010),
    ("roi_gt_1", "auc_roc"): (0.000, 0.010),
    ("roi_gt_2", "auc_roc"): (0.015, 0.035),
}


def main() -> None:
    """Run the lexical ablation, persist artifacts, append RUNS.md row."""
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

    # Lexical-augmented config (revised dialogue-only baseline + lexical).
    feature_cfg = BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
        include_lexical=True,
    )

    # Construct the augmented feature matrix once so we can capture the
    # exact column list for ``features_used.json`` before training.
    X = build_baseline_features(df_train, feature_cfg)
    feature_names = list(X.columns)

    train_cfg = BaselineTrainConfig()

    with save_run(
        phase="phase_3",
        name="lexical_multifamily",
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
        },
        preprocessing={
            "split": "70_15_15_seed_42",
            "stratification": "(primary_genre_bucketed, decade_bucket)",
            "rare_cell_threshold": 5,
            "structural_features_count": 7,
            "log_transform_structural": True,
            "include_log_runtime": True,
            "include_log_budget": False,
            "include_lexical": True,
            "lexical_feature_count": 13,
            "linear_imputer": "SimpleImputer(strategy='median')",
            "linear_scaler": "StandardScaler",
            "histgb_preprocessing": "passthrough (native NaN handling, no scaling)",
            "knn_preprocessing": "SimpleImputer(median) + StandardScaler",
            "svm_preprocessing": "SimpleImputer(median) + StandardScaler",
            "frequency_source": "wordfreq Zipf (English; mixture includes OpenSubtitles, Wikipedia, etc.)",
            "frequency_source_deviation_note": (
                "Proposal v2 specified SUBTLEX-US; canonical SUBTLEX-US download URLs "
                "returned 404 at implementation time. wordfreq is the implementation backend, "
                "with OpenSubtitles in its English source mixture preserving the proposal's "
                "subtitle-domain match argument in spirit. Surfaced for planning-conversation review."
            ),
        },
        features=feature_names,
        notes=(
            "Phase 3b ablation row 1 of 5. Lexical features (14) added on top of revised "
            "dialogue-only baseline, evaluated across 4 model families to disambiguate "
            "feature-issue vs model-family-issue. Pre-registered linear lift: log_roi R^2 "
            "+0.010 to +0.025; roi_gt_1 AUC +0.000 to +0.010; roi_gt_2 AUC +0.015 to +0.035."
        ),
    ) as run:
        logger.info("Starting lexical ablation evaluation across %d families", len(MODEL_FAMILIES))
        rows = evaluate_feature_set(
            df_train, feature_cfg, train_cfg,
            set_name="dialogue_only_logged_lexical",
        )
        # Convert metric rows into the metrics dict for save_run, keyed by family.
        metrics: dict[str, float] = {}
        for r in rows:
            key = f"{r['model_family']}_{r['target']}_{r['metric']}"
            metrics[key] = round(float(r["value"]), 4)
            metrics[f"{key}_ci_lo"] = round(float(r["ci_lo"]), 4)
            metrics[f"{key}_ci_hi"] = round(float(r["ci_hi"]), 4)
        run.record_metrics(metrics)

        # Build the ablation table with per-family predicted vs actual lift.
        floor = _load_phase_3a_floor()
        ablation_rows = _build_ablation_rows(rows, floor)
        ablation_path = paths.REPORTS_TABLES_DIR / "phase3_ablation.csv"
        ablation_df = pd.DataFrame(ablation_rows)
        # Drop any prior lexical rows so we don't accumulate stale entries.
        if ablation_path.is_file():
            existing = pd.read_csv(ablation_path)
            existing = existing[existing["feature_group"] != "lexical"]
            ablation_df = pd.concat([existing, ablation_df], ignore_index=True)
        ablation_df.to_csv(ablation_path, index=False)
        logger.info("Ablation table written: %d rows total", len(ablation_df))

        # Compute headline-row labels for RUNS.md from the linear family,
        # OOF eval set (the historical reference for the proposal's lift
        # predictions). RMSE on log_roi and AUC-ROC on roi_gt_2 are the
        # report-facing summary numbers.
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
            features_group="structural + lexical",
            key_metric=(
                f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}"
            ),
            notes="Phase 3b row 1: lexical (14 features); 4 families × 2 eval sets",
        )


def _build_ablation_rows(metric_rows: list[dict], floor: pd.DataFrame) -> list[dict]:
    """Build per-(family, metric) ablation rows with predicted vs actual lift.

    Each ablation row captures: feature group, model family, target,
    metric, the family-specific Phase 3a floor, the lexical-augmented
    actual, the lift, the pre-registered linear lift range, and
    whether the linear-family actual lift falls inside the
    pre-registered band. Pre-registered bands apply to the linear
    family only (the proposal's prediction was made for the linear
    baseline ablation methodology); other families' lifts are
    reported but ``in_predicted_band`` is left blank.
    """
    out: list[dict] = []
    for r in metric_rows:
        key = (r["model_family"], r["eval_set"], r["target"], r["metric"])
        if key not in floor.index:
            continue
        floor_value = float(floor.loc[key, "floor"])
        actual = float(r["value"])
        # NaN floor (e.g., log_loss for SVM) propagates as NaN lift.
        if np.isnan(floor_value) or np.isnan(actual):
            lift = float("nan")
        else:
            lift = actual - floor_value
        predicted = LEXICAL_PREDICTED_LIFT.get((r["target"], r["metric"]))
        in_band: bool | None
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            # Pre-registered bands apply only to the linear family's
            # OOF (validation) numbers, the historical reference for
            # the proposal's lift predictions.
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": "lexical",
            "model_family": r["model_family"],
            "eval_set": r["eval_set"],
            "target": r["target"],
            "metric": r["metric"],
            "task": r["task"],
            "phase_3a_floor": round(floor_value, 4) if not np.isnan(floor_value) else None,
            "phase_3b_actual": round(actual, 4) if not np.isnan(actual) else None,
            "lift": round(lift, 4) if not np.isnan(lift) else None,
            "predicted_lift_low": predicted[0] if predicted else None,
            "predicted_lift_high": predicted[1] if predicted else None,
            "in_predicted_band": in_band,
            "ci_lo": round(float(r["ci_lo"]), 4) if not np.isnan(r.get("ci_lo", float("nan"))) else None,
            "ci_hi": round(float(r["ci_hi"]), 4) if not np.isnan(r.get("ci_hi", float("nan"))) else None,
        })
    return out


if __name__ == "__main__":
    main()
