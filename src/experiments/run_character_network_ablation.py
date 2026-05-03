"""Phase 3b character-network ablation runner.

Computes the 12-column character-network feature matrix on the full
corpus (plus 1 diagnostic column), saves to
``data/processed/features_character_network.parquet``, and runs the
multi-family baseline trainer with the network-augmented matrix.

The network-augmented matrix is the **revised dialogue-only** baseline
plus the 12 network model features — NOT the lexical-/sentiment-/topic-
augmented matrix. Standalone-lift methodology established by previous
groups; joint evaluation belongs in Phase 3c.

Run from the project root::

    python3 -m src.experiments.run_character_network_ablation
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle

import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_features
from src.features.character_network import (
    CharacterNetworkConfig,
    compute_character_network_features,
)
from src.features.targets import LOG_ROI_COL, add_targets
from src.models.baseline.train import (
    BaselineTrainConfig, MODEL_FAMILIES, evaluate_feature_set,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_phase_3a_floor() -> pd.DataFrame:
    table = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3a_baseline.csv")
    floor = table[table["feature_set"] == "dialogue_only_logged"][
        ["model_family", "eval_set", "target", "metric", "value"]
    ].rename(columns={"value": "floor"})
    return floor.set_index(["model_family", "eval_set", "target", "metric"])


# Pre-registered lift bands from character-network proposal v1 Section 4.
CHARACTER_NETWORK_PREDICTED_LIFT: dict[tuple[str, str], tuple[float, float]] = {
    ("log_roi", "rmse"): (-0.040, -0.010),
    ("log_roi", "mae"): (-0.030, -0.010),
    ("log_roi", "cvrmse"): (-0.030, -0.010),
    ("roi_gt_1", "auc_roc"): (0.000, 0.015),
    ("roi_gt_1", "pr_auc"): (0.000, 0.015),
    ("roi_gt_1", "log_loss"): (-0.020, 0.000),
    ("roi_gt_2", "auc_roc"): (0.020, 0.045),
    ("roi_gt_2", "pr_auc"): (0.015, 0.035),
    ("roi_gt_2", "log_loss"): (-0.025, -0.005),
}


def _ensure_character_network_features() -> pd.DataFrame:
    out_path = paths.DATA_PROCESSED_DIR / "features_character_network.parquet"
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    parquet_path = paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Parsed-screenplay pickle missing at {pkl_path}")

    df = pd.read_parquet(parquet_path).set_index("imdb_id")
    flags = df["data_quality_flag"].astype(bool)

    with pkl_path.open("rb") as f:
        parsed_corpus = pickle.load(f)

    if out_path.is_file():
        cached = pd.read_parquet(out_path)
        if len(cached) == len(parsed_corpus):
            logger.info(
                "Reusing cached character-network features at %s (%d films)",
                out_path, len(cached),
            )
            return cached

    cfg = CharacterNetworkConfig()
    out = compute_character_network_features(parsed_corpus, flags, cfg)
    out.to_parquet(out_path)
    logger.info("Saved character-network features to %s", out_path)
    return out


def main() -> None:
    paths.ensure_dirs()

    cn_df = _ensure_character_network_features()
    logger.info("Character-network feature matrix: %d films x %d cols", *cn_df.shape)

    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_with_targets = add_targets(df)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = (
        df_with_targets[df_with_targets["imdb_id"].isin(train_ids)]
        .reset_index(drop=True)
    )

    feature_cfg = BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
        include_lexical=False,
        include_sentiment=False,
        include_topic=False,
        include_character_network=True,
    )

    X = build_baseline_features(df_train, feature_cfg)
    feature_names = list(X.columns)

    train_cfg = BaselineTrainConfig()

    with save_run(
        phase="phase_3",
        name="character_network_multifamily",
        params={
            "model_families": list(MODEL_FAMILIES),
            "min_dialogue_lines_per_character": 5,
            "top_lead_decile_fraction": 0.10,
            "treat_flagged_films_as_nan": True,
            "n_cv_folds": train_cfg.n_cv_folds,
            "bootstrap_iter": train_cfg.bootstrap_iter,
            "bootstrap_alpha": train_cfg.bootstrap_alpha,
            "seed": train_cfg.seed,
        },
        preprocessing={
            "split": "70_15_15_seed_42",
            "stratification": "(primary_genre_bucketed, decade_bucket)",
            "structural_features_count": 7,
            "log_transform_structural": True,
            "include_log_runtime": True,
            "include_log_budget": False,
            "include_lexical": False,
            "include_sentiment": False,
            "include_topic": False,
            "include_character_network": True,
            "character_network_feature_count": 12,
            "graph_construction": (
                "scene-cooccurrence undirected graph; nodes = significant "
                "characters (>=5 non-empty dialogue lines); edges weighted "
                "by shared-scene count."
            ),
            "data_quality_flag_handling": (
                "NaN-fallback: 24 train-split flagged films receive NaN "
                "across all 12 model features. Trainer's median imputer "
                "handles them at fold-fit time."
            ),
        },
        features=feature_names,
        notes=(
            "Phase 3b ablation row 4 of 5. Character-network features (12) added "
            "on top of revised dialogue-only baseline. Pre-registered linear OOF: "
            "log_roi RMSE -0.040..-0.010; roi_gt_1 AUC 0.000..+0.015; "
            "roi_gt_2 AUC +0.020..+0.045. Strongest remaining genre-orthogonality candidate."
        ),
    ) as run:
        logger.info(
            "Starting character-network ablation evaluation across %d families",
            len(MODEL_FAMILIES),
        )
        rows = evaluate_feature_set(
            df_train, feature_cfg, train_cfg,
            set_name="dialogue_only_logged_character_network",
        )

        metrics: dict[str, float] = {}
        for r in rows:
            key = f"{r['model_family']}_{r['eval_set']}_{r['target']}_{r['metric']}"
            metrics[key] = round(float(r["value"]), 4)
            metrics[f"{key}_ci_lo"] = round(float(r["ci_lo"]), 4)
            metrics[f"{key}_ci_hi"] = round(float(r["ci_hi"]), 4)
        run.record_metrics(metrics)

        floor = _load_phase_3a_floor()
        ablation_rows = _build_ablation_rows(rows, floor)
        ablation_path = paths.REPORTS_TABLES_DIR / "phase3_ablation.csv"
        ablation_df = pd.DataFrame(ablation_rows)
        if ablation_path.is_file():
            existing = pd.read_csv(ablation_path)
            existing = existing[existing["feature_group"] != "character_network"]
            ablation_df = pd.concat([existing, ablation_df], ignore_index=True)
        ablation_df.to_csv(ablation_path, index=False)
        logger.info("Ablation table written: %d rows total", len(ablation_df))

        def _row(family: str, eval_set: str, target: str, metric: str):
            return next(
                r for r in rows
                if r["model_family"] == family
                and r["eval_set"] == eval_set
                and r["target"] == target
                and r["metric"] == metric
            )
        linear_log_roi_rmse = _row("linear", "oof", LOG_ROI_COL, "rmse")
        linear_roi_gt_2_auc = _row("linear", "oof", "roi_gt_2", "auc_roc")
        histgb_log_roi_rmse = _row("histgb", "oof", LOG_ROI_COL, "rmse")
        run.append_to_runs_md(
            model_family="4-family (linear, histgb, knn, svm)",
            features_group="structural + character_network",
            key_metric=(
                f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}"
            ),
            notes="Phase 3b row 4: character network (12 features); 4 families × 2 eval sets",
        )


def _build_ablation_rows(metric_rows: list[dict], floor: pd.DataFrame) -> list[dict]:
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
        predicted = CHARACTER_NETWORK_PREDICTED_LIFT.get((r["target"], r["metric"]))
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": "character_network",
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
