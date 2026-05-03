"""Phase 3b embedding ablation runner.

Two-stage pipeline:

1. Encode every non-empty dialogue line of every film with
   ``sentence-transformers/all-MiniLM-L6-v2``, mean-pool per film
   to a 384-dim vector, and cache to
   ``data/processed/embeddings_minilm_pooled.parquet``. This step
   takes 5 to 30 minutes depending on device (Apple Silicon MPS,
   CUDA, or CPU). Cached after the first run.
2. Fit PCA on the train-fold pooled embeddings only (no-leakage
   discipline), project all 1,713 films to 32 PCA components, and
   save to ``data/processed/features_embedding.parquet``.

Then runs the multi-family baseline trainer on the embedding-
augmented matrix and appends the ablation row to
``reports/tables/phase3_ablation.csv``.

Run from the project root::

    python3 -m src.experiments.run_embedding_ablation
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle

import joblib
import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_features
from src.features.embedding import (
    EmbeddingFeatureConfig,
    compute_embedding_features,
    extract_pooled_embeddings,
    fit_embedding_pca,
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


# Pre-registered lift bands from embedding proposal v1 Section 4.
EMBEDDING_PREDICTED_LIFT: dict[tuple[str, str], tuple[float, float]] = {
    ("log_roi", "rmse"): (-0.050, -0.015),
    ("log_roi", "mae"): (-0.040, -0.015),
    ("log_roi", "cvrmse"): (-0.040, -0.015),
    ("roi_gt_1", "auc_roc"): (0.000, 0.020),
    ("roi_gt_1", "pr_auc"): (0.000, 0.020),
    ("roi_gt_1", "f1"): (0.000, 0.010),
    ("roi_gt_1", "log_loss"): (-0.025, -0.005),
    ("roi_gt_2", "auc_roc"): (0.025, 0.060),
    ("roi_gt_2", "pr_auc"): (0.020, 0.045),
    ("roi_gt_2", "f1"): (0.005, 0.025),
    ("roi_gt_2", "log_loss"): (-0.030, -0.010),
}


def _ensure_embedding_features() -> tuple[pd.DataFrame, list[str], float]:
    """Run the two-stage pipeline and return (features, train_ids, cum_var).

    The features-cache short-circuit at
    ``data/processed/features_embedding.parquet`` is the fast path.
    The pooled-embeddings cache at
    ``data/processed/embeddings_minilm_pooled.parquet`` is the
    medium path. The slow path runs the encoder forward pass.
    """
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    splits_path = paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    features_out_path = paths.DATA_PROCESSED_DIR / "features_embedding.parquet"
    pca_artifact_path = paths.DATA_PROCESSED_DIR / "embedding_pca.joblib"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Parsed-screenplay pickle missing at {pkl_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Split assignments missing at {splits_path}")

    splits = pd.read_parquet(splits_path)
    train_ids = list(splits.loc[splits["split"] == "train", "imdb_id"])

    cfg = EmbeddingFeatureConfig()

    # Stage 1: pooled embeddings (cached).
    with pkl_path.open("rb") as f:
        parsed_corpus = pickle.load(f)
    pooled = extract_pooled_embeddings(parsed_corpus, cfg)

    # Stage 2: PCA fit + transform.
    fitted = fit_embedding_pca(pooled, train_ids, cfg)
    features = compute_embedding_features(pooled, fitted)
    features.to_parquet(features_out_path)
    joblib.dump(fitted.pca, pca_artifact_path)
    logger.info("Saved embedding features and fitted PCA")

    cum_var = float(fitted.pca.explained_variance_ratio_.cumsum()[-1])
    return features, train_ids, cum_var


def main() -> None:
    paths.ensure_dirs()

    emb_df, _train_ids, cum_var = _ensure_embedding_features()
    logger.info(
        "Embedding feature matrix: %d films x %d cols (cumulative variance %.3f)",
        emb_df.shape[0], emb_df.shape[1], cum_var,
    )

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
        include_character_network=False,
        include_embedding=True,
    )

    X = build_baseline_features(df_train, feature_cfg)
    feature_names = list(X.columns)

    train_cfg = BaselineTrainConfig()

    with save_run(
        phase="phase_3",
        name="embedding_multifamily",
        params={
            "model_families": list(MODEL_FAMILIES),
            "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
            "encoder_dim": 384,
            "n_pca_components": 32,
            "pca_random_state": 42,
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
            "include_character_network": False,
            "include_embedding": True,
            "embedding_feature_count": 32,
            "encoder": "sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych 2019; Wang et al. 2020)",
            "pooling": "per-line MiniLM forward pass, mean-pooled per film to 384-dim",
            "dimensionality_reduction": "PCA-32 fit on train fold only",
            "no_leakage_discipline": (
                "Pre-trained encoder applied as-is to all 1,713 films (no fitting). "
                "PCA fit on training-fold pooled embeddings only; transform applied "
                "uniformly to all films."
            ),
            "pca_cumulative_variance_explained": round(cum_var, 4),
        },
        features=feature_names,
        notes=(
            f"Phase 3b ablation row 5 of 5. Embedding features (32 PCA components of "
            f"per-line MiniLM mean-pool) added on top of revised dialogue-only baseline. "
            f"Pre-registered linear OOF: log_roi RMSE -0.050..-0.015; roi_gt_2 AUC "
            f"+0.025..+0.060. PCA cumulative variance explained: {cum_var:.3f}."
        ),
    ) as run:
        logger.info("Starting embedding ablation evaluation across %d families", len(MODEL_FAMILIES))
        rows = evaluate_feature_set(
            df_train, feature_cfg, train_cfg,
            set_name="dialogue_only_logged_embedding",
        )

        metrics: dict[str, float] = {}
        for r in rows:
            key = f"{r['model_family']}_{r['eval_set']}_{r['target']}_{r['metric']}"
            metrics[key] = round(float(r["value"]), 4)
            metrics[f"{key}_ci_lo"] = round(float(r["ci_lo"]), 4)
            metrics[f"{key}_ci_hi"] = round(float(r["ci_hi"]), 4)
        metrics["pca_cumulative_variance_explained"] = round(cum_var, 4)
        run.record_metrics(metrics)

        floor = _load_phase_3a_floor()
        ablation_rows = _build_ablation_rows(rows, floor)
        ablation_path = paths.REPORTS_TABLES_DIR / "phase3_ablation.csv"
        ablation_df = pd.DataFrame(ablation_rows)
        if ablation_path.is_file():
            existing = pd.read_csv(ablation_path)
            existing = existing[existing["feature_group"] != "embedding"]
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
            features_group="structural + embedding",
            key_metric=(
                f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}; "
                f"PCA-32 cumvar {cum_var:.3f}"
            ),
            notes=(
                "Phase 3b row 5: embedding (32 PCA of MiniLM pooled); "
                "4 families × 2 eval sets"
            ),
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
        predicted = EMBEDDING_PREDICTED_LIFT.get((r["target"], r["metric"]))
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": "embedding",
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
