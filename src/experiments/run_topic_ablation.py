"""Phase 3b topic-feature ablation runner.

Fits LDA on the training-fold dialogue tokens (per the no-leakage
discipline in proposal v1 Section 6.2), produces the 22-column topic
feature matrix on the full corpus, saves it to
``data/processed/features_topic.parquet``, and runs the multi-family
baseline trainer with the topic-augmented matrix.

The ablation row is appended to ``reports/tables/phase3_ablation.csv``
and a row is added to ``runs/RUNS.md``. The fitted model artifacts
(vectorizer + LDA + train IDs) are saved to
``data/processed/topic_model_artifacts/`` so downstream phases can
reuse them.

The topic-augmented matrix is the **revised dialogue-only** baseline
plus the 22 topic features — NOT the lexical- or sentiment-augmented
matrix. This matches the standalone-lift methodology established by
the lexical and sentiment groups; joint evaluation belongs in Phase 3c.

Run from the project root::

    python3 -m src.experiments.run_topic_ablation
"""

from __future__ import annotations

# Allow running by file path; no-op under ``python -m``.
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
from src.features.targets import LOG_ROI_COL, add_targets
from src.features.topic import (
    TopicFeatureConfig,
    compute_topic_features,
    fit_topic_model,
    topic_label_table,
)
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


# Pre-registered lift bands from topic proposal v1 Section 4.
TOPIC_PREDICTED_LIFT: dict[tuple[str, str], tuple[float, float]] = {
    ("log_roi", "rmse"): (-0.030, -0.005),
    ("log_roi", "mae"): (-0.025, -0.005),
    ("log_roi", "cvrmse"): (-0.025, -0.005),
    ("roi_gt_1", "auc_roc"): (0.000, 0.015),
    ("roi_gt_1", "pr_auc"): (0.000, 0.015),
    ("roi_gt_1", "log_loss"): (-0.020, 0.000),
    ("roi_gt_2", "auc_roc"): (0.015, 0.040),
    ("roi_gt_2", "pr_auc"): (0.010, 0.030),
    ("roi_gt_2", "log_loss"): (-0.025, -0.005),
}


def _ensure_topic_features() -> tuple[pd.DataFrame, list[str]]:
    """Fit LDA on train fold and compute features on the full corpus.

    Cache is at ``data/processed/features_topic.parquet``. If the
    cache is missing or has the wrong row count the matrix is
    recomputed end-to-end. The fitted artifacts (vectorizer, LDA,
    train IDs) are saved to ``data/processed/topic_model_artifacts/``
    so downstream phases or notebooks can reload them via joblib.

    Returns the feature matrix and the list of train IDs used at fit
    time (for the diagnostic step's univariate-correlation check).
    """
    out_path = paths.DATA_PROCESSED_DIR / "features_topic.parquet"
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    splits_path = paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Parsed-screenplay pickle missing at {pkl_path}; run Phase 2 first"
        )
    if not splits_path.is_file():
        raise FileNotFoundError(
            f"Split assignments missing at {splits_path}; "
            "run `python3 -m src.features.split` first"
        )
    splits = pd.read_parquet(splits_path)
    train_ids = list(splits.loc[splits["split"] == "train", "imdb_id"])

    with pkl_path.open("rb") as f:
        parsed_corpus = pickle.load(f)

    if out_path.is_file():
        cached = pd.read_parquet(out_path)
        if len(cached) == len(parsed_corpus):
            logger.info(
                "Reusing cached topic features at %s (%d films)",
                out_path, len(cached),
            )
            return cached, train_ids

    cfg = TopicFeatureConfig()
    fitted = fit_topic_model(parsed_corpus, train_ids, cfg)
    df = compute_topic_features(parsed_corpus, fitted)
    df.to_parquet(out_path)
    logger.info("Saved topic features to %s", out_path)

    # Save fitted artifacts so downstream phases can reload without refitting.
    artifacts_dir = paths.DATA_PROCESSED_DIR / "topic_model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted.vectorizer, artifacts_dir / "vectorizer.joblib")
    joblib.dump(fitted.lda, artifacts_dir / "lda.joblib")
    np.save(artifacts_dir / "train_ids.npy", np.array(fitted.train_ids))
    label_table = topic_label_table(fitted, top_n_words=10)
    label_table.to_csv(
        paths.REPORTS_TABLES_DIR / "phase3_topic_labels.csv", index=False,
    )
    logger.info(
        "Saved fitted artifacts to %s and topic-label table to reports/tables/",
        artifacts_dir,
    )
    return df, train_ids


def main() -> None:
    """Compute topic features, run multi-family ablation, persist artifacts."""
    paths.ensure_dirs()

    # Step 1: compute or load topic features.
    topic_df, train_ids_from_split = _ensure_topic_features()
    logger.info("Topic feature matrix: %d films x %d cols", *topic_df.shape)

    # Step 2: load corpus + splits, attach targets, restrict to train.
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
        include_topic=True,
    )

    X = build_baseline_features(df_train, feature_cfg)
    feature_names = list(X.columns)

    train_cfg = BaselineTrainConfig()

    with save_run(
        phase="phase_3",
        name="topic_multifamily",
        params={
            "model_families": list(MODEL_FAMILIES),
            "n_topics": 20,
            "lda_min_df": 5,
            "lda_max_df": 0.5,
            "lda_max_iter": 10,
            "lda_learning_method": "batch",
            "lda_random_state": 42,
            "remove_stopwords": True,
            "min_token_length": 3,
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
            "include_topic": True,
            "topic_feature_count": 22,
            "lda_backend": "sklearn LatentDirichletAllocation (variational Bayes, batch)",
            "no_leakage_discipline": (
                "CountVectorizer and LDA fit on train_ids only; "
                "transform applied to all 1,713 films using train-fitted artifacts."
            ),
        },
        features=feature_names,
        notes=(
            "Phase 3b ablation row 3 of 5. Topic features (22) added on top of revised "
            "dialogue-only baseline. Pre-registered linear OOF: log_roi RMSE -0.030..-0.005; "
            "roi_gt_1 AUC 0.000..+0.015; roi_gt_2 AUC +0.015..+0.040. K=20 LDA via sklearn."
        ),
    ) as run:
        logger.info(
            "Starting topic ablation evaluation across %d families",
            len(MODEL_FAMILIES),
        )
        rows = evaluate_feature_set(
            df_train, feature_cfg, train_cfg,
            set_name="dialogue_only_logged_topic",
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
            existing = existing[existing["feature_group"] != "topic"]
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
            features_group="structural + topic",
            key_metric=(
                f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}"
            ),
            notes="Phase 3b row 3: topic (22 features, K=20 LDA); 4 families × 2 eval sets",
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
        predicted = TOPIC_PREDICTED_LIFT.get((r["target"], r["metric"]))
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": "topic",
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
