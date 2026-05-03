"""Phase 3b sentiment-feature ablation runner.

Computes the 22-column sentiment feature matrix on the full corpus,
saves it to ``data/processed/features_sentiment.parquet``, and runs
the multi-family baseline trainer with the sentiment-augmented
matrix. The ablation row is appended to
``reports/tables/phase3_ablation.csv`` and a row is added to
``runs/RUNS.md``.

The sentiment-augmented matrix is the **revised dialogue-only**
baseline (structural counts, era, genre dummies, log_runtime) plus
the 22 sentiment model features — NOT the lexical-augmented matrix.
This matches the lexical group's ablation pattern: each Phase 3b
group's standalone-lift row compares against the same Phase 3a
floor. Joint evaluation of multiple groups belongs in Phase 3c.

Run from the project root::

    python3 -m src.experiments.run_sentiment_ablation

The output appends to (or creates) ``reports/tables/phase3_ablation.csv``
with one row per (model_family, eval_set, target, metric) combination
carrying the predicted-vs-actual lift over the corresponding family's
Phase 3a revised dialogue-only floor.
"""

from __future__ import annotations

# Allow running by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle

import numpy as np
import pandas as pd

from src.experiments.save_run import save_run
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_features
from src.features.sentiment import (
    SentimentFeatureConfig,
    compute_sentiment_features,
)
from src.features.targets import LOG_ROI_COL, add_targets
from src.models.baseline.train import (
    BaselineTrainConfig, MODEL_FAMILIES, evaluate_feature_set,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


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


# Pre-registered lift bands from sentiment proposal v2 Section 4. The
# regression bands were originally posted as RMSE reductions (lower is
# better), so the lower bound is more negative than the upper. The
# classification bands are AUC-ROC lifts (higher is better). All bands
# apply to the linear family's OOF numbers (the historical reference
# for proposal-side pre-registration).
SENTIMENT_PREDICTED_LIFT: dict[tuple[str, str], tuple[float, float]] = {
    ("log_roi", "rmse"): (-0.030, -0.010),
    ("log_roi", "mae"): (-0.030, -0.010),
    ("log_roi", "cvrmse"): (-0.025, -0.010),
    ("roi_gt_1", "auc_roc"): (0.000, 0.010),
    ("roi_gt_1", "log_loss"): (-0.020, 0.000),
    ("roi_gt_2", "auc_roc"): (0.015, 0.030),
    ("roi_gt_2", "pr_auc"): (0.010, 0.025),
    ("roi_gt_2", "log_loss"): (-0.020, -0.005),
}


def _ensure_sentiment_features() -> pd.DataFrame:
    """Compute the sentiment feature matrix on the full corpus, or load if cached.

    The matrix is cached at
    ``data/processed/features_sentiment.parquet``. If the cache is
    missing or has fewer rows than the corpus pickle, the matrix is
    recomputed end-to-end from the parsed-screenplay pickle.
    """
    out_path = paths.DATA_PROCESSED_DIR / "features_sentiment.parquet"
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Parsed-screenplay pickle missing at {pkl_path}; run Phase 2 first"
        )
    with pkl_path.open("rb") as f:
        parsed_corpus = pickle.load(f)

    if out_path.is_file():
        cached = pd.read_parquet(out_path)
        if len(cached) == len(parsed_corpus):
            logger.info(
                "Reusing cached sentiment features at %s (%d films)",
                out_path, len(cached),
            )
            return cached
        logger.info(
            "Cache size mismatch (%d cached vs %d in corpus); recomputing",
            len(cached), len(parsed_corpus),
        )

    cfg = SentimentFeatureConfig()
    df = compute_sentiment_features(parsed_corpus, cfg)
    df.to_parquet(out_path)
    logger.info("Saved sentiment features to %s", out_path)
    return df


def main() -> None:
    """Compute sentiment features, run multi-family ablation, persist artifacts."""
    paths.ensure_dirs()

    # Step 1: compute or load the sentiment feature matrix.
    sent_df = _ensure_sentiment_features()
    logger.info("Sentiment matrix: %d films × %d cols", *sent_df.shape)

    # Step 2: load corpus + splits, attach targets, restrict to training films.
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_with_targets = add_targets(df)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = (
        df_with_targets[df_with_targets["imdb_id"].isin(train_ids)]
        .reset_index(drop=True)
    )

    # Sentiment-augmented config (revised dialogue-only baseline +
    # sentiment, NOT lexical). Standalone-lift methodology.
    feature_cfg = BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
        include_lexical=False,
        include_sentiment=True,
    )

    # Construct the augmented feature matrix once so we can capture the
    # exact column list for ``features_used.json`` before training.
    X = build_baseline_features(df_train, feature_cfg)
    feature_names = list(X.columns)

    train_cfg = BaselineTrainConfig()

    with save_run(
        phase="phase_3",
        name="sentiment_multifamily",
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
            "sentiment_n_quartile_windows": 4,
            "sentiment_arc_template_length": 100,
            "sentiment_remove_stopwords_for_nrc": True,
            "sentiment_archetype_set": "reagan_six",
        },
        preprocessing={
            "split": "70_15_15_seed_42",
            "stratification": "(primary_genre_bucketed, decade_bucket)",
            "rare_cell_threshold": 5,
            "structural_features_count": 7,
            "log_transform_structural": True,
            "include_log_runtime": True,
            "include_log_budget": False,
            "include_lexical": False,
            "include_sentiment": True,
            "sentiment_feature_count": 22,
            "linear_imputer": "SimpleImputer(strategy='median')",
            "linear_scaler": "StandardScaler",
            "histgb_preprocessing": "passthrough (native NaN handling, no scaling)",
            "knn_preprocessing": "SimpleImputer(median) + StandardScaler",
            "svm_preprocessing": "SimpleImputer(median) + StandardScaler",
            "vader_source": "nltk vader_lexicon (Hutto & Gilbert 2014)",
            "nrc_source": "nrclex package (bundled NRC-EmoLex; Mohammad & Turney 2013)",
            "nrc_source_deviation_note": (
                "Proposal v2 specified canonical NRC EmoLex direct download from "
                "saifmohammad.com. The canonical distribution is form-gated (manual "
                "form submission required to receive the zip), which blocks an "
                "automated hash-checked download into data/external/. The nrclex "
                "package ships the same word-emotion mappings under the author's "
                "research-use license and was the documented fallback in proposal "
                "v2 Section 6.2. Surfaced for planning-conversation review; matches "
                "the wordfreq-vs-SUBTLEX-US precedent set in the lexical group."
            ),
            "stopword_removal_for_nrc": "nltk english stopwords (~180 function words)",
            "arc_archetype_templates": "hand-coded smoothed mathematical shapes (z-normalized)",
        },
        features=feature_names,
        notes=(
            "Phase 3b ablation row 2 of 5. Sentiment features (22) added on top of "
            "revised dialogue-only baseline, evaluated across 4 model families. "
            "Pre-registered linear OOF bands: log_roi RMSE -0.030..-0.010 (lower is "
            "better); roi_gt_1 AUC 0.000..+0.010; roi_gt_2 AUC +0.015..+0.030. "
            "NRC source deviation: nrclex package (form-gated canonical download)."
        ),
    ) as run:
        logger.info(
            "Starting sentiment ablation evaluation across %d families",
            len(MODEL_FAMILIES),
        )
        rows = evaluate_feature_set(
            df_train, feature_cfg, train_cfg,
            set_name="dialogue_only_logged_sentiment",
        )

        # Convert metric rows into the metrics dict for save_run.
        metrics: dict[str, float] = {}
        for r in rows:
            key = f"{r['model_family']}_{r['eval_set']}_{r['target']}_{r['metric']}"
            metrics[key] = round(float(r["value"]), 4)
            metrics[f"{key}_ci_lo"] = round(float(r["ci_lo"]), 4)
            metrics[f"{key}_ci_hi"] = round(float(r["ci_hi"]), 4)
        run.record_metrics(metrics)

        # Build the ablation table with per-family predicted vs actual lift.
        floor = _load_phase_3a_floor()
        ablation_rows = _build_ablation_rows(rows, floor)
        ablation_path = paths.REPORTS_TABLES_DIR / "phase3_ablation.csv"
        ablation_df = pd.DataFrame(ablation_rows)
        if ablation_path.is_file():
            existing = pd.read_csv(ablation_path)
            existing = existing[existing["feature_group"] != "sentiment"]
            ablation_df = pd.concat([existing, ablation_df], ignore_index=True)
        ablation_df.to_csv(ablation_path, index=False)
        logger.info("Ablation table written: %d rows total", len(ablation_df))

        # Summary lines for RUNS.md: linear OOF RMSE on log_roi and AUC
        # on roi_gt_2 are the report-facing summary numbers; histgb
        # RMSE provides the strongest-floor cross-check.
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
            features_group="structural + sentiment",
            key_metric=(
                f"linear OOF RMSE {linear_log_roi_rmse['value']:.3f} / "
                f"AUC roi_gt_2 {linear_roi_gt_2_auc['value']:.3f}; "
                f"histgb RMSE {histgb_log_roi_rmse['value']:.3f}"
            ),
            notes="Phase 3b row 2: sentiment (22 features); 4 families × 2 eval sets",
        )


def _build_ablation_rows(metric_rows: list[dict], floor: pd.DataFrame) -> list[dict]:
    """Per-(family, eval_set, metric) ablation rows with predicted-vs-actual lift.

    Mirrors the lexical runner's ablation-row schema. Pre-registered
    bands apply only to the linear-family OOF numbers (the proposal's
    historical reference); other families' lifts are reported but
    ``in_predicted_band`` is left blank.
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
        predicted = SENTIMENT_PREDICTED_LIFT.get((r["target"], r["metric"]))
        in_band: bool | None
        if predicted is None or np.isnan(lift):
            in_band = None
        elif r["model_family"] == "linear" and r["eval_set"] == "oof":
            in_band = predicted[0] <= lift <= predicted[1]
        else:
            in_band = None
        out.append({
            "feature_group": "sentiment",
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
