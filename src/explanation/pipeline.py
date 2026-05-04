"""Phase 7 orchestrator: end-to-end SHAP attribution + scene-level + report.

Runs the locked methodology from ``phase7_preregistration.md``:

1. Build TreeExplainer for XGBoost (roi_gt_2) and RandomForest (log_roi).
2. Compute global SHAP rankings + Spearman correlation vs Phase 4
   native importance.
3. Stability validation: re-run SHAP with ``interventional`` mode
   and check rank correlation.
4. Per-film attributions for the 257 calibration films on the
   headline target; merge with Phase 6 decisions.
5. Scene-level counterfactual attribution on 5 representative
   example films.
6. Persist explainer artifacts + per-film attributions + scene-level
   JSON.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.parse_screenplay import ParsedScreenplay
from src.experiments.save_run import save_run
from src.explanation.global_importance import (
    compare_to_native,
    global_shap_ranking,
    spearman_rank_correlation,
)
from src.explanation.per_film import (
    merge_with_phase6,
    per_film_attributions,
)
from src.explanation.scene_level import (
    SceneContribution,
    per_scene_contributions,
    select_example_films,
)
from src.explanation.shap_explainer import (
    TreeExplainerBundle,
    build_tree_explainer,
    shap_values,
)
from src.features.targets import add_targets
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


HEADLINE_TARGET: str = "roi_gt_2"
SECONDARY_REGRESSION_TARGET: str = "log_roi"


def _load_cal_features(matrix_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_cal, df_cal_meta) keyed by imdb_id."""
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    cal_ids = splits.loc[splits["split"] == "cal", "imdb_id"].tolist()
    df_cal = df_full[df_full["imdb_id"].isin(cal_ids)].reset_index(drop=True)
    matrix_spec = MATRICES[matrix_name]
    X_cal = build_matrix(matrix_spec, df_cal)
    return X_cal, df_cal


def _global_attribution_for_target(
    target: str, family: str,
) -> dict[str, Any]:
    """Run TreeSHAP global ranking for one target."""
    bundle_path = paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{target}.joblib"
    bundle = joblib.load(bundle_path)
    pipeline = bundle["estimator"]
    matrix_name = bundle["matrix"]
    X_cal, df_cal = _load_cal_features(matrix_name)
    feature_names = list(X_cal.columns)

    logger.info("TreeSHAP %s | family=%s | computing global ranking on %d cal films",
                target, family, len(X_cal))
    explainer = build_tree_explainer(
        pipeline, family=family, target=target,
        X_for_background=X_cal,
        feature_perturbation="tree_path_dependent",
    )
    sv_path_dependent = shap_values(explainer, pipeline, X_cal)
    ranking = global_shap_ranking(sv_path_dependent, feature_names)
    comparison = compare_to_native(target, ranking)
    rho_native = spearman_rank_correlation(comparison)
    logger.info("SHAP-vs-native Spearman ρ for %s = %.3f", target, rho_native)

    # Stability check: rerun with interventional mode
    logger.info("TreeSHAP %s | stability check via interventional mode", target)
    explainer_interv = build_tree_explainer(
        pipeline, family=family, target=target,
        X_for_background=X_cal,
        feature_perturbation="interventional",
    )
    sv_interv = shap_values(explainer_interv, pipeline, X_cal)
    ranking_interv = global_shap_ranking(sv_interv, feature_names)
    # Spearman between path_dependent vs interventional rankings
    merged = ranking.merge(
        ranking_interv[["feature", "rank"]].rename(columns={"rank": "interv_rank"}),
        on="feature", how="inner",
    )
    from scipy.stats import spearmanr
    rho_stab, _ = spearmanr(merged["rank"], merged["interv_rank"])
    logger.info("SHAP stability (path_dep vs interv) Spearman ρ = %.3f", rho_stab)

    return {
        "target": target,
        "family": family,
        "explainer_bundle": explainer,
        "shap_values": sv_path_dependent,
        "global_ranking": ranking,
        "comparison_to_native": comparison,
        "rho_vs_native": rho_native,
        "rho_stability": float(rho_stab),
        "X_cal": X_cal,
        "df_cal": df_cal,
        "feature_names": feature_names,
        "matrix_name": matrix_name,
        "pipeline": pipeline,
    }


def _per_film_for_headline(global_result: dict[str, Any]) -> pd.DataFrame:
    """Per-film attribution + Phase 6 decision merge for the headline target."""
    sv = global_result["shap_values"]
    feature_names = global_result["feature_names"]
    imdb_ids = global_result["df_cal"]["imdb_id"].astype(str).tolist()
    base_value = global_result["explainer_bundle"].base_value
    per_film = per_film_attributions(imdb_ids, sv, feature_names, base_value)
    return merge_with_phase6(per_film)


def _scene_level_for_examples(
    headline_target: str,
    per_film_df: pd.DataFrame,
    n_films: int = 5,
) -> tuple[list[dict], pd.DataFrame]:
    """Run scene-removal counterfactual on the selected example films."""
    logger.info("Selecting %d example films for scene-level attribution", n_films)
    if "recommended_action" not in per_film_df.columns:
        logger.warning("per_film_df missing Phase 6 columns; cannot select examples")
        return [], pd.DataFrame()
    example_ids = select_example_films(per_film_df, k_per_category=1)
    example_ids = example_ids[:n_films]
    logger.info("Selected example films: %s", example_ids)

    # Load calibrated wrapper for predict_proba
    p5_bundle = joblib.load(
        paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{headline_target}.joblib"
    )
    cal_wrapper = p5_bundle["probability_calibrator"]
    if cal_wrapper is None:
        # Fall back to phase 4 estimator
        p4 = joblib.load(p5_bundle["phase4_winner_path"])
        cal_wrapper = p4["estimator"]

    # Load parsed screenplays + raw mpnet pooled embeddings
    with open(paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl", "rb") as f:
        parsed_corpus = pickle.load(f)

    # The mpnet embeddings cache is per-film-mean (1 row per film, 768 dim).
    # Per-scene mpnet would require re-encoding; we approximate scene-removal's
    # embedding effect by retreating to per-film mean (no per-scene update).
    # The structural and ratio updates are exact; embedding stays approximate.
    feature_columns = list(per_film_df.columns)

    # Pull base features from the cal-set DataFrame
    matrix_name = p5_bundle["matrix"]
    X_cal, _ = _load_cal_features(matrix_name)
    X_cal_indexed = X_cal.copy()
    X_cal_indexed.index = per_film_df["imdb_id"].astype(str).tolist()[: len(X_cal_indexed)]

    examples_payload: list[dict] = []
    rows_table: list[dict] = []
    for iid in example_ids:
        if iid not in parsed_corpus:
            logger.warning("Skipping %s: parsed screenplay missing", iid)
            continue
        if iid not in X_cal_indexed.index:
            logger.warning("Skipping %s: not in calibration set X", iid)
            continue
        parsed = parsed_corpus[iid]
        base_features = X_cal_indexed.loc[iid]
        logger.info(
            "Scene-level attribution: %s (%d scenes)",
            iid, len(parsed.scenes),
        )
        contributions = per_scene_contributions(
            iid, parsed, base_features, cal_wrapper,
            embeddings_pooled_per_scene=None,
        )
        # Sort by absolute contribution
        contributions_sorted = sorted(contributions, key=lambda c: -abs(c.contribution))
        top_pos = sorted(
            [c for c in contributions if c.contribution > 0],
            key=lambda c: -c.contribution,
        )[:3]
        top_neg = sorted(
            [c for c in contributions if c.contribution < 0],
            key=lambda c: c.contribution,
        )[:3]

        film_meta = per_film_df[per_film_df["imdb_id"] == iid].iloc[0].to_dict()
        examples_payload.append({
            "imdb_id": iid,
            "movie_name": str(film_meta.get("imdb_id", iid)),
            "calibrated_probability": float(film_meta.get("calibrated_probability", float("nan"))),
            "recommended_action": str(film_meta.get("recommended_action", "n/a")),
            "n_scenes": len(parsed.scenes),
            "top_positive_scenes": [
                {
                    "scene_index": c.scene_index,
                    "heading": c.scene_heading,
                    "contribution": c.contribution,
                    "n_dialogue_in_scene": c.n_dialogue_in_scene,
                }
                for c in top_pos
            ],
            "top_negative_scenes": [
                {
                    "scene_index": c.scene_index,
                    "heading": c.scene_heading,
                    "contribution": c.contribution,
                    "n_dialogue_in_scene": c.n_dialogue_in_scene,
                }
                for c in top_neg
            ],
        })
        for c in contributions:
            rows_table.append({
                "imdb_id": iid,
                "scene_index": c.scene_index,
                "scene_heading": c.scene_heading,
                "contribution": c.contribution,
                "n_dialogue_in_scene": c.n_dialogue_in_scene,
            })

    return examples_payload, pd.DataFrame(rows_table)


def run_phase7() -> dict[str, Any]:
    """End-to-end Phase 7."""
    paths.ensure_dirs()
    out: dict[str, Any] = {}

    with save_run(
        phase="phase_7",
        name="phase7_attribution",
        params={
            "primary_target": HEADLINE_TARGET,
            "secondary_regression_target": SECONDARY_REGRESSION_TARGET,
            "scene_level_n_examples": 5,
            "stability_method": "tree_path_dependent_vs_interventional",
        },
        preprocessing={
            "evaluation_set": "calibration",
            "test_set": "untouched_until_phase_8",
        },
        features=[],
    ) as run:
        # 1. Headline (XGBoost on roi_gt_2)
        headline = _global_attribution_for_target(HEADLINE_TARGET, family="xgboost")
        out[HEADLINE_TARGET] = headline

        # 2. Secondary (RF on log_roi)
        try:
            secondary = _global_attribution_for_target(SECONDARY_REGRESSION_TARGET, family="random_forest")
            out[SECONDARY_REGRESSION_TARGET] = secondary
        except Exception as exc:
            logger.warning("Skipping secondary target: %s", exc)
            secondary = None

        # 3. Per-film for headline + merge with Phase 6
        per_film = _per_film_for_headline(headline)
        out["per_film_headline"] = per_film

        # 4. Scene-level on example films
        scene_payload, scene_table = _scene_level_for_examples(
            HEADLINE_TARGET, per_film, n_films=5,
        )
        out["scene_level_examples"] = scene_payload
        out["scene_level_table"] = scene_table

        # ---- Persist tables ----
        headline["global_ranking"].to_csv(
            paths.REPORTS_TABLES_DIR / f"phase7_global_shap_{HEADLINE_TARGET}.csv",
            index=False,
        )
        if secondary:
            secondary["global_ranking"].to_csv(
                paths.REPORTS_TABLES_DIR / f"phase7_global_shap_{SECONDARY_REGRESSION_TARGET}.csv",
                index=False,
            )

        headline["comparison_to_native"].to_csv(
            paths.REPORTS_TABLES_DIR / f"phase7_shap_vs_native_importance.csv",
            index=False,
        )

        per_film.to_csv(
            paths.REPORTS_TABLES_DIR / "phase7_per_film_rationale.csv",
            index=False,
        )

        if not scene_table.empty:
            scene_table.to_csv(
                paths.REPORTS_TABLES_DIR / "phase7_scene_level.csv",
                index=False,
            )

        stability_rows = [{
            "target": HEADLINE_TARGET,
            "rho_path_dependent_vs_interventional": headline["rho_stability"],
            "rho_shap_vs_native": headline["rho_vs_native"],
        }]
        if secondary:
            stability_rows.append({
                "target": SECONDARY_REGRESSION_TARGET,
                "rho_path_dependent_vs_interventional": secondary["rho_stability"],
                "rho_shap_vs_native": secondary["rho_vs_native"],
            })
        pd.DataFrame(stability_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase7_stability.csv",
            index=False,
        )

        # ---- Persist explainer + scene-level JSON ----
        explainer_bundle = {
            "target": HEADLINE_TARGET,
            "family": "xgboost",
            "feature_names": headline["feature_names"],
            "explainer": headline["explainer_bundle"].explainer,
            "base_value": headline["explainer_bundle"].base_value,
            "matrix_name": headline["matrix_name"],
            "phase4_winner_path": str(
                paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{HEADLINE_TARGET}.joblib"
            ),
        }
        joblib.dump(
            explainer_bundle,
            paths.DATA_PROCESSED_DIR / f"phase7_shap_explainer_{HEADLINE_TARGET}.joblib",
        )
        per_film.to_parquet(
            paths.DATA_PROCESSED_DIR / f"phase7_per_film_attribution_{HEADLINE_TARGET}.parquet",
            index=False,
        )
        if scene_payload:
            with open(paths.DATA_PROCESSED_DIR / "phase7_scene_level_examples.json", "w") as f:
                json.dump(scene_payload, f, indent=2, default=str)

        # ---- save_run metrics ----
        run.record_metrics({
            "headline_rho_vs_native": headline["rho_vs_native"],
            "headline_rho_stability": headline["rho_stability"],
            "n_per_film_attributions": int(len(per_film)),
            "n_scene_level_examples": int(len(scene_payload)),
            "top10_global_features_headline": (
                headline["global_ranking"].head(10)["feature"].tolist()
            ),
        })
        run.append_to_runs_md(
            model_family="phase7_treeshap_xgboost+rf",
            features_group=HEADLINE_TARGET,
            key_metric=(
                f"rho_vs_native={headline['rho_vs_native']:.3f}; "
                f"rho_stab={headline['rho_stability']:.3f}; "
                f"n_per_film={len(per_film)}; "
                f"n_scene_examples={len(scene_payload)}"
            ),
            notes="Phase 7 SHAP attribution",
        )

    return out
