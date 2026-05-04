"""Phase 8 entry point — end-to-end integration + final test-set evaluation.

Executes the locked methodology in
``docs/proposals/phase8_preregistration.md``:

1. Verify test-set isolation (programmatic check).
2. Load all four-layer artifacts (Phase 4 / 5 / 6 / 7).
3. Build the test-set feature matrix (257 films, 92 columns,
   ``standalone_positive_union_mpnet``).
4. Run the end-to-end pipeline on every test film; persist the
   per-film triage table.
5. Compute Layer 1 / 2 / 3 / 4 test-set metrics with bootstrap CIs.
6. Run the four pre-registered error-analysis cuts (genre, decade,
   budget tier, length tier) plus the most-correct / most-wrong
   galleries.
7. Curate and render the five-film example gallery.
8. Generate the six pre-registered figures.
9. Check the five escalation triggers and surface their status.
10. save_run + RUNS.md update.

Run::

    python -m src.experiments.run_phase8_evaluation
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# sklearn 1.8 deprecation noise.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.decision.cost_matrix import DEFAULT_COST_MATRIX
from src.evaluation.error_analysis import (
    assign_budget_tier,
    assign_decade_bucket,
    assign_length_tier,
    cut_metrics,
    gallery,
)
from src.evaluation.example_outputs import (
    render_gallery_markdown,
    select_examples,
)
from src.evaluation.figures import (
    plot_calibration_test,
    plot_coverage_test,
    plot_decision_costs_test,
    plot_decision_sensitivity_test,
    plot_per_genre_metrics_test,
    plot_top_shap_test,
)
from src.evaluation.pipeline import (
    HEADLINE_TARGET,
    assert_test_set_isolation,
    build_test_feature_matrix,
    load_four_layer_artifacts,
    run_batch,
    smoke_test_consistency,
)
from src.evaluation.test_eval import (
    DEFAULT_CONFIDENCE_LEVELS,
    classification_calibration_metrics,
    classification_metrics_with_ci,
    conformal_classification_coverage,
    conformal_regression_coverage,
    decision_evaluation,
    refer_cost_sweep,
    regression_metrics_with_ci,
    reliability_table,
    shap_test_global_ranking,
    shap_test_vs_cal_overlap,
    shap_vs_native_test,
)
from src.experiments.save_run import save_run
from src.features.targets import add_targets
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


CLASSIFICATION_TARGETS: tuple[str, ...] = ("roi_gt_1", "roi_gt_2")


# ---------------------------------------------------------------------------
# Subroutines
# ---------------------------------------------------------------------------


def _layer1_metrics(
    df_test: pd.DataFrame, per_film: pd.DataFrame,
) -> dict[str, dict]:
    """Predictive performance on the un-calibrated layer.

    Regression on log_roi uses the Phase 4 RandomForest's ``predict``;
    classification uses the un-calibrated ``predict_proba`` (or the
    Phase 5 calibrator for SVM-RBF roi_gt_1, where uncalibrated is
    not available). The pre-reg locks Layer 1 = uncalibrated; we
    note the SVM-RBF substitution where it occurs.
    """
    out: dict[str, dict] = {}
    # log_roi regression
    y_true = df_test["log_roi"].values
    y_pred = per_film["log_roi_point_prediction"].values
    out["log_roi"] = regression_metrics_with_ci(y_true, y_pred)

    # roi_gt_1: SVM-RBF without native predict_proba; fall back to
    # calibrated probability for AUC etc.
    y_true = df_test["roi_gt_1"].astype(int).values
    y_score_uncal = per_film["roi_gt_1_uncalibrated_probability"].values
    if np.isnan(y_score_uncal).all():
        # SVM-RBF case — substitute calibrated probability and note it.
        y_score_uncal = per_film["calibrated_probability_roi_gt_1"].values
    out["roi_gt_1"] = classification_metrics_with_ci(y_true, y_score_uncal)

    # roi_gt_2: XGBoost has predict_proba.
    y_true = df_test["roi_gt_2"].astype(int).values
    y_score_uncal = per_film["roi_gt_2_uncalibrated_probability"].values
    if np.isnan(y_score_uncal).all():
        y_score_uncal = per_film["calibrated_probability_roi_gt_2"].values
    out["roi_gt_2"] = classification_metrics_with_ci(y_true, y_score_uncal)

    return out


def _layer2_metrics(
    df_test: pd.DataFrame, per_film: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, pd.DataFrame], list[dict], dict[str, float]]:
    """Calibration + coverage. Returns
    (cal_metrics_per_target, reliability_tables, coverage_rows, singleton_at_0p9).
    """
    cal_metrics: dict[str, dict] = {}
    reliability_tables: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict] = []
    singleton_at_0p9: dict[str, float] = {}

    for target in CLASSIFICATION_TARGETS:
        y_true = df_test[target].astype(int).values
        prob_col = f"calibrated_probability_{target}"
        y_prob = per_film[prob_col].values

        cal_metrics[target] = classification_calibration_metrics(y_true, y_prob)
        reliability_tables[target] = reliability_table(y_true, y_prob, n_bins=10)

        # Conformal coverage from the Phase 8 per-film flags
        if target == "roi_gt_2":
            for level in DEFAULT_CONFIDENCE_LEVELS:
                size_col = f"conf_roi_gt_2_set_size_{level}"
                in_set_col_pos = f"conf_roi_gt_2_class1_in_set_{level}"
                in_set_col_neg = f"conf_roi_gt_2_class0_in_set_{level}"
                if size_col not in per_film.columns:
                    continue
                set_size = per_film[size_col].values
                # Empirical "true class is in the predicted set"
                in_set_true = np.where(
                    y_true == 1,
                    per_film[in_set_col_pos].astype(int).values,
                    per_film[in_set_col_neg].astype(int).values,
                ).astype(bool)
                cov = conformal_classification_coverage(
                    y_true, set_size, in_set_true, level=level,
                )
                coverage_rows.append({"target": target, **cov})
                if abs(level - 0.9) < 1e-9:
                    singleton_at_0p9[target] = cov["singleton_rate"]

    # log_roi conformal coverage
    y_true = df_test["log_roi"].values
    for level in DEFAULT_CONFIDENCE_LEVELS:
        lo_col = f"log_roi_lower_{level}"
        hi_col = f"log_roi_upper_{level}"
        if lo_col not in per_film.columns:
            continue
        lower = per_film[lo_col].values
        upper = per_film[hi_col].values
        cov = conformal_regression_coverage(y_true, lower, upper, level=level)
        coverage_rows.append({"target": "log_roi", **cov})

    return cal_metrics, reliability_tables, coverage_rows, singleton_at_0p9


def _layer3_metrics(
    df_test: pd.DataFrame, per_film: pd.DataFrame,
) -> tuple[dict, list[dict], pd.DataFrame]:
    """Decision evaluation: system + 5 baselines + refer-cost sweep."""
    imdb_ids = per_film["imdb_id"].astype(str).tolist()
    probs = per_film["calibrated_probability_roi_gt_2"].values
    true_labels = df_test["roi_gt_2"].astype(int).values
    genres = df_test["primary_genre_bucketed"].astype(str).tolist()

    sys_row, baseline_rows = decision_evaluation(
        imdb_ids, probs, true_labels, genres, DEFAULT_COST_MATRIX,
    )
    sweep_df = refer_cost_sweep(imdb_ids, probs, true_labels)
    return sys_row, baseline_rows, sweep_df


def _layer4_metrics(
    artifacts, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, np.ndarray]:
    """Test-set TreeSHAP + comparisons.

    Returns (test_ranking, comparison_to_native, overlap_df,
    rho_vs_native, jaccard_overlap, raw_shap_array).
    """
    bundle = artifacts.phase7_explainer_roi_gt_2
    pipeline = artifacts.phase4_roi_gt_2["estimator"]
    test_ranking, raw_shap = shap_test_global_ranking(bundle, pipeline, X_test)
    comparison, rho = shap_vs_native_test(test_ranking, target=HEADLINE_TARGET)
    jaccard, overlap_df = shap_test_vs_cal_overlap(test_ranking, target=HEADLINE_TARGET)
    return test_ranking, comparison, overlap_df, rho, jaccard, raw_shap


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_phase8(verbose: bool = True) -> dict[str, Any]:
    """End-to-end Phase 8 orchestrator.

    Persists every deliverable and returns a summary dict for the
    runner / smoke-test harnesses.
    """
    if verbose:
        set_log_level("INFO")
    paths.ensure_dirs()

    # 0. Test-set isolation check.
    assert_test_set_isolation()

    # 1. Build the test-set feature matrix and load all artifacts.
    logger.info("Phase 8: building test-set feature matrix")
    X_test, df_test = build_test_feature_matrix()
    logger.info("Test set: %d films, %d features", len(X_test), X_test.shape[1])

    artifacts = load_four_layer_artifacts(X_test)

    with save_run(
        phase="phase_8",
        name="phase8_evaluation",
        params={
            "matrix_name": "standalone_positive_union_mpnet",
            "headline_target": HEADLINE_TARGET,
            "confidence_levels": DEFAULT_CONFIDENCE_LEVELS,
            "n_test_films": int(len(X_test)),
            "n_features": int(X_test.shape[1]),
            "default_cost_matrix": {
                "greenlight_flop": DEFAULT_COST_MATRIX.cost_greenlight_flop,
                "pass_hit": DEFAULT_COST_MATRIX.cost_pass_hit,
                "refer": DEFAULT_COST_MATRIX.cost_refer_flop,
            },
        },
        preprocessing={
            "phase4_winners": "all on standalone_positive_union_mpnet",
            "phase5_calibrators": "isotonic + LAC for roi_gt_*; absolute for log_roi",
            "phase6_decision_rule": "expected-cost minimization, tie-break to Refer",
            "phase7_explainer": "TreeSHAP path-dependent on XGBoost roi_gt_2",
            "test_set_touched": True,
        },
        features=list(X_test.columns),
    ) as run:

        # 2. Smoke-test consistency on 3 films.
        smoke = smoke_test_consistency(artifacts, n_films=3)
        logger.info("Phase 8 smoke test: %s (%d films, %d mismatches)",
                    "PASS" if smoke["passed"] else "FAIL",
                    smoke["n_films_checked"],
                    len(smoke["mismatches"]))

        # 3. Run the end-to-end pipeline on the test set.
        per_film = run_batch(artifacts)

        # Join with test-set ground truth and metadata.
        meta_cols = [
            "imdb_id", "primary_genre_bucketed", "release_year_parsed",
            "budget", "revenue", "n_scenes",
            "log_roi", "roi_gt_1", "roi_gt_2",
        ]
        per_film_full = per_film.merge(
            df_test[meta_cols].astype({"imdb_id": str}),
            on="imdb_id", how="left",
        )
        per_film_full["true_label"] = per_film_full["roi_gt_2"].astype(int)
        per_film_full["decade_bucket"] = per_film_full["release_year_parsed"].apply(
            assign_decade_bucket
        )
        per_film_full["budget_tier"] = per_film_full["budget"].apply(assign_budget_tier)
        per_film_full["length_tier"] = per_film_full["n_scenes"].apply(assign_length_tier)

        per_film_full.to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_per_film_outputs.csv", index=False,
        )
        logger.info("Wrote phase8_per_film_outputs.csv (%d rows)", len(per_film_full))

        # 4. Layer 1 metrics
        layer1 = _layer1_metrics(df_test, per_film)
        layer1_rows = []
        for target, metrics in layer1.items():
            for metric_name, vals in metrics.items():
                layer1_rows.append({
                    "target": target,
                    "metric": metric_name,
                    "point": vals["point"],
                    "ci_lower": vals["lower"],
                    "ci_upper": vals["upper"],
                })
        pd.DataFrame(layer1_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_test_metrics.csv", index=False,
        )
        logger.info("Wrote phase8_test_metrics.csv (%d rows)", len(layer1_rows))

        # 5. Layer 2 metrics
        cal_metrics, reliability_tables, coverage_rows, singletons = _layer2_metrics(
            df_test, per_film,
        )
        cal_rows = []
        for target, mdict in cal_metrics.items():
            row = {"target": target, **mdict}
            cal_rows.append(row)
        pd.DataFrame(cal_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_calibration_test.csv", index=False,
        )
        pd.DataFrame(coverage_rows).to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_coverage_test.csv", index=False,
        )
        logger.info("Wrote phase8_calibration_test.csv + phase8_coverage_test.csv")

        # 6. Layer 3 metrics
        sys_row, baseline_rows, sweep_df = _layer3_metrics(df_test, per_film)
        decision_df = pd.DataFrame(baseline_rows + [sys_row])
        decision_df.to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_decision_evaluation_test.csv", index=False,
        )
        sweep_df.to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_decision_sensitivity_test.csv", index=False,
        )
        logger.info(
            "Decision: system total cost = $%dM, baselines vs system written",
            int(sys_row["total_cost"] / 1e6),
        )

        # 7. Layer 4 metrics
        test_ranking, comparison, overlap_df, rho_vs_native, jaccard, raw_shap = (
            _layer4_metrics(artifacts, X_test)
        )
        test_ranking.to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_top_shap_test.csv", index=False,
        )
        if not comparison.empty:
            comparison.to_csv(
                paths.REPORTS_TABLES_DIR / "phase8_shap_vs_native_test.csv", index=False,
            )
        if not overlap_df.empty:
            overlap_df["jaccard_top15"] = jaccard
            overlap_df.to_csv(
                paths.REPORTS_TABLES_DIR / "phase8_shap_test_vs_cal.csv", index=False,
            )
        logger.info(
            "Layer 4: SHAP-vs-native ρ=%.3f, test-vs-cal Jaccard=%.3f",
            rho_vs_native, jaccard,
        )

        # 8. Error-analysis cuts
        cuts: dict[str, pd.DataFrame] = {}
        for cut_col, fname in [
            ("primary_genre_bucketed", "phase8_error_by_genre.csv"),
            ("decade_bucket", "phase8_error_by_decade.csv"),
            ("budget_tier", "phase8_error_by_budget_tier.csv"),
            ("length_tier", "phase8_error_by_length_tier.csv"),
        ]:
            cdf = cut_metrics(
                per_film_full,
                cut_col=cut_col,
                target_col="true_label",
                prob_col="calibrated_probability_roi_gt_2",
                action_col="recommended_action",
            )
            cdf.to_csv(paths.REPORTS_TABLES_DIR / fname, index=False)
            cuts[cut_col] = cdf
            logger.info("Wrote %s (%d rows)", fname, len(cdf))

        # 9. Most-correct / most-wrong galleries
        most_correct = gallery(
            per_film_full, target_col="true_label",
            prob_col="calibrated_probability_roi_gt_2",
            direction="most_correct", top_n=15,
        )
        most_wrong = gallery(
            per_film_full, target_col="true_label",
            prob_col="calibrated_probability_roi_gt_2",
            direction="most_wrong", top_n=15,
        )
        gallery_cols = [
            "imdb_id", "movie_name", "primary_genre_bucketed",
            "release_year_parsed", "true_label",
            "calibrated_probability_roi_gt_2", "recommended_action",
            "per_film_log_loss",
        ]
        most_correct[gallery_cols].to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_top_correct_roi_gt_2.csv", index=False,
        )
        most_wrong[gallery_cols].to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_top_wrong_roi_gt_2.csv", index=False,
        )
        logger.info("Wrote top_correct + top_wrong galleries")

        # 10. Example gallery
        selections = select_examples(
            per_film_full,
            target_col="true_label",
            prob_col="calibrated_probability_roi_gt_2",
            action_col="recommended_action",
            genre_col="primary_genre_bucketed",
            name_col="movie_name",
        )
        gallery_md = render_gallery_markdown(
            selections, per_film_full, target_col="true_label",
        )
        (paths.REPORTS_TABLES_DIR / "phase8_example_gallery.md").write_text(gallery_md)
        logger.info("Wrote phase8_example_gallery.md (%d examples)", len(selections))

        # 11. Figures
        plot_calibration_test(reliability_tables)
        plot_coverage_test(pd.DataFrame(coverage_rows))
        plot_decision_costs_test(decision_df)
        plot_decision_sensitivity_test(sweep_df)
        plot_per_genre_metrics_test(cuts["primary_genre_bucketed"])
        plot_top_shap_test(test_ranking)

        # 12. Escalation triggers check
        triggers = _check_escalation_triggers(
            layer1=layer1,
            coverage_rows=coverage_rows,
            sys_total_cost=sys_row["total_cost"],
            jaccard=jaccard,
            smoke_passed=smoke["passed"],
        )
        pd.DataFrame(triggers).to_csv(
            paths.REPORTS_TABLES_DIR / "phase8_escalation_triggers.csv", index=False,
        )
        for t in triggers:
            logger.info("Trigger %s: fired=%s | %s",
                        t["trigger_id"], t["fired"], t["status_note"])

        # 13. Save artifacts to run + RUNS.md
        run.record_metrics({
            "headline_test_auc_roi_gt_2": layer1["roi_gt_2"]["auc_roc"]["point"],
            "headline_test_auc_ci": [
                layer1["roi_gt_2"]["auc_roc"]["lower"],
                layer1["roi_gt_2"]["auc_roc"]["upper"],
            ],
            "log_roi_test_rmse": layer1["log_roi"]["rmse"]["point"],
            "calibrated_ece_roi_gt_2": cal_metrics["roi_gt_2"]["ece"],
            "calibrated_ece_roi_gt_1": cal_metrics["roi_gt_1"]["ece"],
            "system_test_total_cost": sys_row["total_cost"],
            "system_action_p_greenlight": sys_row["p_greenlight"],
            "system_action_p_pass": sys_row["p_pass"],
            "system_action_p_refer": sys_row["p_refer"],
            "shap_vs_native_test_rho": rho_vs_native,
            "shap_test_vs_cal_top15_jaccard": jaccard,
            "smoke_test_passed": bool(smoke["passed"]),
            "n_triggers_fired": int(sum(t["fired"] for t in triggers)),
            "fired_triggers": [t["trigger_id"] for t in triggers if t["fired"]],
        })
        run.append_to_runs_md(
            model_family="phase8_end_to_end",
            features_group="standalone_positive_union_mpnet",
            key_metric=(
                f"test AUC roi_gt_2={layer1['roi_gt_2']['auc_roc']['point']:.3f} "
                f"[{layer1['roi_gt_2']['auc_roc']['lower']:.3f}, "
                f"{layer1['roi_gt_2']['auc_roc']['upper']:.3f}]; "
                f"system cost ${sys_row['total_cost']/1e6:.1f}M; "
                f"jaccard top15={jaccard:.2f}"
            ),
            notes="Phase 8 end-to-end + final test-set evaluation",
        )

        return {
            "layer1": layer1,
            "calibration": cal_metrics,
            "coverage": coverage_rows,
            "system_decision": sys_row,
            "baselines": baseline_rows,
            "sweep": sweep_df,
            "shap_test_ranking": test_ranking,
            "rho_vs_native": rho_vs_native,
            "jaccard": jaccard,
            "smoke": smoke,
            "triggers": triggers,
            "per_film": per_film_full,
        }


def _check_escalation_triggers(
    *,
    layer1: dict,
    coverage_rows: list[dict],
    sys_total_cost: float,
    jaccard: float,
    smoke_passed: bool,
) -> list[dict]:
    """Section 5 of the pre-registration: five potential triggers."""
    triggers: list[dict] = []

    # Trigger #1: predictive-performance gap (test AUC < 0.5520)
    test_auc = layer1["roi_gt_2"]["auc_roc"]["point"]
    triggers.append({
        "trigger_id": "1_predictive_performance_gap",
        "metric": "test_auc_roi_gt_2",
        "value": test_auc,
        "threshold": 0.5520,
        "fired": bool(test_auc < 0.5520),
        "status_note": (
            "Test AUC below the 0.55 floor (Phase 4 OOF 0.6520 - 10pp). "
            "Probable cause: covariate shift across cal/test."
            if test_auc < 0.5520 else
            "Test AUC clears the 0.55 lower band of the Phase 4 forward-expected range."
        ),
    })

    # Trigger #2: coverage out-of-band at 0.90
    cov_targets = []
    for row in coverage_rows:
        if abs(row["level"] - 0.9) < 1e-9:
            cov_targets.append((row["target"], row["empirical_coverage"]))
    fired_targets = [
        (t, v) for (t, v) in cov_targets if v < 0.85 or v > 0.95
    ]
    triggers.append({
        "trigger_id": "2_coverage_out_of_band",
        "metric": "empirical_coverage_at_0.9 per target",
        "value": ", ".join(f"{t}={v:.3f}" for t, v in cov_targets),
        "threshold": "0.85-0.95",
        "fired": bool(len(fired_targets) > 0),
        "status_note": (
            f"Targets out-of-band: {fired_targets!r}"
            if fired_targets else
            "All coverages within the ±5pp band of nominal 0.90."
        ),
    })

    # Trigger #3: decision-cost regression vs cal ($2.6M ceiling)
    triggers.append({
        "trigger_id": "3_decision_cost_regression",
        "metric": "system_total_cost",
        "value": sys_total_cost,
        "threshold": 2_600_000,
        "fired": bool(sys_total_cost > 2_600_000),
        "status_note": (
            f"Total system cost ${sys_total_cost/1e6:.2f}M exceeds 2x the cal "
            "value ($2.6M). Probable cause: a small handful of test "
            "Greenlights landed on flops."
            if sys_total_cost > 2_600_000 else
            f"Total system cost ${sys_total_cost/1e6:.2f}M is within 2x of cal."
        ),
    })

    # Trigger #4: SHAP test-vs-cal top-15 overlap
    triggers.append({
        "trigger_id": "4_shap_test_vs_cal_overlap",
        "metric": "jaccard_top15",
        "value": jaccard,
        "threshold": 0.6,
        "fired": bool(jaccard < 0.6),
        "status_note": (
            f"Jaccard {jaccard:.2f} below the 0.6 floor: SHAP "
            "rankings drift across cal/test."
            if jaccard < 0.6 else
            f"Jaccard {jaccard:.2f} clears the 0.6 floor."
        ),
    })

    # Trigger #5: smoke-test
    triggers.append({
        "trigger_id": "5_smoke_test_mismatch",
        "metric": "consistency_check",
        "value": str(smoke_passed),
        "threshold": "PASS",
        "fired": bool(not smoke_passed),
        "status_note": (
            "Smoke test mismatch: per-film triage is non-deterministic."
            if not smoke_passed else
            "Smoke test passes: per-film triage is deterministic."
        ),
    })

    return triggers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 8 end-to-end evaluation")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce log verbosity")
    args = parser.parse_args()
    run_phase8(verbose=not args.quiet)


if __name__ == "__main__":
    main()
