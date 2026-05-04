"""Phase 5 figure generation.

Reads the saved tables (``phase5_coverage.csv``, ``phase5_set_sizes.csv``,
``phase5_interval_widths.csv``, ``phase5_calibration_metrics.csv``)
and the deployed wrapper artifacts (``phase5_calibrated_model_*.joblib``)
and produces the five pre-registered figures:

* ``phase5_reliability_post.png`` — per-target post-calibration
  reliability diagrams comparing uncalibrated, sigmoid, isotonic.
* ``phase5_coverage_levels.png`` — empirical-vs-nominal coverage
  curves at the four confidence levels per target.
* ``phase5_set_size_distribution.png`` — histogram of conformal
  prediction set sizes per classification target.
* ``phase5_interval_width_distribution.png`` — histogram of
  conformal interval widths for the regression target.
* ``phase5_refer_by_genre.png`` — bar chart of refer rate per
  primary genre on ``roi_gt_2`` at 0.90 confidence.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.metrics import reliability_curve
from src.calibration.probability import _predict_proba
from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    add_targets,
)
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


CONFIDENCE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]


def _load_cal_features(matrix_name: str, target: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)
    cal_ids = splits.loc[splits["split"] == "cal", "imdb_id"]
    df_cal = df_full[df_full["imdb_id"].isin(cal_ids)].reset_index(drop=True)
    matrix_spec = MATRICES[matrix_name]
    X_cal = build_matrix(matrix_spec, df_cal)
    if target in CLASSIFICATION_TARGETS:
        y_cal = df_cal[target].astype(int).values
    else:
        y_cal = df_cal[target].values
    genres = df_cal["primary_genre_bucketed"]
    return X_cal, y_cal, genres


def plot_reliability_post(out_path: Path) -> Path:
    """Per-classification-target reliability diagrams for all 3 methods."""
    cm = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase5_calibration_metrics.csv")
    cls_targets = sorted(set(cm["target"]) & set(CLASSIFICATION_TARGETS))
    if not cls_targets:
        logger.warning("No classification metrics in phase5_calibration_metrics.csv")
        return out_path

    fig, axes = plt.subplots(1, len(cls_targets), figsize=(6 * len(cls_targets), 5),
                             sharey=True)
    if len(cls_targets) == 1:
        axes = [axes]

    method_order = ["uncalibrated", "sigmoid", "isotonic"]
    method_colors = {"uncalibrated": "tab:gray", "sigmoid": "tab:orange", "isotonic": "tab:blue"}

    for ax, target in zip(axes, cls_targets):
        bundle = joblib.load(paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{target}.joblib")
        X_cal, y_cal, _ = _load_cal_features(bundle["matrix"], target)
        # Re-fit each method on full cal to draw the deployed reliability curve.
        from src.calibration.probability import _fit_calibrator
        # Pull the underlying base estimator path through the bundle reference.
        phase4_bundle = joblib.load(bundle["phase4_winner_path"])
        base_est = phase4_bundle["estimator"]
        score_method = bundle["score_method"]

        for method in method_order:
            if method == "uncalibrated":
                probs = _predict_proba(base_est, X_cal, score_method)
            else:
                cal = _fit_calibrator(base_est, X_cal, y_cal, method)
                probs = cal.predict_proba(X_cal)[:, 1]
            mean_pred, accuracy, counts = reliability_curve(y_cal, probs, n_bins=10)
            valid = counts > 0
            ax.plot(
                mean_pred[valid], accuracy[valid],
                marker="o", color=method_colors[method],
                label=f"{method} (full-cal in-sample)",
                linewidth=1.4,
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        ax.set_title(f"{target}\n({bundle['family']}, best={bundle['best_probability_method']})")
        ax.set_xlabel("predicted probability")
        if ax is axes[0]:
            ax.set_ylabel("observed fraction positive")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Phase 5 reliability diagrams (full-calibration-set in-sample fit)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_coverage_levels(out_path: Path) -> Path:
    """Empirical-vs-nominal coverage at the four confidence levels."""
    cov = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase5_coverage.csv")
    targets = sorted(cov["target"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"log_roi": "tab:green", "roi_gt_1": "tab:orange", "roi_gt_2": "tab:blue"}
    for target in targets:
        sub = cov[cov["target"] == target].groupby("level")["empirical_coverage"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            sub["level"], sub["mean"], yerr=sub["std"],
            marker="o", capsize=3, label=target, color=colors.get(target, "tab:gray"),
            linewidth=1.4,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], alpha=0.1, color="gray", label="±5pp band")
    ax.set_xlabel("nominal confidence level")
    ax.set_ylabel("empirical coverage (mean ± std across 5 folds)")
    ax.set_title("Phase 5 conformal coverage: empirical vs nominal")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_set_size_distribution(out_path: Path) -> Path:
    """Singleton vs refer rates per classification target per confidence level."""
    ss = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase5_set_sizes.csv")
    cls_targets = sorted(ss["target"].unique())
    fig, axes = plt.subplots(1, len(cls_targets), figsize=(6 * len(cls_targets), 4.5),
                             sharey=True)
    if len(cls_targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, cls_targets):
        sub = ss[ss["target"] == target].groupby("level")[
            ["singleton_rate", "refer_rate", "empty_rate"]
        ].mean().reset_index()
        x = sub["level"].astype(str).values
        ax.bar(x, sub["singleton_rate"], label="singleton", color="tab:green")
        ax.bar(x, sub["refer_rate"], bottom=sub["singleton_rate"],
               label="refer ({0,1})", color="tab:red")
        ax.bar(x, sub["empty_rate"], bottom=sub["singleton_rate"] + sub["refer_rate"],
               label="empty {}", color="tab:gray")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6,
                   label="50% threshold")
        ax.set_title(f"{target}")
        ax.set_xlabel("confidence level")
        if ax is axes[0]:
            ax.set_ylabel("fraction of films")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(
        "Phase 5 conformal prediction-set size distribution per confidence level",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_interval_width_distribution(out_path: Path) -> Path:
    """Mean conformal interval width per confidence level for the regression target."""
    iw = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase5_interval_widths.csv")
    targets = sorted(iw["target"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target in targets:
        sub = iw[iw["target"] == target].groupby("level")["mean_width"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            sub["level"], sub["mean"], yerr=sub["std"],
            marker="o", capsize=3, label=target, linewidth=1.4,
        )
    ax.set_xlabel("nominal confidence level")
    ax.set_ylabel("mean conformal interval width (log_roi units)")
    ax.set_title("Phase 5 conformal interval widths for the regression target")
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_refer_by_genre(out_path: Path, target: str = "roi_gt_2",
                        confidence_level: float = 0.90) -> Path:
    """Per-genre refer rate at the chosen level (deployed wrapper on full cal)."""
    bundle = joblib.load(paths.DATA_PROCESSED_DIR / f"phase5_calibrated_model_{target}.joblib")
    X_cal, y_cal, genres = _load_cal_features(bundle["matrix"], target)
    wrapper = bundle["conformal_wrapper"]
    _, y_pss = wrapper.predict_set(X_cal)

    level_idx = bundle["deployed_confidence_levels"].index(confidence_level)
    sets = y_pss[:, :, level_idx]  # (n, 2)
    refer = (sets.sum(axis=1) == 2)
    df = pd.DataFrame({
        "genre": genres.values,
        "refer": refer.astype(int),
        "n": 1,
    })
    summary = df.groupby("genre").agg(n=("n", "sum"), refer_rate=("refer", "mean")).reset_index()
    summary = summary[summary["n"] >= 5].sort_values("refer_rate", ascending=False)

    # Also load Phase 4 per-genre AUC to overlay.
    phase4_genre = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase4_per_genre_auc.csv")
    family = bundle["family"]
    phase4_genre = phase4_genre[["slice", family]].rename(columns={"slice": "genre", family: "auc"})

    merged = summary.merge(phase4_genre, on="genre", how="left")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(merged))
    ax1.bar(x, merged["refer_rate"], color="tab:red", alpha=0.7, label=f"refer rate @ {confidence_level}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["genre"], rotation=30, ha="right")
    ax1.set_ylabel("refer rate (Phase 5 conformal)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0, 1.05)
    for i, n in enumerate(merged["n"]):
        ax1.text(x[i], 1.0, f"n={int(n)}", ha="center", fontsize=8, color="black")

    ax2 = ax1.twinx()
    ax2.plot(x, merged["auc"], "o-", color="tab:blue",
             label=f"Phase 4 OOF AUC ({family})", linewidth=1.5)
    ax2.set_ylabel("Phase 4 OOF AUC", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0.4, 0.8)
    ax2.axhline(0.5, color="tab:blue", linestyle=":", linewidth=0.6)

    ax1.set_title(
        f"Phase 5: per-genre refer rate (confidence {confidence_level}) "
        f"vs Phase 4 OOF AUC on {target}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    set_log_level(args.log_level)
    paths.ensure_dirs()
    plot_reliability_post(paths.REPORTS_FIGURES_DIR / "phase5_reliability_post.png")
    plot_coverage_levels(paths.REPORTS_FIGURES_DIR / "phase5_coverage_levels.png")
    plot_set_size_distribution(paths.REPORTS_FIGURES_DIR / "phase5_set_size_distribution.png")
    plot_interval_width_distribution(paths.REPORTS_FIGURES_DIR / "phase5_interval_width_distribution.png")
    plot_refer_by_genre(paths.REPORTS_FIGURES_DIR / "phase5_refer_by_genre.png")


if __name__ == "__main__":
    main()
