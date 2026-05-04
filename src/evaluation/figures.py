"""Phase 8 deliverable figures.

Six figures per ``docs/proposals/phase8_preregistration.md``
Section 6.3. Each function loads the relevant table or runs a
small computation and writes to ``reports/figures/``.

* ``phase8_calibration_test.png`` — reliability diagrams (both
  classification targets, side-by-side).
* ``phase8_coverage_test.png`` — empirical coverage vs nominal at
  the four confidence levels.
* ``phase8_decision_costs_test.png`` — system vs five baselines,
  log scale.
* ``phase8_decision_sensitivity_test.png`` — refer-cost sweep.
* ``phase8_per_genre_metrics_test.png`` — per-genre AUC + refer
  rate.
* ``phase8_top_shap_test.png`` — top-20 mean |SHAP| on test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _save(fig, name: str) -> Path:
    out = paths.REPORTS_FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", out.name)
    return out


# ---------------------------------------------------------------------------
# 1. Reliability diagrams
# ---------------------------------------------------------------------------


def plot_calibration_test(
    reliability_tables: dict[str, pd.DataFrame],
    out_name: str = "phase8_calibration_test.png",
) -> Path:
    """Side-by-side reliability diagrams for each classification target."""
    targets = list(reliability_tables.keys())
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, target in enumerate(targets):
        df = reliability_tables[target]
        valid = df.dropna(subset=["mean_predicted", "empirical_accuracy"])
        ax = axes[0, i]
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
        ax.plot(
            valid["mean_predicted"], valid["empirical_accuracy"],
            marker="o", linewidth=1.5, label="calibrated",
        )
        ax.scatter(
            valid["mean_predicted"], valid["empirical_accuracy"],
            s=valid["count"] * 3, alpha=0.4,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title(f"{target} — test reliability (n={int(df['count'].sum())})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# 2. Coverage curves
# ---------------------------------------------------------------------------


def plot_coverage_test(
    coverage_df: pd.DataFrame,
    out_name: str = "phase8_coverage_test.png",
) -> Path:
    """Empirical coverage vs nominal level per target."""
    targets = sorted(coverage_df["target"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            label="nominal = empirical")
    for target in targets:
        sub = coverage_df[coverage_df["target"] == target].sort_values("level")
        ax.plot(
            sub["level"], sub["empirical_coverage"],
            marker="o", linewidth=2, label=target,
        )
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Phase 8 conformal coverage on the test set")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    # Tolerance band ±5pp at 0.9
    ax.axhspan(0.85, 0.95, xmin=0, xmax=1, alpha=0.05, color="green")
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# 3. Decision costs (system + 5 baselines)
# ---------------------------------------------------------------------------


def plot_decision_costs_test(
    baselines_df: pd.DataFrame,
    out_name: str = "phase8_decision_costs_test.png",
) -> Path:
    """Bar chart of total cost on the test set, log scale."""
    df = baselines_df.copy()
    df = df.sort_values("total_cost", ascending=True)
    colors = ["#2c7fb8" if s == "system" else "#a6cee3" for s in df["strategy"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["strategy"], df["total_cost"].clip(lower=1), color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Total cost on test set (USD, log scale)")
    ax.set_title("Phase 8 — system vs 5 baselines under default cost matrix")
    ax.grid(True, axis="x", alpha=0.3, which="both")
    for strategy, total in zip(df["strategy"], df["total_cost"]):
        ax.text(
            max(total, 1) * 1.1, strategy,
            f"${total/1e6:.1f}M" if total >= 1e6 else f"${total/1e3:.0f}K",
            va="center", fontsize=9,
        )
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# 4. Refer-cost sensitivity sweep
# ---------------------------------------------------------------------------


def plot_decision_sensitivity_test(
    sensitivity_df: pd.DataFrame,
    out_name: str = "phase8_decision_sensitivity_test.png",
) -> Path:
    """Total cost + refer rate vs refer-cost level."""
    df = sensitivity_df.copy()
    # Sort by refer-cost ascending
    df = df.sort_values("cost_refer_flop")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    refer_costs = df["cost_refer_flop"].astype(float).clip(lower=1)
    ax1.plot(refer_costs, df["total_cost_system"], marker="o",
             color="#2c7fb8", label="Total cost (USD)")
    ax2.plot(refer_costs, df["p_refer"], marker="s",
             color="#e6550d", label="Refer rate", linestyle="--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Refer cost (USD, log scale)")
    ax1.set_ylabel("Total cost (USD, log)", color="#2c7fb8")
    ax2.set_ylabel("Refer rate", color="#e6550d")
    ax2.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("Phase 8 — refer-cost sensitivity (test set)")
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# 5. Per-genre metrics
# ---------------------------------------------------------------------------


def plot_per_genre_metrics_test(
    by_genre_df: pd.DataFrame,
    out_name: str = "phase8_per_genre_metrics_test.png",
) -> Path:
    """Two panels: per-genre AUC and per-genre refer rate."""
    df = by_genre_df.copy().sort_values("auc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(df["cut_value"], df["auc"], color="#2c7fb8")
    axes[0].axvline(0.5, color="grey", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Test-set AUC-ROC")
    axes[0].set_title("Per-genre AUC")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[1].barh(df["cut_value"], df["p_refer"], color="#e6550d")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Refer rate")
    axes[1].set_title("Per-genre refer rate (default cost matrix)")
    axes[1].grid(True, axis="x", alpha=0.3)
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# 6. Test-set top SHAP
# ---------------------------------------------------------------------------


def plot_top_shap_test(
    shap_ranking: pd.DataFrame,
    top_n: int = 20,
    out_name: str = "phase8_top_shap_test.png",
) -> Path:
    """Horizontal bar chart of top-N mean |SHAP| on the test set."""
    df = shap_ranking.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse for top-down ordering in the plot
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    colors = [
        "#2c7fb8" if v >= 0 else "#d7301f"
        for v in df["mean_signed_shap"]
    ]
    ax.barh(df["feature"], df["mean_abs_shap"], color=colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Phase 8 — top-{top_n} test-set SHAP features (roi_gt_2)")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, out_name)
