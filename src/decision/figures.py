"""Phase 6 figure generation.

Reads the saved tables and produces the four pre-registered
figures: cost curve over the refer-cost sweep, action distribution
across cost-matrix variants, per-genre action breakdown, and a
baselines comparison bar chart.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def plot_cost_curve(out_path: Path) -> Path:
    """Refer-cost sweep: total cost + refer rate as functions of refer cost."""
    df = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase6_sensitivity.csv")
    refer_sweep = df[df["cost_matrix_name"].str.startswith("refer_")].copy()
    if refer_sweep.empty:
        logger.warning("No refer_* variants in sensitivity table")
        return out_path
    # Parse the dollar amount from cost_matrix_name (e.g. "refer_5K", "refer_1M")
    def parse_amount(name: str) -> float:
        s = name.removeprefix("refer_")
        if s.endswith("K"):
            return float(s[:-1]) * 1_000
        if s.endswith("M"):
            return float(s[:-1]) * 1_000_000
        return float(s)
    refer_sweep["refer_cost"] = refer_sweep["cost_matrix_name"].apply(parse_amount)
    refer_sweep = refer_sweep.sort_values("refer_cost")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(
        refer_sweep["refer_cost"].clip(lower=1),
        refer_sweep["total_cost_system"] / 1e6,
        "o-", color="tab:red", linewidth=1.4, label="system total cost",
    )
    ax1.set_xlabel("refer cost per film (USD; log scale, clipped at $1)")
    ax1.set_ylabel("total cost on calibration set (USD M)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(
        refer_sweep["refer_cost"].clip(lower=1),
        refer_sweep["p_refer"],
        "o--", color="tab:blue", linewidth=1.2, label="refer rate",
    )
    ax2.set_ylabel("refer rate (fraction of films)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(-0.02, 1.02)

    ax1.set_title(
        "Phase 6 sensitivity: total cost + refer rate vs refer cost parameter",
        fontsize=11,
    )
    ax1.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_action_distribution(out_path: Path) -> Path:
    """Stacked-bar action proportion per cost-matrix variant."""
    df = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase6_sensitivity.csv")
    df = df.sort_values("cost_matrix_name").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    ax.bar(x, df["p_greenlight"], label="Greenlight", color="tab:green")
    ax.bar(x, df["p_pass"], bottom=df["p_greenlight"], label="Pass", color="tab:orange")
    ax.bar(x, df["p_refer"],
           bottom=df["p_greenlight"] + df["p_pass"],
           label="Refer", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cost_matrix_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of films")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 6 action distribution per cost-matrix variant")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_per_genre_actions(out_path: Path) -> Path:
    """Per-genre action breakdown under the default cost matrix."""
    df = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase6_per_genre_actions.csv")
    df = df[df["n"] >= 5].sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    ax.bar(x, df["default_p_greenlight"], label="Greenlight", color="tab:green")
    ax.bar(x, df["default_p_pass"],
           bottom=df["default_p_greenlight"], label="Pass", color="tab:orange")
    ax.bar(x, df["default_p_refer"],
           bottom=df["default_p_greenlight"] + df["default_p_pass"],
           label="Refer", color="tab:red")
    for i, n in enumerate(df["n"]):
        ax.text(i, 1.02, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["genre"], rotation=30, ha="right")
    ax.set_ylabel("fraction of films")
    ax.set_ylim(0, 1.1)
    ax.set_title("Phase 6 per-genre action breakdown (default cost matrix)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_baselines_comparison(out_path: Path) -> Path:
    """Total-cost bar chart: system vs five baselines."""
    df = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase6_baselines.csv")
    df = df.sort_values("total_cost", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue" if s == "system" else "tab:gray" for s in df["strategy"]]
    bars = ax.barh(df["strategy"], df["total_cost"] / 1e6, color=colors)
    for bar, value in zip(bars, df["total_cost"] / 1e6):
        ax.text(
            value * 1.02, bar.get_y() + bar.get_height() / 2,
            f"${value:.1f}M", va="center", fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("total cost on calibration set (USD M; log scale)")
    ax.set_title("Phase 6: system vs naive baselines under the default cost matrix")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
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
    plot_cost_curve(paths.REPORTS_FIGURES_DIR / "phase6_cost_curve.png")
    plot_action_distribution(paths.REPORTS_FIGURES_DIR / "phase6_action_distribution.png")
    plot_per_genre_actions(paths.REPORTS_FIGURES_DIR / "phase6_per_genre_actions.png")
    plot_baselines_comparison(paths.REPORTS_FIGURES_DIR / "phase6_baselines_comparison.png")


if __name__ == "__main__":
    main()
