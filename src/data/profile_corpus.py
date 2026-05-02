"""Phase 1, Task 5 — Profile the joined corpus.

Reads ``data/interim/phase1_joined_corpus.parquet`` (produced by
``join_corpus.py``), restricts to films with all four signals (matched
to TMDB AND ``budget > 0`` AND ``revenue > 0``), and produces the
diagnostic plots the brief asks for: year, genre, budget (raw + log10),
revenue (raw + log10), IMDb rating, ROI, screenplay length.

Each plot is saved with a descriptive name; interpretations live in the
phase summary.

Run from the project root:

    python -m src.data.profile_corpus
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_working_corpus() -> pd.DataFrame:
    """Load the Phase 1 joined parquet and filter to films with all four signals.

    The four signals required by the brief: matched to TMDB, budget > 0,
    revenue > 0, IMDb rating > 0.
    """
    parquet_path = paths.DATA_INTERIM_DIR / "phase1_joined_corpus.parquet"
    logger.info("Reading joined corpus from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    matched = df["tmdb_id"].notna()
    has_budget = df["budget"].fillna(0) > 0
    has_revenue = df["revenue"].fillna(0) > 0
    has_rating = df["vote_average"].fillna(0) > 0

    working = df.loc[matched & has_budget & has_revenue & has_rating].copy()
    logger.info(
        "Working corpus (4-signal): %d / %d total joined rows", len(working), len(df)
    )
    return working


def _save_year_plot(df: pd.DataFrame, out_path) -> None:
    years = df["tmdb_release_year"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2), edgecolor="black", alpha=0.85)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Films")
    ax.set_title(f"Joined-corpus year distribution — N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _save_genre_plot(df: pd.DataFrame, out_path) -> None:
    exploded = df.explode("tmdb_genre_names")
    counts = exploded["tmdb_genre_names"].dropna().value_counts()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("Films (multi-genre films counted in each bar)")
    ax.set_title(f"Joined-corpus genre distribution — N={len(df):,}")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _save_budget_revenue_plot(df: pd.DataFrame, out_path) -> None:
    """Two-row plot: top is raw scale, bottom is log10 scale, for budget and revenue."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for col, color, ax_raw, ax_log in (
        ("budget", "steelblue", axes[0, 0], axes[1, 0]),
        ("revenue", "indianred", axes[0, 1], axes[1, 1]),
    ):
        vals = df[col].astype(float)
        # Raw scale, in millions for readability.
        ax_raw.hist(vals / 1e6, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_raw.set_xlabel(f"{col.capitalize()} ($ millions)")
        ax_raw.set_ylabel("Films")
        ax_raw.set_title(f"{col.capitalize()} — raw scale")
        ax_raw.grid(axis="y", linestyle=":", alpha=0.6)
        # Log10 scale.
        log_vals = np.log10(vals.clip(lower=1))
        ax_log.hist(log_vals, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_log.set_xlabel(f"log10({col.capitalize()} $)")
        ax_log.set_ylabel("Films")
        ax_log.set_title(f"{col.capitalize()} — log10 scale")
        ax_log.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"Budget and revenue distributions — N={len(df):,}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _save_rating_plot(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["vote_average"], bins=np.arange(0, 10.5, 0.25), edgecolor="black", alpha=0.85)
    ax.set_xlabel("TMDB vote_average (IMDb-style 0-10)")
    ax.set_ylabel("Films")
    ax.set_title(f"Rating distribution — N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _save_roi_plot(df: pd.DataFrame, out_path) -> None:
    """ROI = revenue / budget. Plot log10 of ROI to handle the heavy tail."""
    roi = (df["revenue"] / df["budget"]).replace([np.inf, -np.inf], np.nan).dropna()
    log_roi = np.log10(roi.clip(lower=1e-3))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(log_roi, bins=60, color="darkgreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="break-even (ROI=1)")
    ax.set_xlabel("log10(revenue / budget)")
    ax.set_ylabel("Films")
    ax.set_title(f"ROI distribution (log10) — N={len(roi):,}")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _save_length_plot(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["script_char_len"] / 1000.0, bins=40, color="orange", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Screenplay length (thousand characters)")
    ax.set_ylabel("Films")
    ax.set_title(f"Screenplay length — joined corpus N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact numeric summary for the phase-summary doc."""
    roi = (df["revenue"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
    rows = [
        ("films_with_all_four_signals", int(len(df))),
        ("year_min", int(df["tmdb_release_year"].min())),
        ("year_max", int(df["tmdb_release_year"].max())),
        ("year_median", int(df["tmdb_release_year"].median())),
        ("budget_median_usd", int(df["budget"].median())),
        ("budget_mean_usd", int(df["budget"].mean())),
        ("revenue_median_usd", int(df["revenue"].median())),
        ("revenue_mean_usd", int(df["revenue"].mean())),
        ("rating_mean", round(float(df["vote_average"].mean()), 3)),
        ("rating_median", round(float(df["vote_average"].median()), 3)),
        ("roi_median", round(float(roi.median()), 3)),
        ("roi_pct_profitable", round(100.0 * float((roi > 1).mean()), 2)),
        ("screenplay_chars_median", int(df["script_char_len"].median())),
        ("screenplay_chars_mean", int(df["script_char_len"].mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    paths.ensure_dirs()
    df = _load_working_corpus()

    figs = paths.REPORTS_FIGURES_DIR
    _save_year_plot(df, figs / "phase1_joined_year_distribution.png")
    _save_genre_plot(df, figs / "phase1_joined_genre_distribution.png")
    _save_budget_revenue_plot(df, figs / "phase1_joined_budget_revenue_distribution.png")
    _save_rating_plot(df, figs / "phase1_joined_rating_distribution.png")
    _save_roi_plot(df, figs / "phase1_joined_roi_distribution.png")
    _save_length_plot(df, figs / "phase1_joined_screenplay_length.png")

    summary = _summary_table(df)
    summary_path = paths.REPORTS_TABLES_DIR / "phase1_joined_corpus_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved %s", summary_path)
    logger.info("Working-corpus summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
