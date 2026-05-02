"""Phase 1 — Profile the joined corpus.

Reads ``data/interim/phase1_joined_corpus.parquet`` (produced by
``join_corpus.py``), restricts to films with all four signals (matched
to the ratings dataset AND ``budget > 0`` AND ``revenue > 0`` AND a
non-zero rating), and produces the diagnostic plots and summary table
the Phase 1 brief asks for.

For the rating column we prefer ``IMDB_Rating`` when present and fall
back to ``vote_average`` otherwise — IMDB_Rating has better coverage
on this dataset and is the metric the report ultimately discusses.

Run from the project root::

    python -m src.data.profile_corpus
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python3 -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_working_corpus() -> pd.DataFrame:
    """Load the Phase 1 joined parquet and filter to films with all four signals.

    Adds an ``effective_rating`` column that prefers ``IMDB_Rating`` and
    falls back to ``vote_average`` when IMDB_Rating is missing — saves
    every downstream chart from re-implementing the preference rule.
    """
    parquet_path = paths.DATA_INTERIM_DIR / "phase1_joined_corpus.parquet"
    logger.info("Reading joined corpus")
    logger.debug("Source parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    matched = df["id"].notna()
    has_budget = df["budget"].fillna(0) > 0
    has_revenue = df["revenue"].fillna(0) > 0
    has_imdb_rating = df["IMDB_Rating"].fillna(0) > 0
    has_tmdb_rating = df["vote_average"].fillna(0) > 0

    working = df.loc[
        matched & has_budget & has_revenue & (has_imdb_rating | has_tmdb_rating)
    ].copy()

    working["effective_rating"] = working["IMDB_Rating"].where(
        working["IMDB_Rating"].fillna(0) > 0,
        working["vote_average"],
    )

    logger.info("Working corpus (4-signal): %d / %d total joined rows",
                len(working), len(df))
    return working


def _save_year_plot(df: pd.DataFrame, out_path) -> None:
    years = df["release_year_parsed"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2),
            edgecolor="black", alpha=0.85)
    ax.set(xlabel="Release year", ylabel="Films",
           title=f"Joined-corpus year distribution — N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _save_genre_plot(df: pd.DataFrame, out_path) -> None:
    exploded = df.explode("genres_parsed")
    counts = exploded["genres_parsed"].dropna().value_counts()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1],
            color="steelblue", edgecolor="black")
    ax.set(xlabel="Films (multi-genre films counted in each bar)",
           title=f"Joined-corpus genre distribution — N={len(df):,}")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _save_budget_revenue_plot(df: pd.DataFrame, out_path) -> None:
    """Two-row plot: top is raw scale ($M), bottom is log10."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, color, ax_raw, ax_log in (
        ("budget",  "steelblue", axes[0, 0], axes[1, 0]),
        ("revenue", "indianred", axes[0, 1], axes[1, 1]),
    ):
        vals = df[col].astype(float)
        ax_raw.hist(vals / 1e6, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_raw.set(xlabel=f"{col.capitalize()} ($M)", ylabel="Films",
                   title=f"{col.capitalize()} — raw")
        ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

        log_vals = np.log10(vals.clip(lower=1))
        ax_log.hist(log_vals, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_log.set(xlabel=f"log10({col.capitalize()})", ylabel="Films",
                   title=f"{col.capitalize()} — log10")
        ax_log.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"Budget and revenue — N={len(df):,}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _save_rating_plot(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["effective_rating"], bins=np.arange(0, 10.5, 0.25),
            edgecolor="black", alpha=0.85)
    ax.set(xlabel="Rating (IMDB_Rating preferred, vote_average fallback)",
           ylabel="Films",
           title=f"Rating distribution — N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _save_roi_plot(df: pd.DataFrame, out_path) -> None:
    """ROI = revenue / budget. Plot log10 to handle the heavy tail."""
    roi = (df["revenue"] / df["budget"]).replace([np.inf, -np.inf], np.nan).dropna()
    log_roi = np.log10(roi.clip(lower=1e-3))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(log_roi, bins=60, color="darkgreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.7,
               label="break-even (ROI=1)")
    ax.set(xlabel="log10(revenue / budget)", ylabel="Films",
           title=f"ROI distribution (log10) — N={len(roi):,}")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _save_length_plot(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["script_char_len"] / 1000.0, bins=40, color="orange",
            edgecolor="black", alpha=0.85)
    ax.set(xlabel="Screenplay length (thousand characters)", ylabel="Films",
           title=f"Screenplay length — joined corpus N={len(df):,}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact numeric summary for the phase-summary doc."""
    roi = (df["revenue"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
    rows = [
        ("films_with_all_four_signals", int(len(df))),
        ("year_min", int(df["release_year_parsed"].min())),
        ("year_max", int(df["release_year_parsed"].max())),
        ("year_median", int(df["release_year_parsed"].median())),
        ("budget_median_usd", int(df["budget"].median())),
        ("budget_mean_usd", int(df["budget"].mean())),
        ("revenue_median_usd", int(df["revenue"].median())),
        ("revenue_mean_usd", int(df["revenue"].mean())),
        ("rating_mean", round(float(df["effective_rating"].mean()), 3)),
        ("rating_median", round(float(df["effective_rating"].median()), 3)),
        ("roi_median", round(float(roi.median()), 3)),
        ("roi_pct_profitable", round(100.0 * float((roi > 1).mean()), 2)),
        ("screenplay_chars_median", int(df["script_char_len"].median())),
        ("screenplay_chars_mean", int(df["script_char_len"].mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> pd.DataFrame:
    """Run Phase 1 profiling end-to-end and return the working corpus."""
    paths.ensure_dirs()
    df = load_working_corpus()

    figs = paths.REPORTS_FIGURES_DIR
    _save_year_plot(df,            figs / "phase1_joined_year_distribution.png")
    _save_genre_plot(df,           figs / "phase1_joined_genre_distribution.png")
    _save_budget_revenue_plot(df,  figs / "phase1_joined_budget_revenue_distribution.png")
    _save_rating_plot(df,          figs / "phase1_joined_rating_distribution.png")
    _save_roi_plot(df,             figs / "phase1_joined_roi_distribution.png")
    _save_length_plot(df,          figs / "phase1_joined_screenplay_length.png")

    summary = summary_table(df)
    summary_path = paths.REPORTS_TABLES_DIR / "phase1_joined_corpus_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.debug("Wrote %s", summary_path)
    logger.info("Saved 6 figures + 1 summary table to reports/")

    print("\n=== Joined corpus (4-signal subset) — Phase 1 summary ===")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
