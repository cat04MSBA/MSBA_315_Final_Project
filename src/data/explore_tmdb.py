"""Phase 1, Task 2 — TMDB exploratory profile.

Reproduces the prior-known TMDB headline numbers (sanity check), then saves
year and genre distribution plots and a small summary-statistics table for
the films-with-both subset (``budget > 0`` AND ``revenue > 0``).

Run from the project root:

    python -m src.data.explore_tmdb

Outputs:

- ``reports/figures/phase1_tmdb_year_distribution.png``
- ``reports/figures/phase1_tmdb_genre_distribution.png``
- ``reports/tables/phase1_tmdb_summary_stats.csv``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; no display required.

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_tmdb import load_tmdb, summarize_tmdb
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Hard-coded reference numbers from PROJECT_CONTEXT.txt Section 5 (prior exploration).
# We expect to reproduce these exactly; a divergence flags a corrupt/wrong CSV.
EXPECTED_COUNTS: dict[str, int] = {
    "total": 4803,
    "with_budget": 3766,
    "with_revenue": 3376,
    "with_both": 3229,
}


def _check_against_expected(actual: dict[str, int]) -> None:
    """Log loudly if the loaded counts don't match the documented baseline."""
    mismatches = {k: (EXPECTED_COUNTS[k], actual[k]) for k in EXPECTED_COUNTS if EXPECTED_COUNTS[k] != actual[k]}
    if mismatches:
        for key, (expected, got) in mismatches.items():
            logger.warning(
                "TMDB sanity-check mismatch: %s expected %d, got %d", key, expected, got
            )
    else:
        logger.info("TMDB sanity check passed: counts match documented baseline.")


def _plot_year_distribution(df_both: pd.DataFrame, out_path) -> None:
    """Histogram of release years for films with both budget and revenue."""
    years = df_both["release_year"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2), edgecolor="black", alpha=0.85)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Films (with budget > 0 AND revenue > 0)")
    ax.set_title(f"TMDB year distribution — {len(years):,} films with budget+revenue")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_genre_distribution(df_both: pd.DataFrame, out_path) -> None:
    """Bar chart of genre counts (a film can contribute to multiple bars)."""
    exploded = df_both.explode("genre_names")
    counts = exploded["genre_names"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("Films (a film with N genres counts in N bars)")
    ax.set_title(f"TMDB genre distribution — {len(df_both):,} films with budget+revenue")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _summary_stats_table(df_both: pd.DataFrame) -> pd.DataFrame:
    """Median / mean / std for budget and revenue, plus revenue/budget ratio."""
    table = pd.DataFrame(
        {
            "budget": df_both["budget"].describe(),
            "revenue": df_both["revenue"].describe(),
            "revenue_to_budget_ratio": (df_both["revenue"] / df_both["budget"]).describe(),
        }
    )
    return table


def main() -> None:
    paths.ensure_dirs()

    df = load_tmdb()
    actual_counts = summarize_tmdb(df)
    logger.info("TMDB counts: %s", actual_counts)
    _check_against_expected(actual_counts)

    mask_both = (df["budget"] > 0) & (df["revenue"] > 0)
    df_both = df.loc[mask_both].copy()
    logger.info("Films with both budget and revenue > 0: %d", len(df_both))

    _plot_year_distribution(
        df_both, paths.REPORTS_FIGURES_DIR / "phase1_tmdb_year_distribution.png"
    )
    _plot_genre_distribution(
        df_both, paths.REPORTS_FIGURES_DIR / "phase1_tmdb_genre_distribution.png"
    )

    table = _summary_stats_table(df_both)
    table_path = paths.REPORTS_TABLES_DIR / "phase1_tmdb_summary_stats.csv"
    table.to_csv(table_path)
    logger.info("Saved %s", table_path)

    # Log a compact view of the table for the phase summary.
    logger.info("Budget/revenue summary (films with both > 0):\n%s", table.round(2).to_string())


if __name__ == "__main__":
    main()
