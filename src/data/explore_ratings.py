"""Phase 1, Task 2 — Standalone ratings-dataset profile.

Produces the dataset-level diagnostics for the IMDb-TMDB metadata CSV
*before* it has been joined to MovieSum: headline counts, year coverage,
genre distribution among financially-complete films, and a budget /
revenue summary table. This is the "do we trust this data source?" pass
— the join itself happens in :mod:`src.data.join_corpus`.

Run from the project root::

    python -m src.data.explore_ratings

Outputs:

- ``reports/figures/phase1_ratings_year_distribution.png``
- ``reports/figures/phase1_ratings_genre_distribution.png``
- ``reports/tables/phase1_ratings_summary_stats.csv``
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python3 -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_ratings import load_ratings, summarize_ratings
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _plot_year_distribution(df_both: pd.DataFrame, out_path) -> None:
    """Histogram of release years for films with both budget AND revenue.

    Clipped to 1900–2025 because a few rows in the raw data carry
    nonsensical future years (2055, 2099) that are scheduled-release or
    error entries.
    """
    years = df_both["release_year_parsed"].dropna().astype(int)
    sane = years[(years >= 1900) & (years <= 2025)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(sane, bins=range(int(sane.min()), int(sane.max()) + 2),
            edgecolor="black", alpha=0.85)
    ax.set(xlabel="Release year",
           ylabel="Films (with budget > 0 AND revenue > 0)",
           title=f"Ratings-dataset year distribution — {len(sane):,} films "
                 f"({len(years) - len(sane)} out-of-range dropped)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _plot_genre_distribution(df_both: pd.DataFrame, out_path) -> None:
    """Bar chart of genre counts among films-with-both."""
    exploded = df_both.explode("genres_parsed")
    counts = exploded["genres_parsed"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1],
            color="steelblue", edgecolor="black")
    ax.set(xlabel="Films (multi-genre films counted in each bar)",
           title=f"Ratings-dataset genre distribution — "
                 f"{len(df_both):,} films with budget+revenue")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.debug("Wrote %s", out_path)


def _summary_stats_table(df_both: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "budget":  df_both["budget"].describe(),
        "revenue": df_both["revenue"].describe(),
        "revenue_to_budget_ratio": (
            df_both["revenue"] / df_both["budget"]
        ).describe(),
    })


def main() -> pd.DataFrame:
    """Run Phase 1 Task 2 end-to-end and return the films-with-both subset."""
    paths.ensure_dirs()

    df = load_ratings()
    counts = summarize_ratings(df)
    logger.debug("Ratings counts: %s", counts)

    mask_both = (df["budget"] > 0) & (df["revenue"] > 0)
    df_both = df.loc[mask_both].copy()
    logger.info("Films with budget AND revenue > 0: %s", f"{len(df_both):,}")

    _plot_year_distribution(
        df_both, paths.REPORTS_FIGURES_DIR / "phase1_ratings_year_distribution.png"
    )
    _plot_genre_distribution(
        df_both, paths.REPORTS_FIGURES_DIR / "phase1_ratings_genre_distribution.png"
    )

    table = _summary_stats_table(df_both)
    table_path = paths.REPORTS_TABLES_DIR / "phase1_ratings_summary_stats.csv"
    table.to_csv(table_path)
    logger.debug("Wrote %s", table_path)
    logger.info("Saved 2 figures + 1 summary table to reports/")

    print("\n=== Ratings dataset — Phase 1 Task 2 summary ===")
    for k, v in counts.items():
        print(f"  {k:>14}: {v:,}")
    print(f"  {'films-with-both':>14}: {len(df_both):,} (used for distributions)")
    print()
    print("Budget / revenue summary (films with both > 0):")
    print(table.round(2).to_string())
    return df_both


if __name__ == "__main__":
    main()
