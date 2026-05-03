"""Phase 2, Task 4 — Validate and re-profile the processed corpus.

This script is the audit pass on the master Parquet built by
``build_corpus.py``. It does three things:

1. Hard-asserts the structural invariants we expect (no nulls in
   critical columns, year range correct, ratios in [0,1], every film
   has parsed scenes, etc.). Loud failure if anything is off.
2. Saves Phase 2 versions of the diagnostic plots from Phase 1, this
   time on the post-filter corpus, plus new plots for the
   screenplay-structural features.
3. Saves Phase 2 summary tables (counts, per-genre, per-decade,
   missingness audit).

Run from the project root::

    python -m src.data.validate_processed_corpus
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


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def load_processed_corpus() -> pd.DataFrame:
    """Load the master Parquet produced by ``build_corpus.py``."""
    parquet_path = paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run `python -m src.data.build_corpus` first."
        )
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded processed corpus: %s films × %d columns",
                f"{len(df):,}", len(df.columns))
    return df


def hard_asserts(df: pd.DataFrame) -> None:
    """Assert every invariant a well-formed processed corpus must satisfy.

    Raises ``AssertionError`` on the first violation. Useful as a
    smoke test in CI / pre-flight, and as a guarantee that downstream
    phases can rely on these properties without re-checking.
    """
    # Identity / keys.
    assert len(df) > 0, "Processed corpus is empty"
    assert df["imdb_id"].is_unique, "Duplicate imdb_id rows"
    assert df["imdb_id"].notna().all(), "Some rows have null imdb_id"

    # Required outcome / financial columns.
    assert df["effective_rating"].notna().all(), "effective_rating has nulls"
    assert (df["effective_rating"].between(0, 10)).all(), "effective_rating out of [0, 10]"
    assert (df["budget"] > 0).all(), "Some rows have non-positive budget"
    assert (df["revenue"] > 0).all(), "Some rows have non-positive revenue"

    # Year range.
    yr = df["release_year_parsed"]
    assert yr.notna().all(), "release_year_parsed has nulls"
    assert (yr >= 1900).all(), f"Year < 1900 found (min={yr.min()})"
    assert (yr <= 2025).all(), f"Year > 2025 found (max={yr.max()})"

    # Log transforms — non-null, finite, sensible.
    for col in ("log_budget", "log_revenue"):
        assert df[col].notna().all(), f"{col} has nulls"
        assert np.isfinite(df[col]).all(), f"{col} has inf/-inf"
        assert (df[col] > 0).all(), f"{col} has non-positive values"

    # Genres.
    assert df["primary_genre"].notna().all(), "primary_genre has nulls"
    assert df["primary_genre_bucketed"].notna().all(), "primary_genre_bucketed has nulls"

    # Screenplay-structural columns.
    assert (df["n_scenes"] > 0).all(), "Films with 0 scenes in corpus"
    assert (df["n_dialogue_lines"] >= 0).all(), "Negative dialogue line counts"
    for ratio_col in ("dialogue_to_action_ratio", "dialogue_to_total_text_ratio"):
        assert df[ratio_col].between(0, 1).all(), f"{ratio_col} outside [0, 1]"

    # Data-quality flag: present and boolean. Informational column,
    # so we don't assert anything about its values.
    assert "data_quality_flag" in df.columns, "data_quality_flag column missing"
    assert df["data_quality_flag"].dtype == bool, (
        "data_quality_flag is not boolean dtype"
    )

    logger.info("All hard assertions passed (%d films, %d columns)",
                len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    logger.debug("Wrote %s", path)


def plot_year_distribution(df: pd.DataFrame, out_path) -> None:
    years = df["release_year_parsed"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2),
            edgecolor="black", alpha=0.85)
    ax.set(xlabel="Release year", ylabel="Films",
           title=f"Phase 2 corpus year distribution (N={len(df):,})")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    _save(fig, out_path)


def plot_genre_distribution(df: pd.DataFrame, out_path) -> None:
    exploded = df.explode("genres_bucketed")
    counts = exploded["genres_bucketed"].dropna().value_counts()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1],
            color="steelblue", edgecolor="black")
    ax.set(xlabel="Films (multi-genre films counted in each bar)",
           title=f"Phase 2 corpus genre distribution (bucketed, N={len(df):,})")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    _save(fig, out_path)


def plot_budget_revenue(df: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, color, ax_raw, ax_log in (
        ("budget",  "steelblue", axes[0, 0], axes[1, 0]),
        ("revenue", "indianred", axes[0, 1], axes[1, 1]),
    ):
        vals = df[col].astype(float)
        ax_raw.hist(vals / 1e6, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_raw.set(xlabel=f"{col.capitalize()} ($M)", ylabel="Films",
                   title=f"{col.capitalize()} (raw)")
        ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

        log_vals = np.log10(vals.clip(lower=1))
        ax_log.hist(log_vals, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax_log.set(xlabel=f"log10({col.capitalize()})", ylabel="Films",
                   title=f"{col.capitalize()} (log10)")
        ax_log.grid(axis="y", linestyle=":", alpha=0.6)
    fig.suptitle(f"Phase 2 corpus: budget and revenue (N={len(df):,})")
    _save(fig, out_path)


def plot_rating_roi_length(df: pd.DataFrame, out_path) -> None:
    roi = df["revenue"] / df["budget"]
    roi_log = np.log10(roi.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=1e-3))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(df["effective_rating"], bins=np.arange(0, 10.5, 0.25),
                 edgecolor="black", alpha=0.85)
    axes[0].set(xlabel="effective_rating", ylabel="Films", title="Rating")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1].hist(roi_log, bins=60, color="darkgreen",
                 edgecolor="black", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.7,
                    label="break-even")
    axes[1].set(xlabel="log10(ROI)", ylabel="Films", title="ROI")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # Use total_dialogue_chars + total_action_chars as a richer
    # "screenplay length" proxy than the raw script string length.
    total_chars = df["total_dialogue_chars"] + df["total_action_chars"]
    axes[2].hist(total_chars / 1000.0, bins=40,
                 color="orange", edgecolor="black", alpha=0.85)
    axes[2].set(xlabel="Total parsed text length (k chars)",
                ylabel="Films", title="Screenplay length (parsed)")
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"Phase 2 corpus: rating, ROI, screenplay length (N={len(df):,})")
    _save(fig, out_path)


def plot_screenplay_structure(df: pd.DataFrame, out_path) -> None:
    """New in Phase 2: distributions of the screenplay-structural features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].hist(df["n_scenes"], bins=40, color="purple",
                    edgecolor="black", alpha=0.85)
    axes[0, 0].set(xlabel="Number of scenes", ylabel="Films",
                   title="n_scenes")
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[0, 1].hist(df["n_unique_characters"], bins=40, color="teal",
                    edgecolor="black", alpha=0.85)
    axes[0, 1].set(xlabel="Number of unique characters", ylabel="Films",
                   title="n_unique_characters")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1, 0].hist(df["n_dialogue_lines"], bins=40, color="darkorange",
                    edgecolor="black", alpha=0.85)
    axes[1, 0].set(xlabel="Number of dialogue lines", ylabel="Films",
                   title="n_dialogue_lines")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1, 1].hist(df["dialogue_to_total_text_ratio"], bins=40, color="firebrick",
                    edgecolor="black", alpha=0.85)
    axes[1, 1].set(xlabel="dialogue / (dialogue + action) chars", ylabel="Films",
                   title="dialogue_to_total_text_ratio")
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"Phase 2 corpus: screenplay-structural features (N={len(df):,})")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summary_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact numeric summary of the post-filter corpus."""
    roi = df["revenue"] / df["budget"]
    rows = [
        ("films_total", int(len(df))),
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
        ("scenes_median", int(df["n_scenes"].median())),
        ("scenes_mean", round(float(df["n_scenes"].mean()), 1)),
        ("unique_characters_median", int(df["n_unique_characters"].median())),
        ("dialogue_lines_median", int(df["n_dialogue_lines"].median())),
        ("dialogue_to_total_text_ratio_mean", round(float(df["dialogue_to_total_text_ratio"].mean()), 3)),
        ("films_with_parse_warnings", int((df["parse_warning_count"] > 0).sum())),
        ("films_data_quality_flagged",
         int(df["data_quality_flag"].sum()) if "data_quality_flag" in df.columns else 0),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def per_decade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and medians by decade, useful for Phase 4 era-stratification planning."""
    decade = (df["release_year_parsed"] // 10 * 10).astype("Int64")
    grouped = df.assign(decade=decade).groupby("decade")
    out = grouped.agg(
        n_films=("imdb_id", "count"),
        budget_median_M=("budget", lambda s: round(s.median() / 1e6, 2)),
        revenue_median_M=("revenue", lambda s: round(s.median() / 1e6, 2)),
        rating_mean=("effective_rating", lambda s: round(s.mean(), 2)),
        scenes_median=("n_scenes", "median"),
        dialogue_lines_median=("n_dialogue_lines", "median"),
    ).reset_index()
    return out


def per_genre_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and medians by primary_genre_bucketed."""
    grouped = df.groupby("primary_genre_bucketed")
    out = grouped.agg(
        n_films=("imdb_id", "count"),
        budget_median_M=("budget", lambda s: round(s.median() / 1e6, 2)),
        revenue_median_M=("revenue", lambda s: round(s.median() / 1e6, 2)),
        rating_mean=("effective_rating", lambda s: round(s.mean(), 2)),
        roi_median=("revenue", lambda s: round((s / df.loc[s.index, "budget"]).median(), 2)),
    ).reset_index()
    return out.sort_values("n_films", ascending=False)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run validation, generate Phase 2 figures and tables, return the corpus."""
    paths.ensure_dirs()

    df = load_processed_corpus()
    hard_asserts(df)

    # Plots.
    figs = paths.REPORTS_FIGURES_DIR
    plot_year_distribution(df, figs / "phase2_year_distribution.png")
    plot_genre_distribution(df, figs / "phase2_genre_distribution.png")
    plot_budget_revenue(df, figs / "phase2_budget_revenue_distribution.png")
    plot_rating_roi_length(df, figs / "phase2_rating_roi_length.png")
    plot_screenplay_structure(df, figs / "phase2_screenplay_structure.png")
    logger.info("Saved 5 figures to reports/figures/")

    # Tables.
    tables = paths.REPORTS_TABLES_DIR
    summary_metrics_table(df).to_csv(tables / "phase2_summary_metrics.csv", index=False)
    per_decade_table(df).to_csv(tables / "phase2_per_decade.csv", index=False)
    per_genre_table(df).to_csv(tables / "phase2_per_genre.csv", index=False)
    logger.info("Saved 3 summary tables to reports/tables/")

    # Print a clean human-readable summary.
    print("\n=== Phase 2 corpus validation passed ===")
    print(summary_metrics_table(df).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
