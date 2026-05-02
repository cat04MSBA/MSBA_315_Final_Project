"""Phase 1 — Join MovieSum and the ratings dataset on IMDb ID.

Both sources carry IMDb IDs natively (MovieSum exposes them on every
screenplay row; the ratings dataset has an ``imdb_id`` column). The
join is therefore a direct exact-ID merge — no normalization, no fuzzy
matching, no external bridge.

The script:

1. Loads MovieSum and deduplicates on IMDb ID (12 same-IMDb-ID pairs
   exist in MovieSum; we keep the longest-script row per ID — that's
   typically the more-complete draft).
2. Loads the ratings dataset (column-selective; see ``load_ratings``).
3. Deduplicates the ratings dataset on IMDb ID (a small fraction of
   films appear twice under different TMDB IDs — alternate cuts /
   regional releases — keep the row with the highest ``vote_count``).
4. Merges them on ``imdb_id`` and saves to
   ``data/interim/phase1_joined_corpus.parquet``.

Run from the project root::

    python -m src.data.join_corpus
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python3 -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.load_moviesum import load_moviesum
from src.data.load_ratings import load_ratings
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns we attach from the ratings dataset onto each MovieSum row.
RATINGS_COLUMNS_TO_ATTACH: tuple[str, ...] = (
    "imdb_id",
    "id",                    # TMDB id
    "title",
    "original_title",
    "release_year_parsed",
    "release_date",
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "IMDB_Rating",
    "AverageRating",
    "Meta_score",
    "popularity",
    "status",
    "genres_parsed",
    "Director",
    "production_companies",
    "production_countries",
)


def dedupe_moviesum(moviesum: pd.DataFrame) -> pd.DataFrame:
    """Collapse MovieSum's same-IMDb-ID duplicate pairs to one row per ID.

    Keeps the row with the longest ``script_char_len`` per IMDb ID — the
    intuition being that longer screenplays are usually more-complete
    drafts. Phase 1 found 12 such pairs out of 2,200 rows; cell-level
    review (``review_duplicates.py``) confirmed they are alternate
    titles or alternate drafts of the same film.
    """
    before = len(moviesum)
    sorted_df = moviesum.sort_values("script_char_len", ascending=False)
    dedup = sorted_df.drop_duplicates(subset="imdb_id", keep="first").reset_index(drop=True)
    logger.info("Dedup MovieSum by IMDb ID: %d → %d (%d collapsed)",
                before, len(dedup), before - len(dedup))
    return dedup


def dedupe_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per IMDb ID in the ratings dataset.

    Some films appear twice under different TMDB IDs (alternate cuts,
    regional releases). Keep the row with the higher ``vote_count`` —
    the better-known variant. Rows without IMDb IDs are dropped (they
    can't be joined anyway).
    """
    before = len(ratings)
    keepable = ratings.dropna(subset=["imdb_id"]).copy()
    keepable = (
        keepable.sort_values("vote_count", ascending=False)
                .drop_duplicates(subset="imdb_id", keep="first")
                .reset_index(drop=True)
    )
    logger.info(
        "Dedup ratings by IMDb ID: %d → %d (%d dropped, %d had no IMDb ID)",
        before, len(keepable), before - len(keepable),
        int(ratings["imdb_id"].isna().sum()),
    )
    return keepable


def join_corpora(
    moviesum: pd.DataFrame,
    ratings: pd.DataFrame,
    columns_to_attach: tuple[str, ...] = RATINGS_COLUMNS_TO_ATTACH,
) -> pd.DataFrame:
    """Left-merge MovieSum onto a column-selected slice of the ratings dataset.

    Parameters
    ----------
    moviesum
        Deduplicated MovieSum DataFrame (one row per IMDb ID).
    ratings
        Deduplicated ratings DataFrame (one row per IMDb ID).
    columns_to_attach
        Columns from ``ratings`` to keep in the joined output.
        Must include ``imdb_id`` (the join key).

    Returns
    -------
    pandas.DataFrame
        MovieSum rows with the selected ratings columns merged in.
        Films that don't match leave the ratings columns as NaN.
    """
    if "imdb_id" not in columns_to_attach:
        raise ValueError("columns_to_attach must include 'imdb_id'")
    return moviesum.merge(
        ratings[list(columns_to_attach)], on="imdb_id", how="left",
        suffixes=("_ms", "_rt"),
    )


def main() -> pd.DataFrame:
    """Run the Phase 1 join end-to-end and save to ``data/interim/``.

    Returns the joined DataFrame for interactive use.
    """
    paths.ensure_dirs()

    moviesum = load_moviesum(include_script=True)
    moviesum = dedupe_moviesum(moviesum)

    ratings = load_ratings()
    ratings = dedupe_ratings(ratings)

    joined = join_corpora(moviesum, ratings)

    matched_mask = joined["id"].notna()
    has_budget   = joined["budget"].fillna(0) > 0
    has_revenue  = joined["revenue"].fillna(0) > 0
    has_rating   = (joined["vote_average"].fillna(0) > 0) | (joined["IMDB_Rating"].fillna(0) > 0)

    n_total = len(joined)
    counts = {
        "moviesum_deduped": n_total,
        "matched_to_ratings": int(matched_mask.sum()),
        "matched_with_budget": int((matched_mask & has_budget).sum()),
        "matched_with_revenue": int((matched_mask & has_revenue).sum()),
        "matched_with_budget_and_revenue": int((matched_mask & has_budget & has_revenue).sum()),
        "matched_with_all_four_signals": int((matched_mask & has_budget & has_revenue & has_rating).sum()),
    }
    counts_table = pd.DataFrame(list(counts.items()), columns=["metric", "count"])

    # Save artifacts.
    out_parquet = paths.DATA_INTERIM_DIR / "phase1_joined_corpus.parquet"
    joined.to_parquet(out_parquet, index=False)
    logger.debug("Wrote %s", out_parquet)

    out_counts = paths.REPORTS_TABLES_DIR / "phase1_join_counts.csv"
    counts_table.to_csv(out_counts, index=False)
    logger.debug("Wrote %s", out_counts)

    # Note: ``title`` exists on both sides of the merge (MovieSum's title-with-year-stripped
    # vs. ratings' canonical title), so they get split into ``title_ms`` / ``title_rt``
    # by the merge's ``suffixes`` argument. Use the MovieSum side here for human readability.
    unmatched = joined.loc[~matched_mask, ["imdb_id", "movie_name", "title_ms", "year_in_title"]]
    out_unmatched = paths.REPORTS_TABLES_DIR / "phase1_unmatched_moviesum_titles.csv"
    unmatched.to_csv(out_unmatched, index=False)
    logger.debug("Wrote %s", out_unmatched)

    logger.info("Saved 1 parquet + 2 CSVs (corpus, counts, %d unmatched titles)",
                len(unmatched))

    print("\n=== MovieSum × ratings join — Phase 1 summary ===")
    print(counts_table.to_string(index=False))
    return joined


if __name__ == "__main__":
    main()
