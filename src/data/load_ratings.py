"""Loader for the project's primary ratings / metadata dataset.

The file ``IMDB TMDB Movie Metadata Big Dataset (1M).csv`` lives at
``data/raw/ratings_data/`` (~950 MB, ~1.07M rows). It is a Kaggle-published
join of TMDB and IMDb metadata, carrying both an ``imdb_id`` column
(``tt``-prefixed string) and a numeric TMDB ``id`` natively — so the
MovieSum→ratings join is a direct exact-ID merge.

Per-row columns of interest for this project (out of 42 raw columns):

- ``imdb_id``, ``id``        — the join keys (IMDb tt-string, TMDB int)
- ``title``, ``original_title``
- ``release_date`` → derived ``release_year_parsed`` (Int64)
- ``budget``, ``revenue``    — financial outcomes (USD, integer)
- ``runtime``                — minutes
- ``vote_average``, ``vote_count``  — TMDB rating + popularity proxy
- ``IMDB_Rating``            — IMDb's own 0-10 rating (more populated
                                than TMDB's for our matched subset)
- ``AverageRating``          — a smoothed/external rating
- ``Meta_score``             — Metacritic 0-100
- ``popularity``, ``status``
- ``genres_list`` → derived ``genres_parsed`` (list[str])
- ``Director``, ``production_companies``, ``production_countries``

This loader is intentionally column-selective; the full file uses ~3-4 GB
of memory. Pass ``columns=None`` to ``load_ratings`` if a downstream task
needs the rest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CSV: Path = (
    paths.DATA_RAW_DIR / "ratings_data" / "IMDB TMDB Movie Metadata Big Dataset (1M).csv"
)

# Columns we read by default. The heavy text columns (overview,
# all_combined_keywords, Cast_list, Poster_Link, etc.) are dropped
# unless the caller asks for them by passing ``columns=None``.
COLUMNS_OF_INTEREST: tuple[str, ...] = (
    "id",
    "imdb_id",
    "title",
    "original_title",
    "release_date",
    "release_year",
    "vote_average",
    "vote_count",
    "IMDB_Rating",
    "AverageRating",
    "Meta_score",
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "status",
    "genres_list",
    "Director",
    "production_companies",
    "production_countries",
)

# Pattern for the stringified Python list values stored in ``genres_list``
# and a few other columns: e.g. ``"['Action', 'Drama']"``.
_LIST_LIKE_RE = re.compile(r"^\[.*\]$")


def _parse_list_column(raw: str | float | None) -> list[str]:
    """Parse a stringified Python list (e.g. ``"['a', 'b']"``) into ``["a", "b"]``.

    Robust to ``NaN``, empty strings, and malformed values — returns
    ``[]`` rather than raising, so the caller can mask on emptiness.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or not _LIST_LIKE_RE.match(s):
        return []
    # The dataset's stringified lists use single quotes; swap to double for
    # ``json.loads`` (low risk because element strings don't contain raw
    # double-quote characters in practice).
    try:
        return [str(x) for x in json.loads(s.replace("'", '"'))]
    except json.JSONDecodeError:
        # Fallback: strip brackets and split on commas. Crude but better
        # than dropping a row's genre info entirely.
        inner = s.strip("[]")
        return [
            piece.strip().strip("'").strip('"')
            for piece in inner.split(",")
            if piece.strip()
        ]


def load_ratings(
    csv_path: Path | str = DEFAULT_CSV,
    columns: tuple[str, ...] | None = COLUMNS_OF_INTEREST,
) -> pd.DataFrame:
    """Load the IMDb-TMDB metadata CSV into a tidy DataFrame.

    Parameters
    ----------
    csv_path
        Path to the CSV. Defaults to the Phase 1 location.
    columns
        Subset of columns to read. ``None`` reads everything (heavy:
        ~3-4 GB memory). Defaults to :data:`COLUMNS_OF_INTEREST`.

    Returns
    -------
    pandas.DataFrame
        One row per film. Adds two derived columns on top of the raw CSV:

        - ``release_year_parsed`` (Int64) — preferred over the raw
          ``release_year`` column (which is float and often NaN even
          when ``release_date`` is well-formed).
        - ``genres_parsed`` (object) — list of genre name strings.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ratings CSV not found at {csv_path}")

    logger.info("Loading ratings dataset (~20-30s)")
    logger.debug("Reading CSV from %s", csv_path)
    read_kwargs: dict = {"low_memory": False}
    if columns is not None:
        read_kwargs["usecols"] = list(columns)
    df = pd.read_csv(csv_path, **read_kwargs)

    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year_parsed"] = parsed_dates.dt.year.astype("Int64")

    if "genres_list" in df.columns:
        df["genres_parsed"] = df["genres_list"].map(_parse_list_column)
    else:
        df["genres_parsed"] = [[] for _ in range(len(df))]

    logger.info(
        "Loaded ratings: %s rows | %s unique IMDb IDs | %s unique TMDB ids",
        f"{len(df):,}",
        f"{df['imdb_id'].nunique(dropna=True):,}",
        f"{df['id'].nunique(dropna=True):,}",
    )
    return df


def summarize_ratings(df: pd.DataFrame) -> dict[str, int]:
    """Headline counts: total / with-budget / with-revenue / with-both / with-imdb_id."""
    return {
        "total": int(len(df)),
        "with_budget": int((df["budget"] > 0).sum()),
        "with_revenue": int((df["revenue"] > 0).sum()),
        "with_both": int(((df["budget"] > 0) & (df["revenue"] > 0)).sum()),
        "with_imdb_id": int(df["imdb_id"].notna().sum()),
    }
