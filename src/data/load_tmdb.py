"""Loader for the TMDB 5000 Movies dataset.

The raw CSV (``tmdb_5000_movies.csv``) lives at
``data/raw/ratings_data/tmdb_5000_movies.csv``. Its source is
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.

Phase 1 only needs a thin loader: parse the JSON-encoded list columns we
actually care about (``genres``), derive a clean ``release_year``, and
return the full DataFrame. Filtering by budget/revenue happens in the
exploration script — the loader stays neutral so other phases can reuse it.

Per ``PROJECT_CONTEXT.txt`` Section 4, this dataset does **not** carry an
``imdb_id`` column natively; the join to MovieSum goes from MovieSum's IMDb
ID to TMDB via title + year (or another bridge), which is wired in Task 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TMDB_CSV: Path = paths.DATA_RAW_DIR / "ratings_data" / "tmdb_5000_movies.csv"


def _parse_genre_names(raw: str | float | None) -> list[str]:
    """Parse the JSON-encoded ``genres`` cell into a list of genre name strings.

    The raw column stores JSON like ``[{"id": 28, "name": "Action"}, ...]``.
    Empty arrays and missing values both become ``[]``.

    Parameters
    ----------
    raw
        The raw cell value as it comes out of pandas (a JSON string, ``""``,
        ``"[]"``, or ``NaN``).

    Returns
    -------
    list[str]
        Genre names in the order TMDB provided them.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # The TMDB dump is well-formed JSON in practice; warn loudly if not.
        logger.warning("Failed to parse genres cell: %r", raw[:80])
        return []
    return [g["name"] for g in parsed if isinstance(g, dict) and "name" in g]


def load_tmdb(
    csv_path: Path | str = DEFAULT_TMDB_CSV,
) -> pd.DataFrame:
    """Load the TMDB 5000 movies CSV into a tidy DataFrame.

    Adds two derived columns on top of the raw CSV:

    - ``release_year`` (Int64): year extracted from ``release_date``,
      ``<NA>`` where the date is missing or unparseable.
    - ``genre_names`` (object): a Python list of genre name strings.

    Parameters
    ----------
    csv_path
        Path to ``tmdb_5000_movies.csv``. Defaults to the location used by
        Phase 1 under ``data/raw/ratings_data/``.

    Returns
    -------
    pandas.DataFrame
        One row per film, all original columns preserved, plus the two
        derived columns above.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at ``csv_path``.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"TMDB CSV not found at {csv_path}")

    logger.info("Loading TMDB 5000 from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Parse release_date -> nullable integer year.
    parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = parsed_dates.dt.year.astype("Int64")

    df["genre_names"] = df["genres"].map(_parse_genre_names)

    logger.info(
        "Loaded TMDB: %d rows, %d unique TMDB ids",
        len(df),
        df["id"].nunique(),
    )
    return df


def summarize_tmdb(df: pd.DataFrame) -> dict[str, int]:
    """Compute the headline counts from ``PROJECT_CONTEXT.txt`` Section 5.

    These are the numbers we expect Phase 1 to reproduce as a sanity check:
    4,803 / 3,766 / 3,376 / 3,229.

    Parameters
    ----------
    df
        Output of :func:`load_tmdb`.

    Returns
    -------
    dict[str, int]
        Keys: ``total``, ``with_budget``, ``with_revenue``, ``with_both``.
    """
    return {
        "total": int(len(df)),
        "with_budget": int((df["budget"] > 0).sum()),
        "with_revenue": int((df["revenue"] > 0).sum()),
        "with_both": int(((df["budget"] > 0) & (df["revenue"] > 0)).sum()),
    }
