"""Loader for the MovieSum screenplay dataset (Saxena & Keller, ACL 2024).

The raw dataset on disk lives under ``data/raw/script_data/`` as three
JSONL files (``train.jsonl``, ``val.jsonl``, ``test.jsonl``) with one
movie per line. Each line has the keys:

- ``movie_name`` — title with ``_YYYY`` year suffix (e.g.
  ``"A Nightmare on Elm Street 3: Dream Warriors_1987"``)
- ``imdb_id``   — ``tt``-prefixed IMDb ID (e.g. ``"tt0093629"``)
- ``script``    — the full screenplay as an XML string
                  (``<script><scene>...</scene>...</script>``)
- ``summary``   — Wikipedia-derived plot summary (string)

Per ``PROJECT_CONTEXT.txt`` the original train/val/test split serves the
upstream summarization task and is **not** the split this project uses.
The loader concatenates all three splits into a single corpus and
preserves the origin in an ``origin_split`` column for traceability;
Phase 3 will draw a fresh, project-specific train/calibration/test split.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MOVIESUM_DIR: Path = paths.DATA_RAW_DIR / "script_data"
SPLIT_FILES: tuple[str, ...] = ("train.jsonl", "val.jsonl", "test.jsonl")

# IMDb tt-IDs are tt + 7-10 digits (modern tt-IDs are 7-8; allow 10 for forward compat).
IMDB_ID_PATTERN = re.compile(r"^tt\d{7,10}$")
# Movie names are formatted "Title with spaces and punctuation_YYYY".
TITLE_YEAR_PATTERN = re.compile(r"^(?P<title>.+)_(?P<year>\d{4})$")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield decoded JSON objects from a JSONL file, one per line.

    Skips blank lines silently; raises on malformed JSON so corruption is
    not hidden.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}: {exc}"
                ) from exc


def _split_title_year(movie_name: str) -> tuple[str, int | None]:
    """Pull the ``_YYYY`` suffix off a MovieSum ``movie_name``.

    Returns ``(title, year)``; ``year`` is ``None`` when the suffix is
    missing or unparseable.
    """
    match = TITLE_YEAR_PATTERN.match(movie_name)
    if not match:
        return movie_name, None
    return match.group("title"), int(match.group("year"))


def load_moviesum(
    moviesum_dir: Path | str = DEFAULT_MOVIESUM_DIR,
    include_script: bool = True,
) -> pd.DataFrame:
    """Load all MovieSum splits and return a single DataFrame.

    Parameters
    ----------
    moviesum_dir
        Directory holding ``train.jsonl``, ``val.jsonl``, ``test.jsonl``.
    include_script
        When False, drop the heavy ``script`` column to save memory for
        analyses that only need IDs and lengths. Default True (full data).

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``imdb_id``          (str)   — ``tt``-prefixed
        - ``movie_name``       (str)   — original "Title_YYYY" string
        - ``title``            (str)   — name with the year suffix stripped
        - ``year_in_title``    (Int64) — year parsed from ``movie_name``
        - ``script``           (str)   — full XML, present iff ``include_script``
        - ``script_char_len``  (int)   — character length of ``script``
        - ``summary``          (str)   — Wikipedia plot summary
        - ``origin_split``     (str)   — ``"train"|"val"|"test"`` (MovieSum's split)

    Raises
    ------
    FileNotFoundError
        If any expected split file is missing.
    """
    moviesum_dir = Path(moviesum_dir)
    logger.info("Loading MovieSum (train + val + test, ~5-10s)")
    rows: list[dict] = []
    for split_name in SPLIT_FILES:
        path = moviesum_dir / split_name
        if not path.is_file():
            raise FileNotFoundError(f"MovieSum split file missing: {path}")
        logger.debug("Reading split %s", split_name)
        split_label = split_name.removesuffix(".jsonl")
        for obj in _iter_jsonl(path):
            title, year = _split_title_year(obj.get("movie_name", ""))
            row = {
                "imdb_id": obj.get("imdb_id"),
                "movie_name": obj.get("movie_name"),
                "title": title,
                "year_in_title": year,
                "script_char_len": len(obj.get("script", "") or ""),
                "summary": obj.get("summary"),
                "origin_split": split_label,
            }
            if include_script:
                row["script"] = obj.get("script")
            rows.append(row)

    df = pd.DataFrame(rows)
    df["year_in_title"] = df["year_in_title"].astype("Int64")
    logger.info("Loaded MovieSum: %s screenplays", f"{len(df):,}")
    return df


def imdb_id_validity(df: pd.DataFrame) -> dict[str, int]:
    """Count how many ``imdb_id`` values match the ``tt\\d{7,10}`` pattern."""
    valid_mask = df["imdb_id"].fillna("").map(lambda s: bool(IMDB_ID_PATTERN.match(s)))
    return {
        "total": int(len(df)),
        "valid_imdb_id": int(valid_mask.sum()),
        "invalid_or_missing": int((~valid_mask).sum()),
        "unique_valid_ids": int(df.loc[valid_mask, "imdb_id"].nunique()),
    }
