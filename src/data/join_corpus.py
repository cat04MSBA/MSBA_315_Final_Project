"""Phase 1, Task 4 — Join MovieSum and TMDB on title + year.

TMDB 5000 has no ``imdb_id`` column (confirmed in Task 2), so the join
goes from MovieSum (which carries IMDb IDs natively) to TMDB via
normalized title + release year, with a ±1-year tolerance to absorb
festival-vs-theatrical / regional release differences.

Strategy:

1. Normalize titles in both datasets: lowercase, strip punctuation,
   collapse whitespace, drop leading articles (``the/a/an``).
2. Index TMDB by ``(normalized_title, release_year)`` and by
   ``(normalized_title, release_year ± 1)``.
3. For each MovieSum row (deduplicated by IMDb ID), look up TMDB.
4. Save the joined corpus to ``data/interim/`` and a small audit table.

Run from the project root:

    python -m src.data.join_corpus
"""

from __future__ import annotations

import re
import string
from typing import Iterable

import pandas as pd

from src.data.load_moviesum import load_moviesum
from src.data.load_tmdb import load_tmdb
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Articles to strip from the start of a title before normalization.
# Kept conservative — only the most common English articles, since stripping
# more aggressively risks merging genuinely-different films.
LEADING_ARTICLES = ("the ", "a ", "an ")

# Films to spot-check by hand from the join output (logged at runtime).
SPOT_CHECK_SAMPLE_SIZE = 20
SPOT_CHECK_SEED = 42


def _normalize_title(title: str | float | None) -> str:
    """Normalize a film title for matching across datasets.

    Steps: lowercase, replace ``&`` with ``and``, drop punctuation, collapse
    whitespace, strip a leading article.

    Returns ``""`` for missing values; the caller should treat empty
    normalized titles as unmatchable rather than letting them collide.
    """
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    s = str(title).lower().replace("&", "and")
    # Replace punctuation with whitespace so word boundaries are preserved
    # (e.g., "X-Men: Days of Future Past" -> "x men days of future past").
    table = str.maketrans({c: " " for c in string.punctuation})
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s).strip()
    for article in LEADING_ARTICLES:
        if s.startswith(article):
            s = s[len(article):]
            break
    return s


def _build_tmdb_lookup(
    tmdb: pd.DataFrame,
) -> dict[tuple[str, int], list[int]]:
    """Build a ``(normalized_title, year) -> [tmdb_row_index, ...]`` index.

    A list (not a single index) is the value because title+year can collide
    in TMDB itself (rare — e.g., reboots) and we want to detect ambiguity.
    """
    lookup: dict[tuple[str, int], list[int]] = {}
    for idx, row in tmdb[["title", "original_title", "release_year"]].iterrows():
        year = row["release_year"]
        if pd.isna(year):
            continue
        year_int = int(year)
        # Index by both the localized title and the original_title to widen
        # the match set without sacrificing precision.
        for raw_title in (row["title"], row["original_title"]):
            key = (_normalize_title(raw_title), year_int)
            if not key[0]:
                continue
            lookup.setdefault(key, []).append(idx)
    return lookup


def _lookup_with_year_tolerance(
    lookup: dict[tuple[str, int], list[int]],
    norm_title: str,
    year: int,
    tolerance: Iterable[int] = (0, -1, 1),
) -> tuple[list[int], int | None]:
    """Look up TMDB indices for a title across a small year window.

    Returns ``(indices, year_offset)`` where ``year_offset`` is the offset
    that produced the match (0 / -1 / +1), or ``None`` if no match was
    found.
    """
    for offset in tolerance:
        key = (norm_title, year + offset)
        if key in lookup:
            return lookup[key], offset
    return [], None


def _dedupe_moviesum(moviesum: pd.DataFrame) -> pd.DataFrame:
    """Collapse MovieSum's 12 same-IMDb-ID duplicate pairs to one row each.

    Keeps the row with the longest ``script_char_len`` per IMDb ID — the
    intuition being that longer screenplays correspond to more-completed
    drafts. Logs which IDs were collapsed.
    """
    before = len(moviesum)
    moviesum_sorted = moviesum.sort_values("script_char_len", ascending=False)
    dedup = moviesum_sorted.drop_duplicates(subset="imdb_id", keep="first").reset_index(drop=True)
    after = len(dedup)
    logger.info("Dedup MovieSum by IMDb ID: %d -> %d rows (%d collapsed)", before, after, before - after)
    return dedup


def _attach_tmdb(
    moviesum: pd.DataFrame, tmdb: pd.DataFrame, lookup: dict[tuple[str, int], list[int]]
) -> pd.DataFrame:
    """For each MovieSum row, look up its TMDB match and attach selected columns.

    Adds: ``tmdb_id``, ``tmdb_title``, ``tmdb_release_year``, ``budget``,
    ``revenue``, ``runtime``, ``vote_average``, ``vote_count``,
    ``tmdb_genre_names``, ``join_year_offset``, ``join_strategy``,
    ``join_match_count``.
    """
    enriched_rows: list[dict] = []

    for _, row in moviesum.iterrows():
        norm_title = _normalize_title(row["title"])
        year = row["year_in_title"]
        if pd.isna(year) or not norm_title:
            enriched_rows.append({"join_strategy": "skipped_no_title_or_year", "join_match_count": 0})
            continue

        indices, offset = _lookup_with_year_tolerance(lookup, norm_title, int(year))
        if not indices:
            enriched_rows.append({"join_strategy": "no_match", "join_match_count": 0, "join_year_offset": None})
            continue

        # When a title+year resolves to multiple TMDB rows, pick the one with
        # the highest vote_count (the better-known one) — rare collision case.
        if len(indices) > 1:
            sub = tmdb.loc[indices].sort_values("vote_count", ascending=False)
            best_idx = sub.index[0]
        else:
            best_idx = indices[0]
        tmdb_row = tmdb.loc[best_idx]

        enriched_rows.append(
            {
                "tmdb_id": int(tmdb_row["id"]),
                "tmdb_title": tmdb_row["title"],
                "tmdb_release_year": int(tmdb_row["release_year"]) if pd.notna(tmdb_row["release_year"]) else None,
                "budget": tmdb_row["budget"],
                "revenue": tmdb_row["revenue"],
                "runtime": tmdb_row["runtime"],
                "vote_average": tmdb_row["vote_average"],
                "vote_count": tmdb_row["vote_count"],
                "tmdb_genre_names": tmdb_row["genre_names"],
                "join_year_offset": offset,
                "join_strategy": "title_year_exact_norm",
                "join_match_count": len(indices),
            }
        )

    enriched = pd.DataFrame(enriched_rows, index=moviesum.index)
    return pd.concat([moviesum.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)


def _spot_check_matches(joined: pd.DataFrame, n: int = SPOT_CHECK_SAMPLE_SIZE) -> None:
    """Log N random MovieSum→TMDB matches for human verification."""
    matched = joined[joined["tmdb_id"].notna()]
    if len(matched) == 0:
        logger.warning("No matched rows to spot-check.")
        return
    sample = matched.sample(min(n, len(matched)), random_state=SPOT_CHECK_SEED)
    logger.info("Spot-checking %d random MovieSum->TMDB matches:", len(sample))
    for _, row in sample.iterrows():
        logger.info(
            "  %s | MovieSum=%r (%s) -> TMDB=%r (%s, off=%s)",
            row["imdb_id"], row["movie_name"], row["year_in_title"],
            row["tmdb_title"], row["tmdb_release_year"], row["join_year_offset"],
        )


def main() -> None:
    paths.ensure_dirs()

    moviesum = load_moviesum(include_script=True)
    moviesum = _dedupe_moviesum(moviesum)
    tmdb = load_tmdb()

    lookup = _build_tmdb_lookup(tmdb)
    logger.info("Built TMDB title-year lookup: %d unique keys", len(lookup))

    joined = _attach_tmdb(moviesum, tmdb, lookup)

    # Headline counts.
    matched_mask = joined["tmdb_id"].notna()
    has_budget = joined["budget"].fillna(0) > 0
    has_revenue = joined["revenue"].fillna(0) > 0
    has_rating = joined["vote_average"].fillna(0) > 0

    n_total = len(joined)
    n_matched = int(matched_mask.sum())
    n_with_budget = int((matched_mask & has_budget).sum())
    n_with_revenue = int((matched_mask & has_revenue).sum())
    n_with_both = int((matched_mask & has_budget & has_revenue).sum())
    n_with_all_four = int((matched_mask & has_budget & has_revenue & has_rating).sum())

    logger.info("=== Phase 1 join headline counts ===")
    logger.info("MovieSum (deduped):                 %d", n_total)
    logger.info("Matched to TMDB:                    %d (%.1f%%)", n_matched, 100 * n_matched / n_total)
    logger.info("Match + budget>0:                   %d", n_with_budget)
    logger.info("Match + revenue>0:                  %d", n_with_revenue)
    logger.info("Match + budget>0 AND revenue>0:     %d", n_with_both)
    logger.info("Match + budget + revenue + rating:  %d", n_with_all_four)

    # Year-offset breakdown of matches.
    offset_counts = joined.loc[matched_mask, "join_year_offset"].value_counts(dropna=False).to_dict()
    logger.info("Year-offset breakdown of matches: %s", offset_counts)

    _spot_check_matches(joined)

    # Save unmatched titles for human review.
    unmatched = joined[~matched_mask][["imdb_id", "movie_name", "title", "year_in_title", "join_strategy"]]
    unmatched_path = paths.REPORTS_TABLES_DIR / "phase1_unmatched_moviesum_titles.csv"
    unmatched.to_csv(unmatched_path, index=False)
    logger.info("Saved %d unmatched MovieSum titles to %s", len(unmatched), unmatched_path)

    # Save the full joined corpus (Phase 6 of the brief — this is the Phase 1
    # working dataset, to be rebuilt cleanly in Phase 2).
    joined_path = paths.DATA_INTERIM_DIR / "phase1_joined_corpus.parquet"
    # Genre lists need to be plain Python objects for parquet roundtrip.
    joined.to_parquet(joined_path, index=False)
    logger.info("Saved joined corpus to %s (%d rows)", joined_path, len(joined))

    # Save a small headline-counts table for the phase summary.
    counts_table = pd.DataFrame(
        {
            "metric": [
                "moviesum_deduped",
                "matched_to_tmdb",
                "matched_with_budget",
                "matched_with_revenue",
                "matched_with_budget_and_revenue",
                "matched_with_all_four_signals",
            ],
            "count": [n_total, n_matched, n_with_budget, n_with_revenue, n_with_both, n_with_all_four],
        }
    )
    counts_path = paths.REPORTS_TABLES_DIR / "phase1_join_counts.csv"
    counts_table.to_csv(counts_path, index=False)
    logger.info("Saved %s", counts_path)


if __name__ == "__main__":
    main()
