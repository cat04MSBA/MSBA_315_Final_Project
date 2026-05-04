"""Build the v2 corpus.

Combines:

* v1 (1,713 films) — re-used as-is from
  ``data/processed/films_joined.parquet`` and ``screenplays_parsed.pkl``.
* v2 survivors — 373 films enriched via Kaggle (230) / TMDB API (128) /
  Wikidata SPARQL (15). Their financial signals come from the
  enrichment cache; the rest of the metadata is pulled from the
  Kaggle CSV (which has full metadata for every survivor — verified).
  Screenplays are parsed two ways depending on the dedup-winning source:
  source='moviesum' → existing v1 XML parser on ``raw_script`` (canonical);
  source='sgogoi' → new adapter on ``scenes`` / ``elements``.

Outputs (under ``data/processed/v2/``):

* ``films_joined_v2.parquet`` — same schema as v1's films_joined.parquet
* ``screenplays_parsed_v2.pkl`` — dict[imdb_id, ParsedScreenplay]

All derived columns (effective_rating, log_budget, log_revenue,
primary_genre_bucketed, genres_bucketed, data_quality_flag) are
recomputed on the v2 corpus. Genre bucketing thresholds are
re-applied so the v2 dummies reflect v2's distribution.

Run from project root::

    python -m src.data.enrichment.build_corpus_v2
"""

from __future__ import annotations

# Allow running by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_corpus import (
    CorpusBuildConfig,
    add_data_quality_flag,
    add_derived_columns,
    attach_screenplay_metrics,
)
from src.data.enrichment.adapter import convert_to_parsed_screenplay
from src.data.enrichment.load_unified_scripts import load_unified_scripts
from src.data.join_corpus import dedupe_ratings
from src.data.load_ratings import load_ratings
from src.data.parse_screenplay import ParsedScreenplay, parse_screenplay
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Columns the v1 master parquet carries (excluding the ``script`` and
# ``summary`` source columns, which are dropped before saving).
_V1_PARQUET_COLS = [
    "imdb_id", "movie_name", "title_ms", "year_in_title", "origin_split",
    "script_char_len", "id", "title_rt", "original_title",
    "release_year_parsed", "release_date", "budget", "revenue", "runtime",
    "vote_average", "vote_count", "IMDB_Rating", "AverageRating",
    "Meta_score", "popularity", "status", "genres_parsed", "Director",
    "production_companies", "production_countries",
]


def _build_survivor_metadata(
    survivors_fin: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-survivor metadata row, mirroring v1's films_joined schema.

    Parameters
    ----------
    survivors_fin
        Output of the enrichment pipeline:
        ``data/processed/v2/candidate_financials.parquet`` with columns
        ``imdb_id, budget, revenue, vote_average, enrichment_source``.
    ratings
        Deduplicated Kaggle ratings DataFrame (one row per imdb_id).
    """
    # Pull the full Kaggle row for every survivor.
    rt_subset = ratings[ratings["imdb_id"].isin(survivors_fin["imdb_id"])].copy()
    logger.info(
        "Survivor metadata: %d/%d survivors found in Kaggle CSV",
        len(rt_subset), len(survivors_fin),
    )

    # Override budget/revenue/vote_average with the trusted enrichment values
    # for non-Kaggle sources (TMDB API, Wikidata).
    fin_idx = survivors_fin.set_index("imdb_id")
    rt_subset = rt_subset.set_index("imdb_id")
    for col in ("budget", "revenue", "vote_average"):
        # For non-Kaggle survivors, the API/Wikidata value supersedes
        # the Kaggle value (which was 0 / unknown).
        non_kaggle_mask = (
            fin_idx["enrichment_source"].reindex(rt_subset.index) != "kaggle"
        )
        rt_subset.loc[non_kaggle_mask, col] = fin_idx.loc[
            rt_subset.index[non_kaggle_mask], col
        ].astype(float)
    rt_subset = rt_subset.reset_index()

    # Synthesize the MovieSum-side fields v1 has (movie_name, title_ms,
    # script_char_len, origin_split). For survivors there's no MovieSum
    # XML — fill with sensible defaults so the parquet schema matches v1.
    rt_subset["movie_name"] = rt_subset["title"].fillna(rt_subset.get("original_title", ""))
    rt_subset["title_ms"] = rt_subset["title"].fillna(rt_subset.get("original_title", ""))
    rt_subset["title_rt"] = rt_subset["title"].fillna(rt_subset.get("original_title", ""))
    rt_subset["year_in_title"] = rt_subset.get("release_year", pd.NA)
    rt_subset["origin_split"] = "v2_survivor"  # marks v2 additions
    rt_subset["script_char_len"] = 0  # filled in below from the parsed screenplay

    # Keep only the v1 schema columns (those that exist; missing ones get NaN).
    out = pd.DataFrame({c: rt_subset[c] if c in rt_subset.columns else np.nan
                        for c in _V1_PARQUET_COLS})
    return out


def _parse_survivor_screenplays(
    survivor_ids: list[str],
    unified: dict[str, dict],
) -> dict[str, ParsedScreenplay]:
    """Parse each survivor's screenplay via the adapter.

    Originally routed source='moviesum' through v1's XML parser, but the
    user's notebook stored newer (2023+) screenplays as plaintext under
    source='moviesum' too — those don't parse as XML. Since the
    notebook has already broken every record into the canonical
    ``scenes`` / ``elements`` structure, the adapter handles all sources
    uniformly. The 1,713 v1 overlap is reused from v1's parses (not
    touched here), so the adapter only acts on net-new survivor films.
    """
    parsed: dict[str, ParsedScreenplay] = {}
    for imdb_id in survivor_ids:
        rec = unified.get(imdb_id)
        if rec is None:
            logger.warning("Survivor %s not in unified_scripts", imdb_id)
            continue
        parsed[imdb_id] = convert_to_parsed_screenplay(rec)
    logger.info("Parsed %d survivor screenplays via adapter", len(parsed))
    return parsed


def _normalize_genres_column(s: pd.Series) -> pd.Series:
    """Coerce mixed ndarray / list / NaN entries into pure Python lists.

    Round-tripping a list-typed column through parquet returns the
    elements as ``numpy.ndarray``, which trips ``compute_primary_genre``'s
    ``if not genres`` test. Convert defensively at the boundary.
    """
    def _coerce(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        # numpy array, pandas NA, etc.
        try:
            return list(x)
        except TypeError:
            return []
    return s.map(_coerce)


def build_corpus_v2(
    config: CorpusBuildConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, ParsedScreenplay]]:
    """Build films_joined_v2.parquet + screenplays_parsed_v2.pkl."""
    config = config or CorpusBuildConfig()
    out_dir = paths.DATA_PROCESSED_DIR / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. v1 corpus (reuse verbatim).
    v1_films = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    with (paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl").open("rb") as f:
        v1_parsed: dict[str, ParsedScreenplay] = pickle.load(f)
    logger.info("v1: %d films, %d parsed screenplays", len(v1_films), len(v1_parsed))

    # --- 2. Survivor metadata.
    survivors_fin = pd.read_parquet(out_dir / "candidate_financials.parquet")
    ratings = dedupe_ratings(load_ratings())
    survivors_meta = _build_survivor_metadata(survivors_fin, ratings)
    logger.info("Survivor metadata rows: %d", len(survivors_meta))

    # --- 3. Survivor screenplays.
    unified = load_unified_scripts()
    survivor_parsed = _parse_survivor_screenplays(
        survivors_meta["imdb_id"].tolist(), unified
    )

    # Drop survivors whose screenplay parsed to 0 scenes (degenerate).
    survivor_ids_ok = [
        i for i, p in survivor_parsed.items() if p.n_scenes > 0
    ]
    n_dropped_parse = len(survivor_parsed) - len(survivor_ids_ok)
    if n_dropped_parse:
        logger.warning(
            "Dropped %d survivors with 0-scene parses",
            n_dropped_parse,
        )
        survivors_meta = survivors_meta[
            survivors_meta["imdb_id"].isin(survivor_ids_ok)
        ].reset_index(drop=True)
        survivor_parsed = {i: survivor_parsed[i] for i in survivor_ids_ok}

    # Set script_char_len from the parsed screenplay (matches v1's convention).
    survivors_meta["script_char_len"] = survivors_meta["imdb_id"].map(
        lambda i: survivor_parsed[i].total_dialogue_chars
        + survivor_parsed[i].total_action_chars
    )

    # --- 4. Combine v1 + survivors at the films-table level.
    # Make sure both have the same column set in the same order.
    for col in _V1_PARQUET_COLS:
        if col not in v1_films.columns:
            logger.warning("v1 films_joined missing expected col %s", col)
    v1_subset = v1_films[[c for c in _V1_PARQUET_COLS if c in v1_films.columns]].copy()
    combined = pd.concat([v1_subset, survivors_meta], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset="imdb_id", keep="first").reset_index(drop=True)
    # Parquet round-trip turns list cells into numpy arrays; coerce.
    combined["genres_parsed"] = _normalize_genres_column(combined["genres_parsed"])
    logger.info("Combined v2 films: %d (v1=%d + new=%d)",
                len(combined), len(v1_films), len(combined) - len(v1_films))

    # --- 5. Re-derive columns on the v2 corpus.
    derived = add_derived_columns(combined, config)

    # --- 6. Combine parses.
    combined_parsed = {**v1_parsed, **survivor_parsed}
    # Filter to only IDs present in the combined films table.
    combined_parsed = {
        i: p for i, p in combined_parsed.items() if i in set(derived["imdb_id"])
    }
    logger.info("Combined v2 parses: %d", len(combined_parsed))

    # --- 7. Attach structural metrics + data-quality flag.
    with_metrics = attach_screenplay_metrics(derived, combined_parsed)
    final = add_data_quality_flag(with_metrics, config)

    # --- 8. Save.
    out_parquet = out_dir / "films_joined_v2.parquet"
    final.to_parquet(out_parquet, index=False)
    out_pkl = out_dir / "screenplays_parsed_v2.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(combined_parsed, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %s (%d rows) and %s (%d entries)",
                out_parquet.name, len(final), out_pkl.name, len(combined_parsed))

    return final, combined_parsed


def main() -> None:
    df, parsed = build_corpus_v2()
    print("\n=== build_corpus_v2 — v2 master corpus ===")
    print(f"  Films:                {len(df):,}")
    print(f"  Year range:           {int(df['release_year_parsed'].min())} – "
          f"{int(df['release_year_parsed'].max())}")
    print(f"  Median budget:        ${int(df['budget'].median()):,}")
    print(f"  Median revenue:       ${int(df['revenue'].median()):,}")
    print(f"  Mean rating:          {df['effective_rating'].mean():.2f}")
    print(f"  Median scenes:        {int(df['n_scenes'].median())}")
    print(f"  Data-quality flagged: {int(df['data_quality_flag'].sum())}")
    print(f"  v2 additions:         {(df['origin_split'] == 'v2_survivor').sum()}")
    print()
    print("Genre distribution (top 12):")
    print(df["primary_genre_bucketed"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
