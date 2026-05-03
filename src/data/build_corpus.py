"""Phase 2 production pipeline: build the canonical processed corpus.

This module is the single entry point that produces the master working
dataset every downstream phase consumes:

* ``data/processed/films_joined.parquet`` — one row per film, with all
  source columns plus the derived columns (``effective_rating``,
  ``log_budget``, ``log_revenue``, ``primary_genre``, ``genres_bucketed``,
  ``primary_genre_bucketed``) and the screenplay-structural metrics
  (``n_scenes``, ``n_unique_characters``, ``n_dialogue_lines``,
  ``total_dialogue_chars``, ``total_stage_direction_chars``,
  ``total_scene_description_chars``, ``total_action_chars``,
  ``dialogue_to_action_ratio``, ``dialogue_to_total_text_ratio``,
  ``parse_warning_count``).
* ``data/processed/screenplays_parsed.pkl`` — pickle of
  ``dict[imdb_id, ParsedScreenplay]``. The full structured screenplay
  is too rich to denormalize into the master table; Phase 3 reads
  both files (master table for metadata and outcomes, structured
  screenplays for feature extraction).

All pipeline knobs are exposed via :class:`CorpusBuildConfig` so testing
alternative preprocessing choices is a one-line config override.

Run from the project root:

    python -m src.data.build_corpus

Idempotent: running twice produces the same outputs.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python3 -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
from collections import Counter
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.join_corpus import dedupe_ratings, join_corpora
from src.data.load_moviesum import load_moviesum
from src.data.load_ratings import load_ratings
from src.data.parse_screenplay import ParsedScreenplay, parse_screenplay
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusBuildConfig:
    """Pipeline knobs. Override any field to test alternatives.

    Examples
    --------
    >>> # Default build:
    >>> df, parsed = build_corpus()
    >>>
    >>> # Test a different year cutoff and genre threshold:
    >>> from dataclasses import replace
    >>> cfg = replace(CorpusBuildConfig(), min_year=1990, genre_min_count=50)
    >>> df, parsed = build_corpus(cfg)
    """
    # --- Filters ---
    # Year cutoff. Drops films with ``release_year_parsed < min_year``.
    # Default is 1900 (no effective cutoff beyond ``year_clip_min``):
    # the planning conversation's original 1995 cutoff was reversed
    # mid-Phase-2 (decision 2026-05-02) after the Phase 1 EDA's
    # pre-1995 count was found to undercount by an order of magnitude
    # (claimed ~50, actually 398). Override this knob to test
    # alternative cutoffs (e.g., 1980 → 1,609-film corpus).
    min_year: int = 1900
    # Clip release_year_parsed to a sane range so the future-year noise
    # in the ratings dataset (e.g., 2055, 2099) doesn't propagate.
    year_clip_min: int = 1900
    year_clip_max: int = 2025

    # --- Derived-column choices ---
    # Monetary transform: how to compute ``log_budget`` and ``log_revenue``.
    # ``log1p`` is the default (handles 0 gracefully, but our filter
    # already excludes zeros — defensive). ``log10`` is the alternative.
    monetary_transform: str = "log1p"
    # Rating priority: column to use first for ``effective_rating``,
    # falling back to the next when the first is missing or 0.
    rating_priority: tuple[str, ...] = ("IMDB_Rating", "vote_average")

    # --- Genre bucketing ---
    # Genres with fewer than this many films in the working corpus
    # collapse into "Other".
    genre_min_count: int = 30

    # --- Data-quality flag (degenerate scene structure) ---
    # Some MovieSum source XMLs have missing or collapsed scene
    # boundaries: the entire screenplay is encoded as one or a few
    # `<scene>` elements containing all the dialogue. Films matching
    # `n_scenes < data_quality_min_scenes AND total_dialogue_chars >
    # data_quality_min_dialogue_chars` are flagged via the
    # `data_quality_flag` boolean column on the master DataFrame.
    # Phase 3 decides whether to filter, downweight, or include them
    # as-is.
    data_quality_min_scenes: int = 10
    data_quality_min_dialogue_chars: int = 50_000

    # --- I/O ---
    # MovieSum dedup CSV (Phase 1 user-filled file). The pipeline reads
    # the ``decision`` column; ``keep`` keeps the row, ``drop`` /
    # ``remove`` (synonyms) drop it.
    moviesum_dedup_csv: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR
        / "phase1_moviesum_duplicates_review.csv"
    )
    out_dir: Path = field(default_factory=lambda: paths.DATA_PROCESSED_DIR)

    # --- Reproducibility ---
    seed: int = 42


# Tokens (case-insensitive, after `.strip()`) that mean "drop this row".
_DROP_TOKENS = frozenset({"drop", "remove", "delete", "discard"})
_KEEP_TOKENS = frozenset({"keep"})


# ---------------------------------------------------------------------------
# Step 1: MovieSum dedup from the user-filled review CSV
# ---------------------------------------------------------------------------

def dedupe_moviesum_from_csv(
    moviesum: pd.DataFrame, dedup_csv: Path
) -> pd.DataFrame:
    """Apply the user's per-pair `keep`/`drop` decisions from the review CSV.

    Parameters
    ----------
    moviesum
        The full MovieSum DataFrame (potentially with duplicate IMDb IDs).
    dedup_csv
        Path to ``reports/tables/phase1_moviesum_duplicates_review.csv``,
        containing one row per duplicate variant with a filled
        ``decision`` column.

    Returns
    -------
    pandas.DataFrame
        MovieSum with the user-flagged drops removed. Rows whose IMDb ID
        is not in the dup-review file are kept untouched (they had no
        duplicates to begin with).

    Raises
    ------
    FileNotFoundError
        If ``dedup_csv`` does not exist.
    ValueError
        If any row has a missing or unrecognized ``decision``, or if a
        duplicate-IMDb-ID pair does not resolve to exactly one ``keep``
        and one ``drop``.
    """
    dedup_csv = Path(dedup_csv)
    if not dedup_csv.is_file():
        raise FileNotFoundError(f"MovieSum dedup CSV not found at {dedup_csv}")

    review = pd.read_csv(dedup_csv)
    review["_decision_norm"] = (
        review["decision"].fillna("").astype(str).str.lower().str.strip()
    )

    # Validate every row has a recognizable decision.
    bad = review[~review["_decision_norm"].isin(_DROP_TOKENS | _KEEP_TOKENS)]
    if not bad.empty:
        msg = (
            f"{len(bad)} row(s) in {dedup_csv.name} have missing or "
            "unrecognized decision values. Edit the CSV to use 'keep' "
            "or 'drop' (or 'remove') and rerun."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Validate every IMDb ID has exactly one keep and one drop.
    bad_pairs = []
    for imdb_id, grp in review.groupby("imdb_id"):
        decisions = sorted(grp["_decision_norm"].tolist())
        if len(decisions) != 2 or not (
            any(d in _KEEP_TOKENS for d in decisions)
            and any(d in _DROP_TOKENS for d in decisions)
        ):
            bad_pairs.append((imdb_id, decisions))
    if bad_pairs:
        msg = (
            f"{len(bad_pairs)} duplicate IMDb ID pair(s) do not have "
            f"exactly one keep + one drop: {bad_pairs[:3]}..."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Identify the (imdb_id, movie_name) tuples to drop. Use both keys
    # because the same imdb_id has multiple rows in MovieSum.
    drops = set(
        (row["imdb_id"], row["movie_name"])
        for _, row in review[review["_decision_norm"].isin(_DROP_TOKENS)].iterrows()
    )

    before = len(moviesum)
    keep_mask = ~moviesum.apply(
        lambda r: (r["imdb_id"], r["movie_name"]) in drops, axis=1
    )
    out = moviesum.loc[keep_mask].reset_index(drop=True)
    logger.info(
        "Dedup MovieSum from review CSV: %d → %d (%d dropped per user decisions)",
        before, len(out), before - len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Step 2: Filters
# ---------------------------------------------------------------------------

def apply_corpus_filters(
    joined: pd.DataFrame, config: CorpusBuildConfig
) -> pd.DataFrame:
    """Apply pre-1995 cutoff, drop unmatched films, clip year range.

    Filters in order:
    1. Drop the 2 (or however many) MovieSum films absent from the
       ratings dataset (no ``id`` after the merge).
    2. Drop films with ``budget == 0`` or ``revenue == 0`` (the
       0-as-missing convention).
    3. Drop films with no rating signal (both ``IMDB_Rating`` and
       ``vote_average`` are 0 or null).
    4. Clip ``release_year_parsed`` to ``[year_clip_min, year_clip_max]``
       (drop rows whose year is outside that range — these are
       scheduled-release noise like 2055, 2099).
    5. Drop films with ``release_year_parsed < min_year`` (pre-1995
       cutoff).
    """
    n0 = len(joined)
    df = joined.copy()

    # 1. Unmatched.
    df = df[df["id"].notna()]
    n1 = len(df)

    # 2. Zero budget or revenue (0-as-missing).
    df = df[(df["budget"].fillna(0) > 0) & (df["revenue"].fillna(0) > 0)]
    n2 = len(df)

    # 3. No rating signal.
    has_imdb_rating = df["IMDB_Rating"].fillna(0) > 0
    has_tmdb_rating = df["vote_average"].fillna(0) > 0
    df = df[has_imdb_rating | has_tmdb_rating]
    n3 = len(df)

    # 4. Year clip.
    year_in_range = (
        df["release_year_parsed"].between(
            config.year_clip_min, config.year_clip_max, inclusive="both"
        )
    )
    df = df[year_in_range]
    n4 = len(df)

    # 5. Pre-min-year cutoff.
    df = df[df["release_year_parsed"] >= config.min_year]
    n5 = len(df)

    logger.info(
        "Corpus filters: %d → unmatched %d → no $ %d → no rating %d → "
        "year_clip %d → min_year(%d) %d",
        n0, n1, n2, n3, n4, config.min_year, n5,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: Derived columns
# ---------------------------------------------------------------------------

def compute_effective_rating(
    df: pd.DataFrame, priority: tuple[str, ...]
) -> pd.Series:
    """Coalesce the rating columns in priority order, treating 0 as missing."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in priority:
        if col not in df.columns:
            continue
        candidate = df[col]
        # Treat 0 (and NaN) as missing so we can fall back to the next column.
        usable = candidate.where(candidate.fillna(0) > 0)
        out = out.fillna(usable)
    return out


def compute_monetary_log(
    series: pd.Series, transform: str
) -> pd.Series:
    """Apply the configured monetary transform to a budget/revenue column."""
    if transform == "log1p":
        return np.log1p(series.astype(float))
    if transform == "log10":
        # +1 to handle the (already-filtered-out) zero case defensively.
        return np.log10(series.astype(float).clip(lower=1.0))
    raise ValueError(
        f"Unknown monetary_transform={transform!r}. Use 'log1p' or 'log10'."
    )


def compute_primary_genre(genres: list[str] | None) -> str:
    """Return the first genre in the list, or ``'Unknown'`` if empty/null."""
    if not genres or not isinstance(genres, (list, tuple)):
        return "Unknown"
    return genres[0] if genres else "Unknown"


def bucket_genres(
    df: pd.DataFrame, threshold: int
) -> tuple[pd.Series, pd.Series, set[str]]:
    """Compute ``genres_bucketed`` and ``primary_genre_bucketed`` columns.

    Genres with fewer than ``threshold`` appearances across the corpus
    collapse into the string ``"Other"``. Returns ``(genres_bucketed,
    primary_genre_bucketed, kept_genres_set)``.
    """
    # Count genre appearances across the corpus.
    counter: Counter[str] = Counter()
    for genre_list in df["genres_parsed"]:
        if isinstance(genre_list, (list, tuple)):
            for g in genre_list:
                counter[g] += 1
    kept = {g for g, n in counter.items() if n >= threshold}

    def _bucket_list(genres):
        if not isinstance(genres, (list, tuple)):
            return ["Unknown"]
        return [g if g in kept else "Other" for g in genres]

    genres_bucketed = df["genres_parsed"].map(_bucket_list)
    primary_bucketed = df["primary_genre"].map(
        lambda g: g if g in kept else ("Unknown" if g == "Unknown" else "Other")
    )
    return genres_bucketed, primary_bucketed, kept


def add_derived_columns(
    df: pd.DataFrame, config: CorpusBuildConfig
) -> pd.DataFrame:
    """Compute every derived column on the corpus DataFrame.

    Adds: ``effective_rating``, ``log_budget``, ``log_revenue``,
    ``primary_genre``, ``genres_bucketed``, ``primary_genre_bucketed``.
    """
    df = df.copy()
    df["effective_rating"] = compute_effective_rating(df, config.rating_priority)
    df["log_budget"] = compute_monetary_log(df["budget"], config.monetary_transform)
    df["log_revenue"] = compute_monetary_log(df["revenue"], config.monetary_transform)
    df["primary_genre"] = df["genres_parsed"].map(compute_primary_genre)

    genres_bucketed, primary_bucketed, kept = bucket_genres(df, config.genre_min_count)
    df["genres_bucketed"] = genres_bucketed
    df["primary_genre_bucketed"] = primary_bucketed

    logger.info(
        "Derived columns added. Kept genres (≥%d films): %s",
        config.genre_min_count, sorted(kept),
    )
    return df


# ---------------------------------------------------------------------------
# Step 4: Screenplay parsing + structural metrics
# ---------------------------------------------------------------------------

def parse_all_screenplays(
    df: pd.DataFrame,
) -> dict[str, ParsedScreenplay]:
    """Parse every screenplay in the corpus DataFrame.

    Iterates ``df`` in row order; expects ``imdb_id`` and ``script``
    columns to be present. Logs progress at INFO every 500 films.
    """
    parsed: dict[str, ParsedScreenplay] = {}
    n_warnings_total = 0
    for idx, row in df.iterrows():
        p = parse_screenplay(row["script"], row["imdb_id"])
        parsed[row["imdb_id"]] = p
        n_warnings_total += len(p.parse_warnings)
        if (idx + 1) % 500 == 0:
            logger.debug("Parsed %d / %d screenplays", idx + 1, len(df))

    n_with_warnings = sum(1 for p in parsed.values() if p.parse_warnings)
    logger.info(
        "Parsed %s screenplays (%s with warnings, %s warnings total)",
        f"{len(parsed):,}", f"{n_with_warnings:,}", f"{n_warnings_total:,}",
    )
    return parsed


def attach_screenplay_metrics(
    df: pd.DataFrame, parsed: dict[str, ParsedScreenplay]
) -> pd.DataFrame:
    """Denormalize the structural metrics from each ParsedScreenplay onto df.

    Adds columns: ``n_scenes``, ``n_unique_characters``, ``n_dialogue_lines``,
    ``total_dialogue_chars``, ``total_stage_direction_chars``,
    ``total_scene_description_chars``, ``total_action_chars``,
    ``dialogue_to_action_ratio``, ``dialogue_to_total_text_ratio``,
    ``parse_warning_count``.
    """
    df = df.copy()
    metric_cols = [
        "n_scenes", "n_unique_characters", "n_dialogue_lines",
        "total_dialogue_chars", "total_stage_direction_chars",
        "total_scene_description_chars", "total_action_chars",
        "dialogue_to_action_ratio", "dialogue_to_total_text_ratio",
    ]

    rows: list[dict] = []
    for imdb_id in df["imdb_id"]:
        p = parsed.get(imdb_id)
        if p is None:
            # Shouldn't happen — parse_all_screenplays runs over df.
            rows.append({c: None for c in metric_cols} | {"parse_warning_count": None})
            continue
        rows.append({
            "n_scenes": p.n_scenes,
            "n_unique_characters": p.n_unique_characters,
            "n_dialogue_lines": p.n_dialogue_lines,
            "total_dialogue_chars": p.total_dialogue_chars,
            "total_stage_direction_chars": p.total_stage_direction_chars,
            "total_scene_description_chars": p.total_scene_description_chars,
            "total_action_chars": p.total_action_chars,
            "dialogue_to_action_ratio": p.dialogue_to_action_ratio,
            "dialogue_to_total_text_ratio": p.dialogue_to_total_text_ratio,
            "parse_warning_count": len(p.parse_warnings),
        })
    metrics_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, metrics_df], axis=1)


# ---------------------------------------------------------------------------
# Step 5: Data-quality flag
# ---------------------------------------------------------------------------

def add_data_quality_flag(
    df: pd.DataFrame, config: CorpusBuildConfig
) -> pd.DataFrame:
    """Mark films whose source XML has degenerate scene structure.

    Some MovieSum source files encode the whole screenplay as one or
    a small number of ``<scene>`` elements (scene boundaries missing
    in the source), even though the dialogue volume is consistent
    with a normal feature-length screenplay. Such films pollute any
    per-scene analysis and any feature derived from
    ``n_scenes``. Adds the boolean column ``data_quality_flag``;
    True means "structurally degenerate, handle with care."

    Phase 3 decides whether to filter, downweight, or include them
    as-is. The flag is informational; it does not exclude any films
    from the master corpus.
    """
    n_scenes = df["n_scenes"]
    total_dialogue_chars = df["total_dialogue_chars"]
    flag = (
        (n_scenes < config.data_quality_min_scenes)
        & (total_dialogue_chars > config.data_quality_min_dialogue_chars)
    )
    df = df.copy()
    df["data_quality_flag"] = flag
    n_flagged = int(flag.sum())
    logger.info(
        "Data-quality flag: %d films flagged (n_scenes < %d AND "
        "total_dialogue_chars > %d)",
        n_flagged, config.data_quality_min_scenes,
        config.data_quality_min_dialogue_chars,
    )
    return df


# ---------------------------------------------------------------------------
# Step 6: Validation + save
# ---------------------------------------------------------------------------

def validate_processed(df: pd.DataFrame, config: CorpusBuildConfig) -> None:
    """Hard assertions on the final DataFrame; raise on anything bad."""
    assert len(df) > 0, "Processed corpus is empty"
    assert df["imdb_id"].is_unique, "Duplicate imdb_id rows in processed corpus"
    assert df["effective_rating"].notna().all(), (
        "effective_rating has nulls"
    )
    assert (df["budget"] > 0).all(), "budget has zeros (0-as-missing convention)"
    assert (df["revenue"] > 0).all(), "revenue has zeros (0-as-missing convention)"
    assert (df["release_year_parsed"] >= config.min_year).all(), (
        f"Films older than min_year={config.min_year} in processed corpus"
    )
    assert (df["release_year_parsed"] <= config.year_clip_max).all(), (
        f"Films newer than year_clip_max={config.year_clip_max}"
    )
    assert (df["n_scenes"] > 0).all(), "Films with 0 parsed scenes in corpus"
    # ratios in [0,1]
    for col in ("dialogue_to_action_ratio", "dialogue_to_total_text_ratio"):
        assert df[col].between(0, 1).all(), f"{col} outside [0,1]"


def save_artifacts(
    df: pd.DataFrame,
    parsed: dict[str, ParsedScreenplay],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Save the master parquet and the parsed-screenplays pickle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop the ``script`` column from the parquet — it's a duplicate of
    # what's already in ``screenplays_parsed.pkl`` and would bloat the
    # file by hundreds of MB. Same for ``summary``.
    parquet_df = df.drop(columns=["script", "summary"], errors="ignore")
    parquet_path = out_dir / "films_joined.parquet"
    parquet_df.to_parquet(parquet_path, index=False)
    logger.info("Saved master parquet: %s rows", f"{len(parquet_df):,}")
    logger.debug("Wrote %s", parquet_path)

    pkl_path = out_dir / "screenplays_parsed.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(parsed, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved parsed-screenplays pickle: %s entries", f"{len(parsed):,}")
    logger.debug("Wrote %s", pkl_path)

    return parquet_path, pkl_path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_corpus(
    config: CorpusBuildConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, ParsedScreenplay]]:
    """Build the processed corpus end-to-end.

    Steps:
    1. Load ratings dataset and dedupe on ``imdb_id``.
    2. Load MovieSum and apply the user's CSV-based dedup policy.
    3. Join on ``imdb_id``.
    4. Apply corpus filters (unmatched, 0-budget/revenue, no rating,
       year-clip, pre-min_year cutoff).
    5. Compute derived columns (effective_rating, log transforms,
       primary/bucketed genres).
    6. Parse every screenplay; attach structural metrics.
    7. Validate; save artifacts.

    Returns the final master DataFrame and the parsed-screenplays dict.
    """
    config = config or CorpusBuildConfig()
    paths.ensure_dirs()

    # --- 1. Ratings.
    ratings = load_ratings()
    ratings = dedupe_ratings(ratings)

    # --- 2. MovieSum.
    moviesum = load_moviesum(include_script=True)
    moviesum = dedupe_moviesum_from_csv(moviesum, config.moviesum_dedup_csv)

    # --- 3. Join.
    joined = join_corpora(moviesum, ratings)

    # --- 4. Filter.
    filtered = apply_corpus_filters(joined, config)

    # --- 5. Derive.
    derived = add_derived_columns(filtered, config)

    # --- 6. Parse screenplays + denormalize metrics.
    parsed = parse_all_screenplays(derived)
    with_metrics = attach_screenplay_metrics(derived, parsed)

    # --- 7. Add data-quality flag for degenerate scene structures.
    final_df = add_data_quality_flag(with_metrics, config)

    # --- 8. Validate + save.
    validate_processed(final_df, config)
    save_artifacts(final_df, parsed, config.out_dir)

    logger.info("build_corpus complete: %s films in master table", f"{len(final_df):,}")
    return final_df, parsed


def main() -> tuple[pd.DataFrame, dict[str, ParsedScreenplay]]:
    """CLI entry point. Default config; prints headline counts."""
    df, parsed = build_corpus()
    print("\n=== build_corpus — Phase 2 master corpus ===")
    print(f"  Films:              {len(df):,}")
    print(f"  Year range:         {int(df['release_year_parsed'].min())} – "
          f"{int(df['release_year_parsed'].max())}")
    print(f"  Median budget:      ${int(df['budget'].median()):,}")
    print(f"  Median revenue:     ${int(df['revenue'].median()):,}")
    print(f"  Mean rating:        {df['effective_rating'].mean():.2f}")
    print(f"  Median scenes:      {int(df['n_scenes'].median())}")
    print(f"  Median dialogue lines: {int(df['n_dialogue_lines'].median()):,}")
    print(f"  Data-quality flagged: {int(df['data_quality_flag'].sum())}  "
          f"(degenerate scene structure)")
    print(f"  Saved parquet to:   {Path(config_default_out())} / films_joined.parquet")
    return df, parsed


def config_default_out() -> Path:
    """Default ``out_dir`` from a fresh CorpusBuildConfig (for the print message)."""
    return CorpusBuildConfig().out_dir


if __name__ == "__main__":
    main()
