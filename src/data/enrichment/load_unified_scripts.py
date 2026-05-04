"""Load and dedupe the user's unified screenplay corpus.

The user's ``MovieScripts.ipynb`` produced two artifacts in
``data/data_enrichment/data/processed/``:

* ``unified_scripts.jsonl`` — 3,243 records / 2,702 unique IMDb IDs
  with screenplays from MovieSum (``source='moviesum'``) and sgogoi
  (``source='sgogoi'``). Some IDs appear under both sources (529 of
  them), the dedup logic here picks the better-parsed copy.
* ``corpus_final.jsonl`` — 1,943 records (1,713 MovieSum from v1 +
  230 sgogoi additions) with budget / revenue / vote_average already
  joined from the Kaggle IMDB-TMDB Big Dataset.

The schema-adapter (``ParsedScreenplay`` builder) lives in this module
too, used downstream by the v2 pipeline once the financial-enrichment
step has identified which films will survive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default paths under data/data_enrichment/data/processed/
_ENRICHMENT_DIR = Path(__file__).resolve().parents[3] / "data" / "data_enrichment" / "data" / "processed"
UNIFIED_SCRIPTS_PATH = _ENRICHMENT_DIR / "unified_scripts.jsonl"
CORPUS_FINAL_PATH = _ENRICHMENT_DIR / "corpus_final.jsonl"


# ---------------------------------------------------------------------------
# Dedupe records by imdb_id
# ---------------------------------------------------------------------------

def _dialogue_count(rec: dict) -> int:
    """Pull the per-record dialogue-element count for the dedup tiebreak."""
    pq = rec.get("parse_quality") or {}
    ebt = pq.get("elements_by_type") or {}
    return int(ebt.get("dialogue", 0))


def dedupe_unified_records(records: list[dict]) -> dict[str, dict]:
    """Pick one record per imdb_id.

    Sort key: longest ``raw_script`` wins; on tie, higher
    ``parse_quality.elements_by_type.dialogue`` wins. This matches both
    the v1 MovieSum dedup convention (longest script) and the prompt's
    spec for the sgogoi/MovieSum cross-source overlap.
    """
    by_id: dict[str, dict] = {}
    for rec in records:
        imdb_id = rec["imdb_id"]
        prev = by_id.get(imdb_id)
        if prev is None:
            by_id[imdb_id] = rec
            continue
        prev_len = len(prev.get("raw_script") or "")
        cur_len = len(rec.get("raw_script") or "")
        if cur_len > prev_len:
            by_id[imdb_id] = rec
        elif cur_len == prev_len and _dialogue_count(rec) > _dialogue_count(prev):
            by_id[imdb_id] = rec
    logger.info("Deduped unified records: %d → %d", len(records), len(by_id))
    return by_id


# ---------------------------------------------------------------------------
# Top-level loaders
# ---------------------------------------------------------------------------

def load_unified_scripts(path: Path = UNIFIED_SCRIPTS_PATH) -> dict[str, dict]:
    """Load and dedupe ``unified_scripts.jsonl``.

    Returns a dict keyed by imdb_id. Values are the raw record dicts
    with at minimum ``imdb_id, title, source, raw_script, scenes``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"unified_scripts.jsonl not found at {path}")
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d raw records from %s", len(records), path.name)
    return dedupe_unified_records(records)


def load_corpus_final(path: Path = CORPUS_FINAL_PATH) -> pd.DataFrame:
    """Load ``corpus_final.jsonl`` as a flat DataFrame.

    Columns kept: ``imdb_id, source, title, budget, revenue,
    vote_average, vote_count, release_year, runtime``. Drops the
    ``raw_script`` / ``scenes`` payload (not needed at this stage —
    we re-derive from unified_scripts on the survivor set).
    """
    if not path.is_file():
        raise FileNotFoundError(f"corpus_final.jsonl not found at {path}")
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "imdb_id": rec["imdb_id"],
                "source": rec.get("source"),
                "title": rec.get("title"),
                "budget": rec.get("budget"),
                "revenue": rec.get("revenue"),
                "vote_average": rec.get("vote_average"),
                "vote_count": rec.get("vote_count"),
                "release_year": rec.get("release_year"),
                "runtime": rec.get("runtime"),
            })
    df = pd.DataFrame(rows)
    logger.info("Loaded %d corpus_final rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Candidate-pool computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidatePools:
    """Sets of IMDb IDs computed for the v2 build."""
    v1_ids: frozenset[str]
    unified_ids: frozenset[str]
    moviesum_unified_ids: frozenset[str]
    sgogoi_unified_ids: frozenset[str]

    @property
    def candidates(self) -> frozenset[str]:
        """All candidates needing enrichment: unified IDs not already in v1."""
        return self.unified_ids - self.v1_ids

    @property
    def new_sgogoi_only(self) -> frozenset[str]:
        """Sgogoi-only adds (never seen in v1 or MovieSum raw)."""
        return self.sgogoi_unified_ids - self.v1_ids - self.moviesum_unified_ids

    @property
    def v1_moviesum_drops(self) -> frozenset[str]:
        """MovieSum films present in v1 raw but excluded from v1 corpus."""
        return self.moviesum_unified_ids - self.v1_ids


def compute_candidate_pools(
    v1_films_joined: pd.DataFrame,
    unified: dict[str, dict],
) -> CandidatePools:
    """Build the candidate-pool sets from v1 corpus and unified scripts."""
    v1_ids = frozenset(v1_films_joined["imdb_id"].tolist())
    unified_ids = frozenset(unified.keys())
    moviesum_ids: set[str] = set()
    sgogoi_ids: set[str] = set()
    for imdb_id, rec in unified.items():
        # A record is one source after dedup; for cross-source presence
        # we'd need the pre-dedup data. The dedup already kept the
        # better-parsed copy; for the moviesum/sgogoi-presence question
        # we treat a record as "available from its source". Original
        # cross-source presence (pre-dedup) is captured separately in
        # the exploration script.
        src = rec.get("source")
        if src == "moviesum":
            moviesum_ids.add(imdb_id)
        elif src == "sgogoi":
            sgogoi_ids.add(imdb_id)
    return CandidatePools(
        v1_ids=v1_ids,
        unified_ids=unified_ids,
        moviesum_unified_ids=frozenset(moviesum_ids),
        sgogoi_unified_ids=frozenset(sgogoi_ids),
    )
