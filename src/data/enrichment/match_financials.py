"""TMDB API enrichment for v2 candidate films.

Two passes:

* **Kaggle pass** is consumed from ``corpus_final.jsonl`` (the user's
  ``MovieScripts.ipynb`` already ran this for the union of MovieSum
  and sgogoi screenplays — 1,943 four-signal films total, 230 of which
  are sgogoi additions beyond v1).
* **TMDB API pass** fills the gap. For each candidate IMDb ID we hit
  ``/3/find/{imdb_id}?external_source=imdb_id`` to resolve the TMDB id,
  then ``/3/movie/{tmdb_id}`` to fetch ``budget``, ``revenue``, and
  ``vote_average``. Threaded with ``ThreadPoolExecutor`` and cached to
  ``data/processed/v2/imdb_api_cache.parquet`` so re-runs are free.

Per the planning conversation: only those three fields are captured
(speed). The cache schema is ``(imdb_id, budget, revenue,
vote_average, fetched_at, status)``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Hardcoded fallback key from data/data_enrichment/ROI_Enrichment.ipynb;
# user-approved for v2 enrichment. Override via the TMDB_API_KEY env var.
_DEFAULT_API_KEY = "94b34b988e4d8293a6dac442dfdb4522"


@dataclass(frozen=True)
class TMDBConfig:
    """Knobs for the TMDB enrichment pass."""
    api_key: str = ""  # resolved via env var or fallback
    max_workers: int = 15
    timeout_s: int = 10
    cache_path: Path = Path()  # set in __post_init__-style helper

    @staticmethod
    def resolve() -> "TMDBConfig":
        """Build a TMDBConfig with env-aware defaults."""
        key = os.environ.get("TMDB_API_KEY") or _DEFAULT_API_KEY
        return TMDBConfig(
            api_key=key,
            max_workers=15,
            timeout_s=10,
            cache_path=paths.DATA_PROCESSED_DIR / "v2" / "imdb_api_cache.parquet",
        )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> pd.DataFrame:
    """Load the API cache, or return an empty frame with the right schema."""
    if cache_path.is_file():
        df = pd.read_parquet(cache_path)
        logger.info("API cache: loaded %d rows", len(df))
        return df
    logger.info("API cache: none yet at %s", cache_path)
    return pd.DataFrame(
        columns=["imdb_id", "budget", "revenue", "vote_average", "fetched_at", "status"]
    )


def save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Atomically write the cache parquet (overwrites)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    logger.info("API cache: saved %d rows", len(df))


# ---------------------------------------------------------------------------
# TMDB calls
# ---------------------------------------------------------------------------

def fetch_one(imdb_id: str, api_key: str, timeout: int) -> dict:
    """Resolve one IMDb ID to (budget, revenue, vote_average) via TMDB.

    Returns a dict with keys ``imdb_id``, ``budget``, ``revenue``,
    ``vote_average``, ``status``. ``budget`` and ``revenue`` are None
    when missing or the source returns 0 (TMDB's missing-value sentinel).
    Never raises; failures are recorded in ``status``.
    """
    base = "https://api.themoviedb.org/3"
    out = {"imdb_id": imdb_id, "budget": None, "revenue": None, "vote_average": None, "status": "ok"}
    try:
        r1 = requests.get(
            f"{base}/find/{imdb_id}",
            params={"api_key": api_key, "external_source": "imdb_id"},
            timeout=timeout,
        )
        if r1.status_code != 200:
            out["status"] = f"find_http_{r1.status_code}"
            return out
        results = r1.json().get("movie_results", [])
        if not results:
            out["status"] = "no_movie_results"
            return out
        tmdb_id = results[0]["id"]
        r2 = requests.get(
            f"{base}/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=timeout,
        )
        if r2.status_code != 200:
            out["status"] = f"movie_http_{r2.status_code}"
            return out
        d = r2.json()
        budget = d.get("budget") or None
        revenue = d.get("revenue") or None
        vote_average = d.get("vote_average")
        if budget == 0:
            budget = None
        if revenue == 0:
            revenue = None
        if vote_average in (0, 0.0):
            vote_average = None
        out["budget"] = budget
        out["revenue"] = revenue
        out["vote_average"] = vote_average
    except requests.RequestException as exc:
        out["status"] = f"request_error:{type(exc).__name__}"
    except (ValueError, KeyError) as exc:
        out["status"] = f"parse_error:{type(exc).__name__}"
    return out


def fetch_many(
    imdb_ids: list[str],
    config: TMDBConfig,
    log_every: int = 100,
) -> pd.DataFrame:
    """Threaded TMDB fetch over a list of IDs. Returns a DataFrame."""
    if not imdb_ids:
        return pd.DataFrame(columns=["imdb_id", "budget", "revenue", "vote_average", "status", "fetched_at"])
    logger.info("TMDB fetch: %d ids, %d workers", len(imdb_ids), config.max_workers)
    rows: list[dict] = []
    fetched_at = pd.Timestamp.utcnow().isoformat()
    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        futures = [ex.submit(fetch_one, i, config.api_key, config.timeout_s) for i in imdb_ids]
        for n, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            res["fetched_at"] = fetched_at
            rows.append(res)
            if n % log_every == 0:
                logger.debug("TMDB fetch: %d / %d done", n, len(imdb_ids))
    out = pd.DataFrame(rows)
    n_ok = (out["status"] == "ok").sum()
    n_b = out["budget"].notna().sum()
    n_r = out["revenue"].notna().sum()
    n_v = out["vote_average"].notna().sum()
    n_all3 = (out["budget"].notna() & out["revenue"].notna() & out["vote_average"].notna()).sum()
    logger.info(
        "TMDB fetch done: %d/%d ok | budget %d | revenue %d | rating %d | all3 %d",
        n_ok, len(out), n_b, n_r, n_v, n_all3,
    )
    return out


def enrich_via_api(
    imdb_ids: list[str],
    config: TMDBConfig | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Run the API for the given IDs, with cache.

    Parameters
    ----------
    imdb_ids
        IMDb IDs to enrich. Duplicates are deduped.
    config
        Optional TMDBConfig. Defaults to ``TMDBConfig.resolve()``.
    refresh
        If True, ignore cache and re-fetch every ID. Default False.

    Returns
    -------
    DataFrame with one row per requested ID and columns
    ``imdb_id, budget, revenue, vote_average, status, fetched_at``.
    The cache on disk is updated in-place.
    """
    config = config or TMDBConfig.resolve()
    requested = sorted(set(imdb_ids))
    cache = load_cache(config.cache_path)

    if refresh:
        to_fetch = requested
    else:
        cached_ids = set(cache["imdb_id"])
        to_fetch = [i for i in requested if i not in cached_ids]
    logger.info(
        "enrich_via_api: %d requested, %d cached, %d to fetch",
        len(requested), len(requested) - len(to_fetch), len(to_fetch),
    )

    if to_fetch:
        new_rows = fetch_many(to_fetch, config)
        cache = pd.concat([cache, new_rows], ignore_index=True)
        cache = cache.drop_duplicates(subset="imdb_id", keep="last").reset_index(drop=True)
        save_cache(cache, config.cache_path)

    return cache[cache["imdb_id"].isin(requested)].reset_index(drop=True)
