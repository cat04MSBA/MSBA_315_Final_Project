"""Wikidata SPARQL fallback for the residual films TMDB couldn't fully fill.

Wikidata's structured property graph carries:

* P345  → IMDb ID (the join key).
* P2130 → cost (production budget, in USD or local currency w/ unit).
* P2142 → box office.
* (Wikidata has no IMDb-rating equivalent; rating must come from TMDB
  or be dropped.)

The endpoint is ``https://query.wikidata.org/sparql``, free, no key,
50 IDs per query is a safe batch size. We keep only USD values
(the budget / box-office numerics are returned as raw amounts; the
unit URI is checked separately).

Cache location: ``data/processed/v2/wikidata_cache.parquet``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://query.wikidata.org/sparql"
_USER_AGENT = "MSBA315-ScriptTriage/0.1 (academic project)"
_BATCH_SIZE = 50


def _build_query(imdb_ids: list[str]) -> str:
    """Build a VALUES-batched SPARQL query for a chunk of IMDb IDs."""
    values = " ".join(f'"{i}"' for i in imdb_ids)
    return f"""
    SELECT ?imdb_id ?budget ?budget_unit ?box_office ?box_office_unit WHERE {{
      VALUES ?imdb_id {{ {values} }}
      ?film wdt:P345 ?imdb_id .
      OPTIONAL {{
        ?film p:P2130 ?b_stmt .
        ?b_stmt psv:P2130 ?b_value .
        ?b_value wikibase:quantityAmount ?budget ;
                 wikibase:quantityUnit ?budget_unit .
      }}
      OPTIONAL {{
        ?film p:P2142 ?bo_stmt .
        ?bo_stmt psv:P2142 ?bo_value .
        ?bo_value wikibase:quantityAmount ?box_office ;
                  wikibase:quantityUnit ?box_office_unit .
      }}
    }}
    """


# Wikidata Q-id for "United States dollar". Other currencies (EUR, GBP)
# would need conversion; v2 keeps it simple by accepting only USD.
_USD_QID = "http://www.wikidata.org/entity/Q4917"


def _fetch_chunk(imdb_ids: list[str], timeout: int = 60) -> pd.DataFrame:
    """Fetch one batch of up to _BATCH_SIZE IDs."""
    query = _build_query(imdb_ids)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/sparql-results+json"}
    r = requests.post(
        _ENDPOINT,
        data={"query": query, "format": "json"},
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    rows: list[dict] = []
    for b in bindings:
        imdb_id = b.get("imdb_id", {}).get("value")
        budget_val = b.get("budget", {}).get("value")
        budget_unit = b.get("budget_unit", {}).get("value")
        bo_val = b.get("box_office", {}).get("value")
        bo_unit = b.get("box_office_unit", {}).get("value")
        rows.append({
            "imdb_id": imdb_id,
            "budget_usd": float(budget_val) if budget_val and budget_unit == _USD_QID else None,
            "revenue_usd": float(bo_val) if bo_val and bo_unit == _USD_QID else None,
        })
    return pd.DataFrame(rows)


def _coalesce_per_id(df: pd.DataFrame) -> pd.DataFrame:
    """One row per imdb_id: max-per-field over multiple Wikidata statements."""
    if df.empty:
        return df
    return (
        df.groupby("imdb_id", as_index=False)
          .agg({"budget_usd": "max", "revenue_usd": "max"})
    )


def fetch_wikidata(
    imdb_ids: list[str],
    cache_path: Path | None = None,
    refresh: bool = False,
    sleep_between_batches: float = 1.0,
) -> pd.DataFrame:
    """Fetch budget / box-office from Wikidata for a list of IMDb IDs.

    Returns a DataFrame with one row per requested ID and columns
    ``imdb_id, budget_usd, revenue_usd``. Cache is a parquet keyed by
    imdb_id with the same schema.
    """
    cache_path = cache_path or (paths.DATA_PROCESSED_DIR / "v2" / "wikidata_cache.parquet")
    requested = sorted(set(imdb_ids))

    if cache_path.is_file() and not refresh:
        cache = pd.read_parquet(cache_path)
        cached_ids = set(cache["imdb_id"])
        to_fetch = [i for i in requested if i not in cached_ids]
        logger.info(
            "Wikidata: %d requested, %d cached, %d to fetch",
            len(requested), len(requested) - len(to_fetch), len(to_fetch),
        )
    else:
        cache = pd.DataFrame(columns=["imdb_id", "budget_usd", "revenue_usd"])
        to_fetch = requested

    new_chunks: list[pd.DataFrame] = []
    for start in range(0, len(to_fetch), _BATCH_SIZE):
        chunk_ids = to_fetch[start:start + _BATCH_SIZE]
        try:
            chunk_df = _fetch_chunk(chunk_ids)
            new_chunks.append(chunk_df)
            logger.debug(
                "Wikidata batch %d–%d: %d hits",
                start, start + len(chunk_ids), len(chunk_df),
            )
        except requests.RequestException as exc:
            logger.warning("Wikidata batch %d failed: %s", start, exc)
        time.sleep(sleep_between_batches)

    # Coalesce, even if fetched-this-run is empty.
    if new_chunks:
        new_df = _coalesce_per_id(pd.concat(new_chunks, ignore_index=True))
        # Fill in NaN slots for IDs we asked for that came back empty,
        # so the cache marks them as "looked up".
        looked_up_ids = set()
        for ch in new_chunks:
            looked_up_ids.update(set(to_fetch[:len(ch)]))  # we don't track per-batch ids precisely; use to_fetch as the request set
        # Simpler: for any to_fetch id not in new_df, write a NaN row
        missing_ids = sorted(set(to_fetch) - set(new_df["imdb_id"]))
        if missing_ids:
            empty_rows = pd.DataFrame({
                "imdb_id": missing_ids,
                "budget_usd": [None] * len(missing_ids),
                "revenue_usd": [None] * len(missing_ids),
            })
            new_df = pd.concat([new_df, empty_rows], ignore_index=True)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset="imdb_id", keep="last").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
        logger.info(
            "Wikidata cache: saved %d rows (%d with budget, %d with revenue)",
            len(cache),
            int(cache["budget_usd"].notna().sum()),
            int(cache["revenue_usd"].notna().sum()),
        )

    return cache[cache["imdb_id"].isin(requested)].reset_index(drop=True)
