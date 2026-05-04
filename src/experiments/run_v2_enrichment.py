"""v2 corpus enrichment driver.

Steps:

1. Load v1 ``films_joined.parquet`` and the user's ``unified_scripts``.
2. Compute candidate pools (unified IDs not in v1).
3. Reuse ``corpus_final.jsonl`` as the Kaggle pass output (the user's
   notebook already ran it; 230 sgogoi films are pre-enriched).
4. Run the TMDB API for every candidate not yet four-signal-complete.
5. Report counts of survivors per source and per pool, write
   ``data/processed/v2/enrichment_summary.json``.

Run from the project root::

    python -m src.experiments.run_v2_enrichment

Idempotent — the API cache at ``data/processed/v2/imdb_api_cache.parquet``
makes re-runs free (only fetches IDs not yet cached).
"""

from __future__ import annotations

# Allow running by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import pandas as pd

from src.data.enrichment.load_unified_scripts import (
    compute_candidate_pools,
    load_corpus_final,
    load_unified_scripts,
)
from src.data.enrichment.match_financials import TMDBConfig, enrich_via_api
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> dict:
    """Run the enrichment pipeline; return the summary dict."""
    out_dir = paths.DATA_PROCESSED_DIR / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load v1 + unified.
    v1 = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    unified = load_unified_scripts()
    pools = compute_candidate_pools(v1, unified)
    candidates = sorted(pools.candidates)
    logger.info(
        "Pools: v1=%d, unified=%d, candidates=%d "
        "(new_sgogoi=%d, v1_moviesum_drops=%d)",
        len(pools.v1_ids), len(pools.unified_ids), len(candidates),
        len(pools.new_sgogoi_only), len(pools.v1_moviesum_drops),
    )

    # 2. Reuse user's Kaggle pass via corpus_final.jsonl.
    corpus_final = load_corpus_final()
    cf_ids_with_4 = set(corpus_final["imdb_id"])
    kaggle_admit = sorted(set(candidates) & cf_ids_with_4)
    logger.info(
        "Kaggle pass (from corpus_final.jsonl): %d/%d candidates have all 3 "
        "financial signals", len(kaggle_admit), len(candidates),
    )

    # 3. API pass on the rest.
    needs_api = sorted(set(candidates) - cf_ids_with_4)
    logger.info("API pass: %d candidates remaining", len(needs_api))
    api_results = enrich_via_api(needs_api, TMDBConfig.resolve())

    # 4. Build a unified financials table for survivors.
    # Rows from corpus_final → already four-signal complete.
    cf_for_candidates = corpus_final[corpus_final["imdb_id"].isin(kaggle_admit)].copy()
    cf_for_candidates["enrichment_source"] = "kaggle"

    # Rows from API → keep only those with all three financial signals.
    api_ok = api_results[
        api_results["budget"].notna()
        & api_results["revenue"].notna()
        & api_results["vote_average"].notna()
    ].copy()
    api_ok["enrichment_source"] = "tmdb_api"

    api_combined = pd.DataFrame({
        "imdb_id": pd.concat([cf_for_candidates["imdb_id"], api_ok["imdb_id"]], ignore_index=True),
        "budget": pd.concat([cf_for_candidates["budget"], api_ok["budget"]], ignore_index=True),
        "revenue": pd.concat([cf_for_candidates["revenue"], api_ok["revenue"]], ignore_index=True),
        "vote_average": pd.concat([cf_for_candidates["vote_average"], api_ok["vote_average"]], ignore_index=True),
        "enrichment_source": pd.concat(
            [cf_for_candidates["enrichment_source"], api_ok["enrichment_source"]],
            ignore_index=True,
        ),
    })
    api_combined.to_parquet(out_dir / "candidate_financials.parquet", index=False)

    # 5. Report.
    survivors = set(api_combined["imdb_id"])
    sgogoi_survivors = survivors & pools.sgogoi_unified_ids
    moviesum_survivors = survivors & pools.moviesum_unified_ids
    api_failed = [i for i in needs_api if i not in survivors]

    summary = {
        "pools": {
            "v1_ids": len(pools.v1_ids),
            "unified_ids": len(pools.unified_ids),
            "candidates_total": len(candidates),
            "new_sgogoi_only": len(pools.new_sgogoi_only),
            "v1_moviesum_drops": len(pools.v1_moviesum_drops),
        },
        "kaggle_pass": {
            "admitted": len(kaggle_admit),
            "of_total_candidates": len(candidates),
        },
        "api_pass": {
            "requested": len(needs_api),
            "admitted": int(len(api_ok)),
            "failed": len(api_failed),
        },
        "survivors_total": len(survivors),
        "survivors_by_source": {
            "sgogoi": len(sgogoi_survivors),
            "moviesum": len(moviesum_survivors),
        },
        "v2_corpus_size_estimate_pre_filter": len(pools.v1_ids) + len(survivors),
    }

    summary_path = out_dir / "enrichment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary → %s", summary_path)

    print("\n=== v2 enrichment summary ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
