"""v2 corpus enrichment.

Adds new screenplays from ``data/data_enrichment/data/processed/`` and
recovers v1 MovieSum drops by enriching their financial data via the
TMDB API. Outputs are written to ``data/processed/v2/`` (or with a
``_v2`` suffix where co-located with v1 artifacts), so v1 stays
untouched.
"""
