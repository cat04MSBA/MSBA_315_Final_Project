# Phase 4 — v2 corpus addendum

**Status:** v2 build complete; **comparison says do NOT promote**.
**Date:** 2026-05-04

This document records the v2-corpus enrichment experiment run in
parallel with the Phase 5 calibration work. v2 outputs live under
``data/processed/v2/`` so v1 artifacts and Phase 5 work were untouched.

---

## What we built

* **v1 corpus**: 1,713 films (kept verbatim).
* **v2 candidate pool**: 989 IMDb IDs from the user's
  ``unified_scripts.jsonl`` that are not already in v1
  (514 pure-sgogoi adds + 475 v1-MovieSum-drops).
* **Financial enrichment** (gated *before* parsing — only films that
  ended up four-signal-complete were ever parsed):
  * Kaggle pass via the user's ``corpus_final.jsonl``: **+230** sgogoi
    films four-signal-complete.
  * TMDB API pass on the remaining 759 (only budget / revenue /
    vote_average captured per user instruction): **+128**.
  * Wikidata SPARQL fallback on the residual 631
    (USD-quantity-only, joined with TMDB rating): **+15**.
  * **Stopped** the fallback chain here — Wikidata yield 2.4%, OMDb
    has no budget, and scraping Box Office Mojo / The Numbers
    wasn't worth the engineering.
* **v2 corpus**: **2,086 films** (1,713 + 373 survivors), **+21.8%**
  over v1.
* **Year range**: 1927 – 2023 (v1: 1932 – 2023; v2 picks up older /
  newer titles via sgogoi).
* **Median budget** $22M (v1: $25M); **median revenue** $55M
  (v1: $64M); the new films are slightly smaller-budget on average.
* **v2 train / cal / test**: 1,460 / 313 / 313 (v1: 1,199 / 257 / 257).
  Re-carved fresh under the same Phase 3a stratified-shuffle scheme.

---

## Verdict: don't promote v2

v2 hurts the headline `roi_gt_2` AUC by **-0.038** on the v1 winner cell
(SVM-RBF / standalone_positive_union_mpnet → XGBoost / same on v2,
0.6520 → 0.6137).

Per-cell deltas (v2 OOF − v1 OOF on the matched cell):

| Target | Encoder | Matrix | v1 | v2 | Δ |
|---|---|---|---:|---:|---:|
| `log_roi` (RMSE ↓) | mpnet | standalone | 1.3102 | **1.3038** | **-0.006 (better)** |
| `log_roi` (RMSE ↓) | minilm | standalone | 1.3164 | 1.3029 | -0.014 (better) |
| `roi_gt_1` (AUC ↑) | mpnet | standalone | 0.6353 | 0.6172 | **-0.018 (worse)** |
| `roi_gt_1` (AUC ↑) | minilm | standalone | 0.6159 | 0.6276 | +0.012 (better) |
| `roi_gt_2` (AUC ↑) | mpnet | standalone | **0.6520** | 0.6137 | **-0.038 (worse)** |
| `roi_gt_2` (AUC ↑) | minilm | standalone | 0.6346 | 0.6049 | -0.030 (worse) |

**Headline:** the v2 lift is **negative on the headline target**
despite a 22% larger train set. The new films contribute more noise
than signal:

* Their parses are lower quality on average — sgogoi screenplays use
  inconsistent formatting, and the "moviesum"-tagged 2023+ additions
  in ``unified_scripts.jsonl`` are pre-formatted plain text rather
  than the canonical MovieSum XML.
* Their financials come from heterogeneous sources (Kaggle / TMDB API /
  Wikidata) with different budget-recording conventions.
* The header ``script_char_len`` field for survivors is filled from
  the parsed metrics, not from MovieSum's authoritative count.

In other words: **v1 had the better signal-to-noise ratio**. Larger ≠
better when the marginal data is noisier than the existing corpus.

Recommendation:

1. **Do not promote v2 to canonical.** Phase 5 should continue against
   v1's ``phase4_primary_model_*.joblib`` artifacts.
2. The v2 artifacts on disk are fine to keep as an audit trail for the
   eventual Phase 9 report's "we tried adding sgogoi screenplays — it
   didn't help" subsection.
3. If a future iteration wants to revisit, the lever to pull is
   **higher-fidelity sgogoi parsing** (run the v1-style XML parser on
   re-templated sgogoi text, or re-extract scenes via a more uniform
   parser) rather than corpus size per se.

---

## Files produced (all under v2 sub-paths or with `_v2` suffix)

* Code:
  * ``src/data/enrichment/load_unified_scripts.py``
  * ``src/data/enrichment/match_financials.py`` (TMDB API)
  * ``src/data/enrichment/wikidata_fallback.py`` (Wikidata SPARQL)
  * ``src/data/enrichment/adapter.py`` (unified → ParsedScreenplay)
  * ``src/data/enrichment/build_corpus_v2.py``
  * ``src/experiments/run_v2_enrichment.py``
  * ``src/experiments/run_v2_features.py``
  * ``src/experiments/run_phase4_benchmark_v2.py``
  * ``src/experiments/compare_v1_v2.py``
* Data (under ``data/processed/v2/``):
  * ``imdb_api_cache.parquet`` (759 rows; 603 OK, 156 not-on-TMDB)
  * ``wikidata_cache.parquet`` (631 rows; 45 with budget, 41 with revenue)
  * ``candidate_financials.parquet`` (373 rows)
  * ``enrichment_summary.json``
  * ``films_joined_v2.parquet`` (2,086 × 41)
  * ``screenplays_parsed_v2.pkl`` (290 MB)
  * ``split_assignments_v2.parquet`` (2,086 × 3)
  * ``features_lexical_v2.parquet`` (2,086 × 14)
  * ``features_sentiment_v2.parquet`` (2,086 × 24)
  * ``features_topic_v2.parquet`` (2,086 × 22)
  * ``features_character_network_v2.parquet`` (2,086 × 13)
  * ``features_embedding_v2.parquet`` (2,086 × 32)
  * ``features_embedding_mpnet_v2.parquet`` (2,086 × 32)
  * ``embeddings_minilm_pooled_v2.parquet`` (2,086 × 384)
  * ``embeddings_mpnet_pooled_v2.parquet`` (2,086 × 768)
  * ``embedding_pca_v2.joblib``, ``embedding_pca_mpnet_v2.joblib``
  * ``topic_model_artifacts_v2/`` (vectorizer + LDA + train ids)
  * ``features_v2.parquet``, ``features_v2_mpnet.parquet`` (consolidated)
  * ``phase4_primary_model_<target>_v2.joblib`` (mpnet winners)
* Tables (under ``reports/tables/``):
  * ``phase4_benchmark_v2.csv`` (432 rows)
  * ``phase4_benchmark_mpnet_v2.csv`` (432 rows)
  * ``phase4_paired_tests_v2.csv`` (360 rows)
  * ``phase4_paired_tests_mpnet_v2.csv`` (360 rows)
  * ``phase4_v1_vs_v2_comparison.csv`` (12 rows; the headline)
  * ``phase3_split_diagnostics_v2.csv``
* Run dirs: ``runs/phase_4_v2/`` (24 cells; v1 ``runs/phase_4/`` untouched)

---

## What we deliberately skipped

* Stacking, posthoc, diagnostic, importance, error-analysis on v2 —
  none of those would change the verdict, and the headline benchmark
  already says "don't promote".
* Pre-1980 / per-genre slicing on v2 — same reason.
* DistilBERT fine-tune — out of scope per Phase 4 pre-registration.
* OMDb / Box Office Mojo / The Numbers fallbacks — diminishing returns
  after Wikidata yielded only +15.
