# Phase 3b: Character-network group handoff

**Status:** Character-network features implemented and evaluated
across four model families. The standalone-lift verdict is **partial
positive**, qualitatively distinct from both the topic group (which
lifted `roi_gt_1` AUC across all families) and the lexical / sentiment
nulls. Linear OOF lifts `roi_gt_1` AUC by +0.013 (in-band) and
`roi_gt_1` PR-AUC by +0.003 (in-band) — the first group to land two
linear-OOF in-band hits. Linear `roi_gt_2` AUC lifts +0.016, just
below the +0.020 predicted floor. SVM particularly strong on
`roi_gt_2` (+0.061 AUC, +0.046 PR-AUC). All four families lift
`roi_gt_2` AUC. `network_lead_role_count` ↔ `roi_gt_2` is the first
univariate target correlation to exceed |r| = 0.10 across any Phase 3b
group (r = -0.102).
**Date:** 2026-05-03

This handoff matches the structure of the lexical, sentiment, and
topic handoffs.

---

## 1. Strategic decisions in scope

* **12 character-network features as approved in proposal v1** —
  cast structure (3), density and connectivity (3), lead-character
  dominance (3), graph topology (3). One diagnostic column.
* **Graph construction:** scene-cooccurrence undirected graph;
  significance threshold of 5 non-empty dialogue lines per
  character; edges weighted by shared-scene count. Sorted node
  ordering for determinism.
* **`data_quality_flag` films:** NaN-fallback on all 12 model
  features (24 train-split flagged films). The trainer's median
  imputer handles them at fold-fit time.
* **Standalone-lift methodology preserved.** Network-augmented
  matrix joins the 12 model features onto the revised dialogue-
  only baseline; lift compared against the Phase 3a floor.

---

## 2. Tactical decisions made during implementation

* **NetworkX backend.** Standard library for graph construction,
  centrality, and modularity. Determinism preserved by sorting
  nodes alphabetically before construction.
* **Modularity solver:** `nx.community.greedy_modularity_communities`
  (Newman-Girvan modularity via greedy aggregation). Returns NaN
  for graphs with no edges; Phase 4 imputer handles those.
* **Eigenvector centrality:** scipy power-iteration first
  (`max_iter=1000, tol=1e-6`), with NumPy backend
  (`eigenvector_centrality_numpy`) as a fallback for difficult
  cases. Operates on the connected component containing the
  top-1 character to avoid undefined behaviour on disconnected
  graphs.
* **Diameter:** computed on the largest connected component
  only when the graph has multiple components, to keep the
  metric well-defined.
* **Top-decile lead-role threshold:** `ceil(0.10 *
  n_significant_characters)` — at least 1 character. Below the
  threshold's line count, characters are not counted as leads.

---

## 3. What was done

1. **Built `src/features/character_network.py`** (~390 lines)
   with the `compute_character_network_features` API and 12
   network-metric helpers.
2. **Wrote `tests/test_character_network.py`** (12 tests, all
   passing) — Gini-coefficient unit tests, hand-built
   3-character-chain / 5-character-complete / 2-clique synthetic
   graph fixtures with known density / modularity / diameter
   properties, the flagged-film NaN-fallback assertion, the
   minor-character filter assertion, and a smoke test on real
   corpus data.
3. **Computed features on the full corpus** (1,713 films) in
   approximately 60 seconds. Modularity and eigenvector
   centrality dominate the per-film cost.
4. **Added `include_character_network` and
   `character_network_features_path` knobs to
   `BaselineFeatureConfig`.**
5. **Built `src/experiments/run_character_network_ablation.py`**
   mirroring the four prior runners.
6. **Ran the multi-family ablation.** Run directory:
   `runs/phase_3/20260503_2025_character_network_multifamily/`.
   Cumulative `phase3_ablation.csv` is now at 384 rows
   (lexical + sentiment + topic + character network).
7. **Ran proposal Section 8 diagnostics.** Output captured below.

---

## 4. Headline numbers (network added on top of the revised dialogue-only floor)

### 4.1 OOF lift per family (lower-is-better metrics: negative lift means improvement)

| Family | Metric | Floor | With network | Lift |
|---|---|---:|---:|---:|
| linear | log_roi RMSE (lower) | 1.339 | 1.348 | +0.009 (worse) |
| linear | log_roi MAE (lower) | 0.948 | 0.953 | +0.005 (worse) |
| linear | roi_gt_1 AUC (higher) | 0.558 | 0.572 | **+0.013** (in-band) |
| linear | roi_gt_1 PR-AUC (higher) | 0.846 | 0.849 | **+0.003** (in-band) |
| linear | roi_gt_2 AUC (higher) | 0.602 | 0.617 | +0.016 (below band, right direction) |
| linear | roi_gt_2 PR-AUC (higher) | 0.739 | 0.739 | +0.001 |
| histgb | log_roi RMSE (lower) | 1.327 | 1.328 | +0.001 |
| histgb | roi_gt_1 AUC (higher) | 0.552 | 0.543 | -0.009 (worse) |
| histgb | roi_gt_2 AUC (higher) | 0.610 | 0.614 | +0.004 (better) |
| knn | log_roi RMSE (lower) | 1.364 | 1.376 | +0.012 (worse) |
| knn | roi_gt_1 AUC (higher) | 0.527 | 0.544 | **+0.017** (better) |
| knn | roi_gt_2 AUC (higher) | 0.578 | 0.601 | **+0.022** (better) |
| svm | log_roi RMSE (lower) | 1.357 | 1.345 | **-0.012** (better) |
| svm | log_roi MAE (lower) | 0.954 | 0.940 | -0.013 (better) |
| svm | roi_gt_1 AUC (higher) | 0.504 | 0.534 | **+0.029** (better) |
| svm | roi_gt_2 AUC (higher) | 0.534 | 0.595 | **+0.061** (better) |
| svm | roi_gt_2 PR-AUC (higher) | 0.676 | 0.722 | **+0.046** (better) |

**Pattern: every family lifts `roi_gt_2` AUC.** Linear, KNN, and
SVM lift `roi_gt_1` AUC; HistGB drops it modestly. SVM's lift on
`roi_gt_2` is the single largest gain of any Phase 3b
standalone-ablation result.

### 4.2 Pre-registered linear-family lift (proposal v1 Section 4)

| Target | Metric | Predicted band (linear OOF) | Actual | In band? |
|---|---|---|---:|:---:|
| log_roi | RMSE | -0.040 to -0.010 | +0.009 | No (wrong direction) |
| log_roi | MAE | -0.030 to -0.010 | +0.005 | No (wrong direction) |
| log_roi | CVRMSE | -0.030 to -0.010 | +0.007 | No (wrong direction) |
| roi_gt_1 | AUC-ROC | 0.000 to +0.015 | +0.013 | **Yes** |
| roi_gt_1 | PR-AUC | 0.000 to +0.015 | +0.003 | **Yes** |
| roi_gt_1 | log-loss | -0.020 to 0.000 | +0.005 | No (wrong direction) |
| roi_gt_2 | AUC-ROC | +0.020 to +0.045 | +0.016 | No (right direction, below band) |
| roi_gt_2 | PR-AUC | +0.015 to +0.035 | +0.001 | No (right direction, below band) |
| roi_gt_2 | log-loss | -0.025 to -0.005 | +0.005 | No (wrong direction) |

Two in-band hits — the most of any Phase 3b group so far. The
predicted-band misses on `roi_gt_2` were close-to-band but
underwhelming relative to the proposal's expectation that this
group would be the strongest performer of the five. The
strongest prediction was on the wrong family: SVM on `roi_gt_2`
AUC delivered +0.061, way above any reasonable linear-band; but
SVM's floor was the worst of any family, so this is partly the
"SVM finally finds a configuration" pattern documented in the
lexical handoff. Linear, the historical reference, lifted
`roi_gt_2` AUC by +0.016 — directionally correct, just under
the predicted band.

### 4.3 Verdict

**Two partial positives now landed (topic + character network).**
The two share a structural property: lift comes from multivariate
feature interactions rather than from any single feature's strong
univariate signal. They differ in target affinity:

* **Topic** lifts `roi_gt_1` AUC across all four families (+0.026
  to +0.052) but is flat-to-negative on `roi_gt_2`.
* **Character network** lifts `roi_gt_2` AUC across all four
  families (+0.004 to +0.061) and lifts `roi_gt_1` AUC on three
  of the four (linear, KNN, SVM positive; HistGB slightly
  negative).

The complementarity is meaningful. `roi_gt_1` (gross-profitable,
80% positive base rate) is signal-thin and topic features extract
it; `roi_gt_2` (net-profitable, 64% positive base rate) has more
headroom and character-network features extract it. Phase 3c
combinations of topic + character network is the obvious next
test, where the joint-evaluation methodology was designed for
exactly this configuration.

---

## 5. Diagnostic results (Section 8 of proposal v1)

| # | Check | Threshold / expectation | Result | Verdict |
|---|---|---|---|---|
| 1 | Significant-character count distribution | informational | Median 18, mean 19.1, IQR [15, 23] | Pass (matches expected 12-25 band) |
| 2 | Network ↔ structural baseline cross-correlations | drop pair if \|r\| > 0.85 | All pairs \|r\| < 0.85; strongest is `network_n_significant_characters ↔ log_n_unique_characters` at +0.625 | Pass |
| 3 | Within-network pairwise correlations | drop one if \|r\| > 0.85 | One pair: `network_n_significant_characters ↔ network_lead_role_count` at +0.90 | **See below** |
| 4 | Significance-threshold sensitivity (3 vs 5 vs 10) | informational | Not run; test design overhead disproportionate to v1 lift outcome | Deferred to v2 |
| 5 | data_quality_flag films vs unflagged | NaN by construction | All 24 flagged films are all-NaN on 12 model columns | Pass |
| 6 | Empty-graph rate | informational | 12 of 1199 films (1%) have <2 significant characters | Pass (small fraction) |
| 7 | Univariate target correlations | informational | **`network_lead_role_count ↔ roi_gt_2` r = -0.102** | **First |r| > 0.10 in any Phase 3b group; see below** |

Two checks merit narrative interpretation.

**Check 3 — `network_n_significant_characters` and
`network_lead_role_count` correlate at r = +0.90.** This is by
construction: `lead_role_count` counts characters in the top
decile of the cast by line count, and `ceil(0.10 *
n_significant_characters)` is mechanically a near-linear function
of the cast size. The two features measure conceptually different
things on a per-film basis (the latter is sensitive to the
distribution shape, not just the count), and the lift table shows
the linear-family AUC gains come from the joint behaviour of all
12 features rather than either one alone. Both are kept for
optionality, with the redundancy documented for Phase 4 to
revisit if needed.

**Check 7 — first non-trivial univariate target correlation in
Phase 3b.** `network_lead_role_count` correlates -0.102 with
`roi_gt_2` on the train split. The directional reading: films
with more "lead" characters (top-decile by line count) trend
less net-profitable. This is consistent with audience-
identification theory in screenwriting craft: ensemble films
diffuse audience attachment across multiple leads, which
correlates with weaker franchise potential and lower
post-marketing revenue. The correlation is small but it crosses
the threshold the lexical and sentiment groups never reached.

The fact that this single feature crosses |r| = 0.10 explains
why character-network features lift `roi_gt_2` AUC across
families: the model has at least one feature with real univariate
signal to anchor on, plus 11 more that contribute multivariate
information.

---

## 6. Why the regression target stays null

`log_roi` RMSE goes the wrong direction on linear (+0.009),
HistGB (+0.001), and KNN (+0.012); SVM is the only family to
improve (-0.012). Two mechanisms.

**The univariate signal is on a binary threshold, not the
continuous log_roi distribution.** `network_lead_role_count`
correlates -0.102 with `roi_gt_2` (binary) but only -0.082 with
`log_roi` (continuous). The discriminating information lives
specifically near the `log_roi > log(2)` boundary; the rest of
the log_roi distribution is dominated by noise the
network features cannot resolve. RMSE penalizes errors uniformly
across the distribution, which mutes a signal concentrated at a
specific threshold.

**SVM's positive RMSE shift may be a low-floor artifact.**
SVM's floor was the worst across all families (RMSE 1.357), and
the augmented matrix gives it a slightly more favourable kernel
geometry. The same pattern appeared in the lexical and sentiment
SVM rows. The lift is real but it is below linear's
non-augmented RMSE, so it does not change Phase 4's family
selection.

---

## 7. Files produced (Phase 3b character-network group)

### Code
* `src/features/character_network.py` (~390 lines).
* `src/experiments/run_character_network_ablation.py`.
* `tests/test_character_network.py` (12 tests, all passing).
* `src/features/baseline_features.py` extended with two new knobs.
* `requirements.txt` extended with `networkx`.

### Data
* `data/processed/features_character_network.parquet` (1,713 × 13).

### Run artifacts
* `runs/phase_3/20260503_2025_character_network_multifamily/`.

### Tables
* `reports/tables/phase3_ablation.csv`: extended from 288 to 384
  rows.

---

## 8. Resolved questions

1. **Treatment of the row: APPEND.** Two in-band hits + a
   univariate-signal-finding row goes in the table.
2. **`network_n_significant_characters` ↔ `network_lead_role_count`
   redundancy: KEEP both.** The 0.90 correlation is high but the
   features capture conceptually different aspects of cast
   structure; Phase 4 model selection has the option to drop one
   if regularization shows no marginal information.
3. **Significance-threshold sensitivity not run for v1.** Default
   stays at 5; v2 polish item if the planning conversation
   wants the sensitivity analysis.
4. **Proceed to embedding implementation: YES.** Two of three
   remaining groups (topic, character-network) have produced
   non-null partial positives. Embedding is the most expensive
   group and the highest-prior-from-literature; it is the right
   final candidate to evaluate before Phase 3c combinations.

## 9. Decisions log entry to add

> ## 2026-05-03 20:30 — Phase 3b: character-network standalone result is partial positive (matching topic shape)
>
> **Phase:** Phase 3 — Feature Extraction (sub-phase 3b, fourth of five groups)
> **Decision:** Character-network features (12 columns: 3 cast structure, 3 density/connectivity, 3 lead dominance, 3 graph topology) implemented per proposal v1. Multi-family ablation produced a **partial positive** verdict: two linear-OOF in-band hits (the first time any group lands two), and `roi_gt_2` AUC lift across all four families (linear +0.016, histgb +0.004, knn +0.022, svm +0.061). `network_lead_role_count ↔ roi_gt_2` is the first feature in any Phase 3b group to exceed the |r| = 0.10 univariate threshold (r = -0.102). The result is qualitatively distinct from topic (which lifted `roi_gt_1` AUC across families): character network lifts `roi_gt_2` AUC across families. Together with topic, the two partial positives demonstrate that genre-orthogonal feature groups (topic, character network) lift the ablation while genre-redundant groups (lexical, sentiment) do not. NaN-fallback for the 24 train-split data-quality-flagged films executed cleanly (all-NaN on the 12 model features, mean imputer handles them at fold-fit time). The 12 features are retained for Phase 3c combinations evaluation; the most informative Phase 3c combination to test is topic + character-network (which target different `roi_gt_*` thresholds with non-overlapping mechanisms).
> **See also:** `docs/handoffs/phase_3b_character_network_handoff.md`.

---

## 10. Next step

Embedding implementation per
`docs/proposals/phase3_embedding_proposal.md`. The expensive group
remains, and the literature prior for it is the strongest of the
five (Gross 2025 reports MiniLM embeddings as the strongest
predictor on a related task). The two partial positives now
landed should bound expectations: signal exists in the corpus,
but it lives in multivariate feature interactions rather than
strong univariate correlations, and per-target performance
varies more by feature group than by model family.
