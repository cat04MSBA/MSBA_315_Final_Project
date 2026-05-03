# Phase 3a — Baseline handoff

**Status:** Revised baseline complete; awaiting planning-conversation
sign-off on the new floor before Phase 3b.
**Date:** 2026-05-03
**Last revised:** 2026-05-03 — the planning conversation directed two
baseline changes: (a) ``log1p`` the heavy-tailed structural counts
before z-scoring, and (b) add ``log_runtime`` to the dialogue-only
baseline. Both applied; original (un-logged, no-runtime) numbers kept
in `phase3a_baseline.csv` for the report's before/after comparison.

This is an interim handoff at the Phase 3a / 3b boundary, per the
brief's instruction to "revert back to the planning conversation with a
summary of 3a and ask on next steps." It is **not** the Phase 3 final
summary (that comes at end of Task 5, using the standard template).

---

## Strategic decisions in scope

These were locked in by the planning conversation (Phase 3 brief §2)
and implemented directly:

- **Three targets in parallel.** `log_roi` (regression), `roi_gt_1` and
  `roi_gt_2` (classification). Threshold-consistent: `roi_gt_1 ==
  (log_roi > 0)` and `roi_gt_2 == (log_roi > ln 2)`.
- **Single train / cal / test split**, 70 / 15 / 15, seed 42, stratified
  by `(primary_genre_bucketed, decade_bucket)`.
- **Sub-phase boundary mandatory**: Phase 3a baselines complete before
  any Phase 3b feature engineering begins.
- **Calibration set carved here**, reserved for Phase 5 conformal
  prediction. Not touched in 3a.

## Tactical choices made in 3a

- **Decade bucketing for stratification:** `pre_1980s` (one stratum
  pooling 103 films across 1932-1979), `1980s`, `1990s`, `2000s`,
  `2010s_2020s` (one stratum because 2020-2023 is too thin alone).
- **Rare-cell pooling:** composite (genre, decade) cells with fewer
  than 5 films are pooled into a single `rare|rare` stratum so the
  stratified split is well-defined for every named cell. 38 films
  (2.2%) land in this pool.
- **Baseline model families:** `RidgeCV` for regression (alpha grid
  `np.logspace(-3, 3, 13)`, internal LOO-GCV alpha selection),
  `LogisticRegressionCV` with L2 for classification (C grid same
  spacing, inner 5-fold StratifiedKFold scoring AUC-ROC). Both wrapped
  in a `Pipeline` with `StandardScaler` on numeric features and
  passthrough on the genre one-hot dummies.
- **Cross-validation:** 5-fold on the **train split only** (n=1199).
  `KFold` for regression, `StratifiedKFold` for classification. OOF
  predictions concatenated; bootstrap 95% CIs computed on the OOF
  series (1000 iterations, percentile method).
- **Baseline feature set (original):** 7 structural metrics
  (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
  `total_dialogue_chars`, `total_action_chars`,
  `dialogue_to_total_text_ratio`, `parse_warning_count`),
  `release_year_parsed`, and `primary_genre_bucketed` one-hot (13
  dummies). Sanity-check baseline adds `log_budget` separately.
- **Baseline feature set (revised, planning conversation 2026-05-03):**
  same as the original, with two changes:
  1. `log1p` applied to the six heavy-tailed structural counts
     (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
     `total_dialogue_chars`, `total_action_chars`,
     `parse_warning_count`) before z-scoring; the bounded ratio
     `dialogue_to_total_text_ratio` is left untransformed.
     Transformed columns are renamed `log_<original>`.
  2. `log_runtime` added (computed inline as `log1p(runtime)`; the
     master parquet stores raw `runtime` in minutes only). Runtime is
     leak-free pre-greenlight: a script's intended runtime is
     implicit in page count, ~1 page per minute by industry
     convention.

## Headline results

Bootstrap 95% CIs in brackets. Full table at
`reports/tables/phase3a_baseline.csv` (28 rows: 4 feature sets × 3
targets × {3, 2, 2} metrics). The **revised** rows are the new
floor; the **original** rows are kept for the report's before/after
comparison.

### Revised dialogue-only baseline — NEW FLOOR (deployable)

| Target    | Metric  | Value | 95% CI         | Δ vs. original |
|-----------|---------|------:|----------------|---------------:|
| log_roi   | R²      | 0.052 | [0.026, 0.080] | +0.002 |
| log_roi   | MAE     | 0.948 | [0.896, 1.002] | -0.007 |
| log_roi   | RMSE    | 1.338 | [1.219, 1.471] | -0.001 |
| roi_gt_1  | AUC-ROC | 0.558 | [0.519, 0.599] | -0.001 |
| roi_gt_1  | PR-AUC  | 0.846 | [0.821, 0.874] | -0.001 |
| roi_gt_2  | AUC-ROC | 0.602 | [0.572, 0.636] | **+0.020** |
| roi_gt_2  | PR-AUC  | 0.739 | [0.709, 0.773] | **+0.016** |

### Revised with-budget sanity-check (not deployable)

| Target    | Metric  | Value | 95% CI         | Δ vs. original |
|-----------|---------|------:|----------------|---------------:|
| log_roi   | R²      | 0.099 | [0.041, 0.156] | **+0.021** |
| log_roi   | MAE     | 0.947 | [0.896, 0.996] | -0.014 |
| log_roi   | RMSE    | 1.305 | [1.202, 1.409] | -0.015 |
| roi_gt_1  | AUC-ROC | 0.555 | [0.515, 0.597] | -0.010 |
| roi_gt_1  | PR-AUC  | 0.845 | [0.819, 0.872] | -0.005 |
| roi_gt_2  | AUC-ROC | 0.603 | [0.573, 0.637] | **+0.032** |
| roi_gt_2  | PR-AUC  | 0.738 | [0.707, 0.773] | **+0.022** |

### Original dialogue-only baseline (kept for comparison)

| Target    | Metric  | Value | 95% CI         |
|-----------|---------|------:|----------------|
| log_roi   | R²      | 0.051 | [0.024, 0.080] |
| log_roi   | MAE     | 0.955 | [0.901, 1.010] |
| log_roi   | RMSE    | 1.340 | [1.224, 1.470] |
| roi_gt_1  | AUC-ROC | 0.559 | [0.521, 0.599] |
| roi_gt_1  | PR-AUC  | 0.847 | [0.823, 0.874] |
| roi_gt_2  | AUC-ROC | 0.582 | [0.552, 0.617] |
| roi_gt_2  | PR-AUC  | 0.723 | [0.692, 0.761] |

### Original with-budget sanity-check baseline (kept for comparison)

| Target    | Metric  | Value | 95% CI         |
|-----------|---------|------:|----------------|
| log_roi   | R²      | 0.078 | [0.030, 0.122] |
| log_roi   | MAE     | 0.961 | [0.909, 1.010] |
| log_roi   | RMSE    | 1.320 | [1.213, 1.432] |
| roi_gt_1  | AUC-ROC | 0.565 | [0.527, 0.604] |
| roi_gt_1  | PR-AUC  | 0.850 | [0.825, 0.877] |
| roi_gt_2  | AUC-ROC | 0.571 | [0.539, 0.605] |
| roi_gt_2  | PR-AUC  | 0.716 | [0.685, 0.755] |

## Threshold check

Applied to the **revised dialogue-only** floor (the new deployable
baseline). Brief's escalation thresholds (R² < 0.05 OR AUC-ROC < 0.55
across all targets) are **NOT tripped**:

- Regression R² = 0.052, just clear of the 0.05 floor.
- `roi_gt_1` AUC-ROC = 0.558, just clear of the 0.55 floor.
- `roi_gt_2` AUC-ROC = 0.602, **comfortably clear** (post-revision the
  cleanest of the three; CI lower bound 0.572 is fully above floor).

CI lower bounds are still below the point-estimate floors for the two
weaker targets (`log_roi` R² CI starts at 0.026; `roi_gt_1` AUC-ROC CI
starts at 0.519). The revision doesn't change that materially — the
floor passes on point estimate but the *CI lower bound* clears 0.55
only on `roi_gt_2`. Consistent with the planning conversation's
diagnosis that `roi_gt_1`'s 80% positive base rate makes it the
noisiest of the three targets.

## Interpretation

Five points worth flagging:

1. **The revision lift lands where the planning conversation
   predicted it would.** Headline post-revision movements on the
   deployable dialogue-only set: `log_roi` MAE drops 0.007 (modestly
   better calibration on a heavy-tailed target — exactly what the
   `log1p` was meant to buy), `roi_gt_2` AUC-ROC lifts 0.020 to
   0.602 (the only target whose CI lower bound now fully clears the
   floor), `roi_gt_2` PR-AUC lifts 0.016 to 0.739. R² lift is small
   (+0.002) and well within CI noise. With-budget shows a larger R²
   lift (+0.021 to 0.099); most of that gain comes from the log
   transforms interacting better with `log_budget`'s already-log scale.

2. **`roi_gt_1` actually gets marginally worse** after the revision
   (-0.001 AUC-ROC dialogue-only, -0.010 with-budget; both inside CI
   noise). Consistent with the planning conversation's diagnosis: at
   80% positive, the target is noise-dominated and a more honest
   feature scale doesn't help when the underlying signal is thin.

3. **Budget barely lifts deployable performance even after the
   revision.** With-budget revised R² (0.099) is roughly twice the
   deployable revised R² (0.052), but in absolute terms still
   negligible. AUC-ROC on classification targets is essentially
   unchanged by adding `log_budget`. Same conclusion as before: the
   corpus is heavily survivorship-filtered (80% gross-profitable), so
   films with different budgets share the "made it onto a major
   aggregator" selection. **Good news for project framing**: dialogue
   features in Phase 3b aren't competing against an obvious dominant
   budget signal.

4. **`roi_gt_2` is comfortably the most tractable of the three
   targets.** Post-revision AUC-ROC 0.602 with CI lower bound 0.572
   (the only target whose entire 95% interval sits above the brief's
   0.55 floor). The "net-profitable / blockbuster" distinction tracks
   observable features (Action and Animation lean blockbuster; small
   genres lean sub-2x) more cleanly than the gross-profitable
   distinction does. The planning conversation's note that "roi_gt_2
   and log_roi may emerge as the more tractable primary candidates"
   is already visible in the baseline.

5. **PR-AUC for `roi_gt_1` is misleading on its own.** 0.846 (revised
   dialogue-only) looks strong but the 80% positive base rate sets
   the random-guess floor at ~0.80. The lift over random is only ~4-5
   PR-AUC points either before or after the revision. For the
   80%-positive class, AUC-ROC is the more honest summary.

## Verification + revision (2026-05-03)

**Verification.** After the worktree → main consolidation (see
`.gitignore` change and the Phase 3a code/tables now living directly
under `src/features/`, `src/models/baseline/`, `data/processed/`,
`reports/tables/`), the pipeline was re-run end-to-end from the main
checkout to confirm nothing drifted in transit:

- `python -m src.features.split` → 1199 / 257 / 257, 57 strata, 38
  rare-pool films. Identical to the original worktree run.
- `python -m src.models.baseline.train` → all 14 original baseline
  rows reproduce to 4 decimals.

**Revision (planning-conversation directives 2026-05-03).** The
verification pass flagged `log_runtime` as omitted from the original
baseline; the planning conversation directed two changes:

1. **Add `log_runtime`** to the dialogue-only baseline (and by
   extension the with-budget sanity-check). Runtime is leak-free
   pre-greenlight (page count → minutes, ~1 page per minute by
   industry convention) so it belongs in the deployable set.
2. **Log-transform the heavy-tailed structural counts** before
   z-scoring. `log1p` applied to six of the seven structural columns
   (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
   `total_dialogue_chars`, `total_action_chars`,
   `parse_warning_count`); `dialogue_to_total_text_ratio` is left
   alone (already a bounded ratio). `log1p` because
   `parse_warning_count` is zero on the majority of films.

Both changes implemented as flags on `BaselineFeatureConfig`
(`log_transform_structural`, `include_log_runtime`); the trainer now
runs four feature sets (original × revised, each with and without
`log_budget`) and writes all 28 rows to `phase3a_baseline.csv`.
Original numbers preserved verbatim so the report can show the
before/after.

## Files produced

### Code
- `src/features/__init__.py`
- `src/features/split.py` — split logic, `SplitConfig` dataclass,
  CLI runner
- `src/features/targets.py` — `add_targets`, target name constants
- `src/features/baseline_features.py` — `build_baseline_features`,
  `BaselineFeatureConfig`
- `src/models/__init__.py`, `src/models/baseline/__init__.py`
- `src/models/baseline/metrics.py` — metric registry, bootstrap CI
  helper
- `src/models/baseline/train.py` — orchestrator + CLI runner

### Data
- `data/processed/split_assignments.parquet` — one row per film:
  `imdb_id`, `stratum`, `split` ∈ {train, cal, test}

### Tables
- `reports/tables/phase3_split_diagnostics.csv` — per-stratum split
  counts (57 strata; every stratum has ≥1 film in each split)
- `reports/tables/phase3a_baseline.csv` — full baseline table (14
  rows: 2 feature sets × 3 targets × {3, 2, 2} metrics)

## Resolved questions (planning conversation 2026-05-03)

The planning conversation reviewed the original handoff and resolved
every question except primary-outcome choice (formally deferred).
Outcomes recorded here for the project audit trail:

1. **Revisit the baseline before Phase 3b — RESOLVED.** Apply
   `log1p` to the heavy-tailed structural counts before z-scoring,
   add `log_runtime` as a deployable feature, re-run both baselines,
   keep originals in the table for comparison. **Implemented.** New
   numbers in the Headline-results section above. Threshold check
   passes; `roi_gt_2` is the cleanest target post-revision.

2. **Primary outcome — DEFERRED per brief.** All three targets
   continue through Phase 3b. The planning conversation flagged
   `roi_gt_1` as likely the noisiest (80% positive); the post-
   revision numbers confirm — `roi_gt_2` shows the strongest lift,
   `roi_gt_1` actually nudged down. Formal decision still at end of
   Phase 4.

3. **`data_quality_flag` films (n=30) — RESOLVED.** Plan confirmed:
   keep all 30 films in train / cal / test for sample size; apply
   per-feature handling in Phase 3b — features that depend on scene-
   level integrity (character network, scene-aware sentiment) exclude
   or downweight flagged films, features robust to scene-level issues
   (whole-screenplay lexical, aggregate sentiment) use them as-is.
   Per-group handling will be documented in `FEATURE_NOTES.md` as
   each group lands.

4. **Group ordering for Phase 3b — RESOLVED.** Lexical + Sentiment
   together (shared preprocessing), then Topic, then Character
   network, embeddings last (most expensive). Matches the brief's
   suggested order.

5. **`log_runtime` omission — RESOLVED.** Add now. Folded into the
   Q1 baseline re-run as a single change. **Implemented.**

## Next step

Planning conversation has asked to confirm the new floor before Phase
3b starts. Numbers above. Once confirmed, Phase 3b kicks off with the
**Lexical proposal** at `docs/proposals/phase3_lexical_proposal.md`
(per the brief, the proposal precedes implementation and is reviewed
by the planning conversation against the literature-reference doc).
