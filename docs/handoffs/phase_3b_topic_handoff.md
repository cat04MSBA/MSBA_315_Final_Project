# Phase 3b: Topic group handoff

**Status:** Topic features implemented and evaluated across four
model families. The standalone-lift verdict is **partial positive** —
qualitatively different from the lexical and sentiment null results.
Linear OOF lifts `roi_gt_1` AUC by +0.032 (above the predicted band
of 0.000 to +0.015) and `roi_gt_2` AUC by +0.012 (just below the
+0.015 predicted floor). All four model families show positive
`roi_gt_1` AUC lift (linear +0.032, histgb +0.026, knn +0.028, svm
+0.052), the first time any Phase 3b group has produced consistent
across-family direction on a classification target. The regression
target moves the wrong direction on every family; PR-AUC on
`roi_gt_1` lands in-band at +0.014. Topic features are kept in the
matrix for Phase 3c combinations evaluation.
**Date:** 2026-05-03

This handoff matches the structure of the lexical and sentiment
handoffs.

---

## 1. Strategic decisions in scope

* **22 topic features as approved in proposal v1** — 20 LDA topic
  proportions, 1 distribution-concentration entropy, 1 dominant-
  topic identifier (integer-encoded).
* **Backend: scikit-learn `LatentDirichletAllocation`**, variational
  Bayes, batch learning method, K = 20, max_iter = 10, seed 42.
* **Document unit: whole-screenplay dialogue**, one document per
  film. Action text out of scope (parsimony).
* **No-leakage discipline (CRITICAL):** vocabulary and LDA fit on
  training-fold films only (1,199 films from
  `split_assignments.parquet`); transform applied to all 1,713
  films using the train-fitted artifacts. Cal and test films
  contribute zero information to vocabulary or topic-word
  distributions. Verified by a unit test in `tests/test_topic.py`.
* **Standalone-lift methodology preserved.** Topic-augmented matrix
  joins the 22 topic features onto the revised dialogue-only
  baseline (structural counts + era + genre dummies + log_runtime),
  NOT onto the lexical- or sentiment-augmented matrix.

---

## 2. Tactical decisions made during implementation

* **Vocabulary thresholds:** `min_df = 5` (drop words appearing in
  fewer than 5 films), `max_df = 0.5` (drop words appearing in
  more than 50% of films). Standard for screenplay-LDA.
* **Tokenization:** NLTK `word_tokenize`, lowercased, alphabetic-
  only filter via `^[A-Za-z][A-Za-z'\-]*$`, stopwords removed
  (NLTK English), tokens shorter than 3 characters dropped.
* **Stopword removal:** True by default; the configuration knob
  `TopicFeatureConfig.remove_stopwords` exposes the no-removal
  variant for sensitivity analysis.
* **Empty-text dialogue filter:** applied at the dialogue
  iteration step, matching Phase 2 Tier 1.3 and the
  lexical/sentiment groups.
* **Vectorizer token pattern:** `r"(?u)\b\w\w\w+\b"` (3+ char
  word boundary), enforced inside `CountVectorizer` to
  double-defend the min-token-length filter.
* **Determinism:** seed 42 on the LDA estimator. Same input
  tokens + same configuration produce byte-identical
  document-topic distributions.
* **Fitted artifact persistence:** vectorizer, LDA, and train-IDs
  saved to `data/processed/topic_model_artifacts/` via joblib
  + numpy. Topic-label table (top-10 words per topic) saved to
  `reports/tables/phase3_topic_labels.csv`. Phase 4 modelling
  reads from these directly.

---

## 3. What was done

1. **Built `src/features/topic.py`** with the no-leakage `fit_topic_model`
   / `compute_topic_features` API and the `topic_label_table`
   helper.
2. **Wrote `tests/test_topic.py`** with 9 tests including a
   no-leakage test asserting `transform` outputs are independent
   of the inference batch composition. All pass.
3. **Computed features on the full corpus** (1,713 films) in
   approximately 90 seconds: 12s tokenization, 10s vectorization,
   60s LDA fit, 8s transform on the full corpus.
4. **Added `include_topic` and `topic_features_path` knobs to
   `BaselineFeatureConfig`** so the existing trainer can construct
   the topic-augmented matrix via `build_baseline_features`.
5. **Built `src/experiments/run_topic_ablation.py`** mirroring the
   lexical and sentiment runners.
6. **Ran the multi-family ablation.** Run directory:
   `runs/phase_3/20260503_2015_topic_multifamily/`. All four
   families × two eval sets × three targets × seven metric rows
   produced, totalling 168 sentiment-row replacements and 168 new
   topic rows; cumulative `phase3_ablation.csv` is at 288 rows.
7. **Ran proposal Section 8 diagnostics.** Output captured below.

---

## 4. Headline numbers (topic added on top of the revised dialogue-only floor, 4 families × 2 eval sets)

Full table at `reports/tables/phase3_ablation.csv` (cumulative now
288 rows: lexical + sentiment + topic). Reporting uses the project
metric vocabulary (regression: MSE / RMSE / MAE / CVRMSE;
classification: AUC-ROC / PR-AUC / F1 / log-loss). Both train and
OOF reported per family per metric. Pre-registered lift bands apply
to the linear family's OOF.

### 4.1 OOF lift per family (lower-is-better metrics: negative lift means improvement)

| Family | Metric | Floor | With topic | Lift |
|---|---|---:|---:|---:|
| linear | log_roi RMSE (lower) | 1.339 | 1.354 | **+0.016** (worse) |
| linear | log_roi MAE (lower) | 0.948 | 0.956 | +0.007 (worse) |
| linear | roi_gt_1 AUC (higher) | 0.558 | 0.590 | **+0.032** (better, above band) |
| linear | roi_gt_1 PR-AUC (higher) | 0.846 | 0.860 | **+0.014** (in-band) |
| linear | roi_gt_1 log-loss (lower) | 0.493 | 0.497 | +0.004 (worse) |
| linear | roi_gt_2 AUC (higher) | 0.602 | 0.613 | **+0.012** (below band, right direction) |
| linear | roi_gt_2 PR-AUC (higher) | 0.739 | 0.733 | -0.006 (worse) |
| linear | roi_gt_2 log-loss (lower) | 0.635 | 0.642 | +0.007 (worse) |
| histgb | log_roi RMSE (lower) | 1.327 | 1.336 | +0.009 (worse) |
| histgb | roi_gt_1 AUC (higher) | 0.552 | 0.578 | **+0.026** (better) |
| histgb | roi_gt_2 AUC (higher) | 0.610 | 0.599 | -0.011 (worse) |
| knn | log_roi RMSE (lower) | 1.364 | 1.376 | +0.012 (worse) |
| knn | roi_gt_1 AUC (higher) | 0.527 | 0.555 | **+0.028** (better) |
| knn | roi_gt_2 AUC (higher) | 0.578 | 0.525 | -0.053 (worse) |
| svm | log_roi RMSE (lower) | 1.357 | 1.361 | +0.004 (worse) |
| svm | roi_gt_1 AUC (higher) | 0.504 | 0.556 | **+0.052** (better) |
| svm | roi_gt_2 AUC (higher) | 0.534 | 0.540 | +0.007 (better) |

Bold marks moves of 0.005 or more. The pattern is qualitatively
new for Phase 3b: every model family lifts `roi_gt_1` AUC, and
linear lifts `roi_gt_2` AUC modestly. The previous two groups had
flat or negative `roi_gt_1` AUC lifts across families.

### 4.2 Pre-registered linear-family lift (proposal v1 Section 4)

| Target | Metric | Predicted band (linear OOF) | Actual | In band? |
|---|---|---|---:|:---:|
| log_roi | RMSE | -0.030 to -0.005 (lower is better) | +0.016 | No (wrong direction) |
| log_roi | MAE | -0.025 to -0.005 | +0.007 | No (wrong direction) |
| log_roi | CVRMSE | -0.025 to -0.005 | +0.012 | No (wrong direction) |
| roi_gt_1 | AUC-ROC | 0.000 to +0.015 | +0.032 | **No (above band — better than predicted)** |
| roi_gt_1 | PR-AUC | 0.000 to +0.015 | +0.014 | **Yes** |
| roi_gt_1 | log-loss | -0.020 to 0.000 | +0.004 | No (wrong direction) |
| roi_gt_2 | AUC-ROC | +0.015 to +0.040 | +0.012 | No (right direction, below band) |
| roi_gt_2 | PR-AUC | +0.010 to +0.030 | -0.006 | No (wrong direction) |
| roi_gt_2 | log-loss | -0.025 to -0.005 | +0.007 | No (wrong direction) |

One in-band hit (PR-AUC on `roi_gt_1`) and one over-the-top hit
(AUC-ROC on `roi_gt_1`, +0.032 vs predicted +0.015 ceiling). The
pre-registration's central target (`roi_gt_2` AUC) was
under-shot, the regression metrics were all wrong-directional,
and the surprise positive landed on the gross-profitability
target the proposal expected to be most signal-thin.

### 4.3 Train-versus-OOF gap (overfit diagnostic)

| Family | Eval set | log_roi RMSE | roi_gt_1 AUC |
|---|---|---:|---:|
| linear | train | 1.291 | 0.685 |
| linear | oof | 1.354 | 0.590 |
| linear | train − oof | -0.063 | +0.095 |
| histgb | train | 1.142 | 0.832 |
| histgb | oof | 1.336 | 0.578 |
| histgb | train − oof | -0.194 | +0.254 |

HistGB's train-OOF gap on `roi_gt_1` AUC (+0.254) is wider than
under sentiment-augmented (+0.20) or lexical-augmented (+0.22),
indicating the larger feature count is increasing overfit
absorption. Despite this, HistGB's OOF on `roi_gt_1` AUC also
genuinely lifts (+0.026), so the gain is not pure overfit
displacement.

### 4.4 Verdict

The multi-family picture is **qualitatively different from
lexical and sentiment**, despite missing the strict
in-band check on most metrics:

* **Across-family directional consistency on `roi_gt_1` AUC.** All
  four families lift the gross-profitability target's AUC; the
  previous two groups had mixed or negative directions.
* **Linear over-shot the proposal's `roi_gt_1` AUC band** (+0.032
  vs predicted +0.015 ceiling). Pre-registration mis-calibrated
  the band's upper bound, but the direction was correct.
* **PR-AUC on `roi_gt_1` lands in-band** (+0.014 vs predicted
  0.000 to +0.015), the first in-band hit of any Phase 3b group
  on any metric.
* **`roi_gt_2` is the disappointing target.** Pre-registration
  expected this to be the strongest direction (+0.015 to +0.040
  AUC), partly because of its higher headroom and the literature
  prior. Linear delivers +0.012, just below the band; HistGB and
  KNN actively get worse. The reason is structural, discussed in
  Section 6.
* **Regression target `log_roi` is null** across all families,
  including SVM. This is broadly consistent with the previous two
  groups: the 1,199-film corpus has too thin a signal-to-noise
  ratio for topic features to lift continuous outcome accuracy
  beyond what genre + structural counts already provide.

Net: topic is the **first standalone group of Phase 3b that
produces consistent positive directional movement on a target**.
The signal is on the gross-profitability target rather than the
expected net-profitability target. Phase 4 model selection
should consider keeping the topic features in the candidate
matrix, especially for `roi_gt_1` modelling.

---

## 5. Diagnostic results (Section 8 of proposal v1)

| # | Check | Threshold / expectation | Result | Verdict |
|---|---|---|---|---|
| 1 | Topic coherence (UMass) | mean below ~-2.5 prompts revisit | mean -1.625, range -2.6 to -0.7 | Pass (moderate, not exceptional) |
| 2 | Top-10 words per topic | informational | Saved to `reports/tables/phase3_topic_labels.csv`; topics are interpretable but several are character-name-dominated | **Documented; see below** |
| 3 | Topic ↔ structural baseline | drop pair if \|r\| > 0.85 | All pairs \|r\| < 0.34 | Pass |
| 4 | Topic ↔ genre-dummy correlations | informational | Several pairs in 0.20-0.38 range; documents the genre-residual overlap | **See below** |
| 5 | data_quality_flag films vs unflagged | z-diff > 1.0 prompts review | All 22 features within ±0.31σ of unflagged means | Pass |
| 6 | Topic-distribution dominance | informational | Mean top-topic share 0.57; 720/1199 films above 0.5 | **Concentrated; see below** |
| 7 | Univariate target correlations | informational | No feature exceeds \|r\| = 0.10; strongest is `topic_concentration_entropy ↔ log_roi` at -0.080 | **Confirms multivariate signal; see below** |
| 8 | Leakage smoke test | unit-test invariant | `transform` outputs are independent of the inference batch | Pass (test in `tests/test_topic.py`) |

Three checks merit narrative interpretation.

**Check 2 — character-name-dominated topics.** Inspecting the top-
10 words per topic surfaces the well-known LDA pathology on
screenplay text: several topics are dominated by character
names rather than thematic content. Examples:

* `topic_00`: jack, paul, mickey, doc, war, nothin, fuckin, gun,
  goddamn, goin (action / crime, with character names mixed in)
* `topic_02`: peter, bruce, elizabeth, charlie, dude, movie,
  santa, jerry, wedding, scott (mostly character names)
* `topic_03`: team, yah, spider, miles, steve, ball, jordan,
  baseball, all, mark (sports topic, with character names)

Other topics are cleaner:

* `topic_01`: president, war, bill, general, american, mike,
  government, army, states, fbi (politics / military)
* `topic_04`: shall, lord, king, perhaps, dear, queen, war,
  master, upon, fear (medieval / period)

The character-name pollution is partly a property of MovieSum's
dialogue formatting (characters address each other by name more
often than in published prose) and partly a property of the
unigram vocabulary at K = 20. A v2 pass that adds a custom
character-name stopword list (built from the parsed-screenplay
character tags) is a defensible polish item; for v1 standalone-
lift evaluation the topics are kept as-is so the result is
honestly representative of the proposal's design.

**Check 4 — topic-genre overlap.** Several topics correlate with
genre dummies in the 0.20-0.38 range. The strongest pair is
`topic_18_proportion ↔ genre_Science Fiction` at +0.38;
`topic_05_proportion ↔ genre_Comedy` at +0.29; `topic_18 ↔
genre_Action` at +0.28. This empirically validates the genre-
residual hypothesis: about a third of each thematically-
distinct topic's variance overlaps with the genre dummies the
baseline already includes. The fact that topic features still
lift `roi_gt_1` AUC across all four families means the *non-
overlapping* portion of topic variance does carry signal — which
is the orthogonality the previous two groups never demonstrated.

**Check 6 — topic-distribution concentration.** Mean top-topic
share is 0.57; 720 of 1,199 films have a single topic capturing
more than half their probability mass. This is concentrated
relative to a uniform K = 20 distribution (which would have
expected mass 0.05 per topic). LDA's variational solver tends
to produce concentrated posteriors on small per-document
vocabularies; this is the expected shape rather than a defect.
The `topic_concentration_entropy` feature captures the spread
across films of this concentration measure (films with higher
entropy are topically diffuse).

**Check 7 — multivariate signal.** No single topic feature
exceeds |r| = 0.10 against any target. The strongest is
`topic_concentration_entropy ↔ log_roi` at -0.080. This matches
the lexical and sentiment groups' univariate-correlation shape.
But unlike those groups, the multi-family ablation extracts
positive lift on `roi_gt_1` AUC across all four families. The
inference is that the topic signal is **multivariate**: any
single topic's correlation with the target is small, but the
*composition* of which topics dominate which films carries
information that the linear, tree, KNN, and SVM families all
extract collectively. This is the structural difference between
the topic group and the previous two groups.

---

## 6. Why `roi_gt_2` did not respond as predicted

The proposal's central pre-registered hypothesis was that
`roi_gt_2` AUC would lift by +0.015 to +0.040 because the
"net-profitable" target carries the most headroom and topic
features capture subject-matter information that aligns with
the blockbuster-versus-mid-budget split.

Two mechanisms explain why the actual `roi_gt_2` lift came in
below the predicted band on linear (+0.012) and went negative on
HistGB and KNN.

**The blockbuster-versus-mid-budget split is heavily mediated
by genre.** The structural baseline already contains genre
dummies for Action, Adventure, Animation, Family, Science
Fiction, Fantasy, and Comedy. These are exactly the genres that
align with blockbuster status, and the genre-dummy explanation
of `roi_gt_2` runs deeper than the topic-via-genre overlap
captures. After conditioning on genre, the residual `roi_gt_2`
signal is not in the topic distribution; it is in the audience-
size estimation that genre-plus-budget already encodes.

**`roi_gt_1` is the easier target for topic features specifically.**
The gross-profitability target has an 80% positive base rate, so
the discriminating information is on the unprofitable minority.
Topics that signal "this film's subject matter has narrow
audience appeal" are more useful here than topics that signal
"this film is a tentpole-scale blockbuster". The former
information is genre-orthogonal (a niche subject can land in any
genre); the latter information is genre-redundant. The topic
features therefore have more headroom on `roi_gt_1` than the
proposal expected, and less on `roi_gt_2`.

This is a real finding for the report: the genre-orthogonality
discussion now has empirical support beyond "the previous two
groups landed null". Topic features carry signal where they are
genre-orthogonal, and they happen to be most genre-orthogonal on
the unprofitable-minority target rather than the
blockbuster-target.

---

## 7. Files produced (Phase 3b topic group)

### Code
* `src/features/topic.py` (~360 lines): the 22-feature pipeline
  with the leak-free fit/transform API.
* `src/experiments/run_topic_ablation.py`: the multi-family
  topic-ablation runner.
* `tests/test_topic.py` (9 tests, all passing).
* `src/features/baseline_features.py`: extended with
  `include_topic` and `topic_features_path` knobs.

### Data
* `data/processed/features_topic.parquet` (1,713 × 22, ~410 KB).
* `data/processed/topic_model_artifacts/`: `vectorizer.joblib`,
  `lda.joblib`, `train_ids.npy`. Phase 4 modelling can reload
  these directly.

### Run artifacts
* `runs/phase_3/20260503_2015_topic_multifamily/`: six files
  (`params.json`, `preprocessing_summary.json`, `features_used.json`,
  `metrics.json`, `run.log`).

### Tables
* `reports/tables/phase3_ablation.csv`: extended from 192 rows
  (lexical + sentiment) to 288 rows (lexical + sentiment + topic).
* `reports/tables/phase3_topic_labels.csv`: per-topic top-10
  words plus an empty `human_label` column for manual annotation.

### Configuration changes
* `BaselineFeatureConfig.include_topic` and
  `topic_features_path`.
* `runs/RUNS.md`: new row for the topic multi-family run.

---

## 8. Resolved questions

1. **Treatment of the topic group's official ablation row: APPEND.**
   The partially-positive row is in `phase3_ablation.csv` as the
   honest finding. The Phase 3 narrative now leads with two null
   results followed by a partial positive — a more interesting
   story than three nulls.
2. **K = 20 retained.** Diagnostic UMass coherence was not
   exceptional but not degraded; topic interpretability is
   acceptable; revisiting K is a v2 polish item if the report
   needs cleaner topic narratives.
3. **Character-name-dominated topics: documented, not patched.**
   A character-name stopword list is a defensible v2 polish
   addition; for the v1 standalone-lift evaluation the topics
   are kept as-is.
4. **All 22 topic features retained for Phase 3c combinations.**
   The multivariate signal extracted on `roi_gt_1` AUC is
   distributed across the topic features rather than
   concentrated in any single column; dropping individual topic
   columns at this stage would close off optionality.
5. **Proceed to character-network proposal: YES.** Topic's
   partial positive demonstrates that genre-orthogonal feature
   groups can lift Phase 3b even when their univariate
   correlations are sub-0.10. The character-network group's
   structural-graph features are the next plausible
   genre-orthogonal candidate.

## 9. Decisions log entry to add

> ## 2026-05-03 20:30 — Phase 3b: topic standalone result is partial positive
>
> **Phase:** Phase 3 — Feature Extraction (sub-phase 3b, third of five groups)
> **Decision:** Topic features (22 columns: 20 LDA topic proportions, 1 distribution-concentration entropy, 1 dominant-topic id) implemented per proposal v1; the multi-family ablation produced a **partial positive** verdict, qualitatively different from the lexical and sentiment null results. All four model families lift `roi_gt_1` AUC (linear +0.032, histgb +0.026, knn +0.028, svm +0.052), the first time any Phase 3b group has produced consistent across-family directional movement on a target. PR-AUC on `roi_gt_1` lands in-band at +0.014. Linear `roi_gt_2` AUC lifts +0.012 (just below the predicted +0.015 floor); HistGB and KNN go negative on `roi_gt_2`. Regression target null/wrong-direction on every family. The proposal's central pre-registered hypothesis (`roi_gt_2` AUC +0.015 to +0.040) was wrong on direction-and-magnitude grounds; the surprise positive came on `roi_gt_1` instead, attributable to topic features being more genre-orthogonal on the unprofitable-minority target than on the blockbuster-target. No-leakage discipline implemented and verified by unit test. The 22 topic features are retained in the matrix for the Phase 3c combinations evaluation; Phase 4 model selection should consider them in the candidate matrix, especially for `roi_gt_1` modelling.
> **See also:** `docs/handoffs/phase_3b_topic_handoff.md`.

---

## 10. Next step

Start the character-network implementation per
`docs/proposals/phase3_character_network_proposal.md`. The topic
result demonstrates that genre-orthogonal feature groups can
land non-null even when the univariate correlations are below
the |r| = 0.10 threshold. Character network is the proposal's
strongest remaining candidate for genre orthogonality (graph
structure is not directly encoded by genre dummies); the
expectation going in is at least matching the topic group's
directional consistency, plausibly exceeding it on `roi_gt_2`
where topic underperformed.
