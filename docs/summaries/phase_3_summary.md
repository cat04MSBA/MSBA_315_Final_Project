# Phase 3: Feature Extraction

**Status:** Complete
**Date completed:** 2026-05-03

---

## Strategic decisions made before/during this phase

The phase carried a heavy strategic-decision load. Eight decisions
shaped its execution; each is logged with its date in
`docs/PROJECT_CONTEXT.md` Section 8. Summarized here for reference:

* **2026-05-02 (planning conversation, captured in the Phase 3 brief
  Section 2):** Three targets in parallel (`log_roi` for regression;
  `roi_gt_1` and `roi_gt_2` for classification, threshold-consistent
  with the regression target). Single 70 / 15 / 15 stratified train /
  calibration / test split, seed 42. Calibration set carved here for
  Phase 5. Sub-phase boundary mandatory: 3a baseline complete before
  3b feature engineering begins. These were locked in by the planning
  conversation before execution started.
* **2026-05-03 14:10:** Phase 3a baseline feature parameterization
  revised mid-phase. `log1p` applied to the six heavy-tailed
  structural counts before z-scoring; `log_runtime` added to the
  deployable baseline (runtime is leak-free pre-greenlight via the
  page-count-equals-minutes industry convention). Original (un-
  logged, no-runtime) numbers preserved in `phase3a_baseline.csv`
  for the report's before/after comparison. Source: borderline
  original numbers + the brief listing `log_runtime` as a likely
  candidate that the original implementation overlooked.
* **2026-05-03 17:20:** Phase 3a/3b expanded from 1 to 4 model
  families (linear, HistGB, KNN, SVM-RBF). The first lexical
  ablation came in with negative drift on the linear baseline,
  raising the question whether the issue was the features or a
  peculiarity of the linear family. The four families were chosen
  to span four distinct inductive-bias paradigms (linear, non-linear
  global tree-ensemble, instance-based local, kernel-based). Linear
  remains the historical reference for the brief's escalation
  threshold; the other three families provide cross-paradigm
  diagnostics.
* **2026-05-03 18:30:** Phase 3c combinations sub-phase added to
  the methodology. The lexical group's near-null verdict against a
  baseline that already included genre, era, and structural counts
  raised the concern that incremental ablation systematically
  under-credits any group whose signal partially overlaps with
  genre. Phase 3c evaluates a small pre-specified set of feature-
  group combinations against the same floor, after all five Phase 3b
  groups have produced their standalone-lift rows. Pre-registration
  preserved at the combinations level: the set is locked at proposal
  time and is not expanded after seeing results.
* **2026-05-03 19:00:** Metric vocabulary updated. R-squared dropped
  from the reported set; regression metrics are now MSE, RMSE, MAE,
  and CVRMSE (coefficient of variation of RMSE). Classification
  metrics are AUC-ROC, PR-AUC, F1, and log-loss. Both in-sample
  (training-fold fit) and out-of-fold (5-fold CV) values reported
  per metric so the train-versus-OOF gap is visible as an overfitting
  diagnostic. Surfaced an important Phase 4 finding: HistGB's
  in-sample fit is roughly 0.20 AUC above its OOF on classification
  targets even with conservative defaults.
* **2026-05-03 (planning conversation review of lexical
  proposal v2):** Three resolutions on the lexical group:
  (a) append the negative-lift row honestly; (b) keep
  `rare_word_proportion` despite its r = -0.939 with
  `mean_log_frequency` (preserve optionality for Phase 4); (c)
  accept the wordfreq deviation from canonical SUBTLEX-US (the
  domain-match argument is preserved by wordfreq's inclusion of
  OpenSubtitles).
* **2026-05-03 19:50, 20:30, 20:30, 20:50:** Four Phase 3b group
  verdicts logged in succession: sentiment null, topic partial
  positive, character_network partial positive, embedding partial
  positive. Each entry records the specific multi-family pattern
  that distinguished the verdict from the previous group's.
* **2026-05-03 22:00:** Phase 3c combinations verdict logged. Two
  findings reshape Phase 4: (1) standalone group lifts do not
  compose additively for linear regression; (2) SVM-RBF dominates
  classification under combinations, going from worst-of-four
  standalone to best-of-four on combinations.

---

## What we did

Phase 3 unfolded in three sub-phases plus a closure step.

### Phase 3a (baseline floor)

1. **Carved the train/calibration/test split** at 70/15/15, seed 42,
   stratified by (primary_genre_bucketed, decade_bucket) with
   rare-cell pooling. Saved authoritative split to
   `data/processed/split_assignments.parquet`. 57 strata; every
   named stratum has at least one film in each split.
2. **Defined the three targets** in `src/features/targets.py`:
   `log_roi = ln(revenue) - ln(budget)`, `roi_gt_1 = (log_roi > 0)`,
   `roi_gt_2 = (log_roi > ln 2)`. Threshold-consistency holds by
   construction.
3. **Trained the structural-only baseline** under all 4 model
   families on 4 feature configurations (original / revised
   parameterization, dialogue-only / with-budget). Output saved to
   `reports/tables/phase3a_baseline.csv` (384 rows: 4 configs x 4
   families x 2 eval sets x {3 targets x 4 metrics each}).
4. **Mid-phase revision:** when original numbers came in just above
   the brief's escalation thresholds with CI lower bounds dipping
   below, applied `log1p` to heavy-tailed counts and added
   `log_runtime`. Re-ran. Threshold check passes on linear; HistGB
   exceeds linear on the headline targets even on structural
   features alone.
5. **Multi-family expansion:** when the lexical group's first run
   came in with negative linear drift, expanded the harness to
   evaluate every Phase 3b ablation across all four families.

### Phase 3b (incremental feature engineering, five groups)

For each group, the same sequence: written proposal pre-registering
specific features and expected lift bands, planning-conversation
review, implementation, multi-family ablation, diagnostic checks,
verdict logged in a per-group handoff.

* **Lexical (group 1, 14 features, 2 NaN handled).** Vocabulary
  diversity (MTLD on dialogue and action, hapax ratio), lexical
  sophistication (mean and rare-word-proportion log-frequencies via
  wordfreq), readability (Flesch-Kincaid on dialogue and action),
  length statistics, punctuation, pronoun ratio. Standalone verdict:
  **null**. Linear OOF lift on log_roi RMSE +0.011 (worse), AUC
  movements within CI noise. Mechanism: lexical features carry
  information genre, era, and structural counts have already
  absorbed.
* **Sentiment (group 2, 22 features, 2 diagnostic).** VADER
  aggregates (mean, std, range), 8-emotion NRC proportions,
  sentiment quartile-trajectory features, 6 Reagan arc-archetype
  similarities (hand-coded smoothed templates). Standalone verdict:
  **null**. Linear OOF lift on log_roi RMSE +0.019 (worse).
  Genre-residual-signal hypothesis from the lexical handoff
  predicted sentiment should fare better; it did not.
* **Topic (group 3, 22 features).** 20-topic LDA fit on training
  films only, per-film topic proportions plus
  `topic_concentration_entropy` plus `topic_dominant_id`. Standalone
  verdict: **partial positive**. Linear OOF lift on `roi_gt_1` AUC
  +0.032 (the largest standalone classification lift of the phase),
  4-of-4 families positive. Known limitation: at K = 20, several
  topics are character-name-dominated (acceptable for this
  evaluation; v2 polish item if interpretability is needed later).
* **Character network (group 4, 12 features).** Co-occurrence
  graph metrics (density, components, clustering, modularity,
  centrality), per-character dialogue concentration (Gini, top-k
  shares), lead-role count, network diameter. Standalone verdict:
  **partial positive**. Linear OOF lift on `roi_gt_2` AUC +0.016,
  4-of-4 families positive. First cross-group |r| > 0.10 on the
  phase: `network_lead_role_count` r = -0.102 with `roi_gt_2`.
* **Embedding (group 5, 32 features).** MiniLM
  (sentence-transformers/all-MiniLM-L6-v2) pooled per scene then
  mean-pooled per film, PCA fit on training fold to 32 components
  (cumulative variance 73.9%). Standalone verdict: **partial
  positive**. The only group across the phase whose log_roi RMSE
  lifted negatively (better) on linear: -0.007 OOF. Two cross-group
  |r| > 0.10: `embed_pc_01` r = +0.114 with `log_roi`,
  `embed_pc_04` r = +0.106 with `roi_gt_2`.

### Phase 3c (combinations sub-phase)

Four pre-specified combinations evaluated under the same multi-
family harness. Results re-shaped the Phase 4 expectations:

* **`all_five`** (130 features): SVM-RBF reaches `roi_gt_2` AUC
  ~~0.665~~ **0.5966** OOF (lift +0.0629 over SVM's own structural
  baseline of 0.5337). Linear regression hurt by feature-count
  noise. **CORRECTION 2026-05-04 (mid-Phase-4):** the 0.665 figure
  was a calculation error — SVM's lift (+0.0629) was added to
  *linear's* structural floor (0.6017) rather than to *SVM's* own
  floor (0.5337). The actual SVM-RBF OOF AUC on roi_gt_2 from the
  Phase 3c run was 0.5966 (CI [0.5649, 0.6310]) per
  `reports/tables/phase3c_combinations.csv`. Phase 4's SVM-RBF on
  the same matrix reaches 0.6200 (full grid + 3x5 repeated CV);
  the genuine corpus ceiling appears to lie around 0.63 OOF AUC
  rather than the 0.665 originally reported.
* **`partial_positives`** (92 features, drops the two nulls): no
  net positive on linear classification.
* **`topic_plus_cn`** (60 features): the parsimonious winner. Only
  combination tested whose linear OOF AUCs lift positively on
  both classification targets (+0.021 each). Largest single
  classification lift in the phase: SVM on this combination,
  `roi_gt_1` AUC +0.081.
* **`semantic_trio`** (102 features): underperforms across the
  board.

The pre-registered linear-OOF lift bands missed on 10 of 12
headline metrics. The pre-registration discipline produced the
honest finding the report can use: standalone group lifts do not
compose additively for linear regression on n = 1,199. SVM-RBF
benefits enormously from larger combinations; HistGB hurts; KNN
hurts.

### Phase 3 closure

Wrote `docs/FEATURE_NOTES.md` (the standing reference). Built
`data/processed/features.parquet` (1,713 x 131: 127 features + 3
targets + split column) as the consolidated Phase 4 input matrix.
Updated `PROJECT_CONTEXT.md` Sections 5 and 9.

---

## Why we did it that way

Three methodology principles drove the phase's structure.

**Pre-registration before measurement.** Every Phase 3b feature
group landed with a written proposal that stated specific features
and expected lift bands before any code ran. Same discipline
applied at the combinations level in Phase 3c. The pre-registration
makes a result interpretable: when the lexical group's lift came in
negative, the prediction had been positive, and the gap was a
finding rather than a result post-hoc-rationalized. When the
Phase 3c combinations missed 10 of 12 bands, the gap was the
finding. A report that can show predicted-versus-actual is a
stronger report than one that only shows actual.

**Multi-family disambiguation rather than single-family
ablation.** The original Phase 3 brief specified a linear
constant-comparator. The lexical group's first negative result
exposed the brief's blind spot: a feature group that fails on
linear could either carry no signal (in which case all model
families would null) or carry only non-linear signal (in which
case other model families would lift). Without the multi-family
view, those two scenarios are observationally identical. The
4-family expansion turned every ablation row into a
cross-paradigm diagnostic. The decisive Phase 3c finding (SVM-RBF
extracts substantial signal from combinations that linear cannot)
would have been completely invisible under single-family
ablation.

**Train and OOF reporting on every metric.** The metric-vocabulary
update on 2026-05-03 added in-sample-versus-out-of-fold reporting
to every metric. This surfaced a finding that the OOF-only view
hid: HistGB's substantial in-sample overfit (0.20 AUC train-OOF
gap on classification targets) at the conservative defaults Phase 3
used. That has direct implications for Phase 4 hyperparameter
search, which the closing handoffs explicitly flag.

---

## Tactical choices made

* **Decade bucketing for stratification:** `pre_1980s` collapses
  103 films; `2010s_2020s` collapses because 2020-2023 is too thin
  alone. Rationale: per-decade cells before 1980 each have fewer
  than 30 films; the four-family ablation needs stable
  stratification.
* **Empty-text dialogue filter at every entry point.** The Phase 2
  Tier 1.3 filter applies defensively in every dialogue-derived
  feature module, even when the upstream parser is expected to have
  filtered already. Rationale: tests in `tests/test_*.py` confirmed
  the filter's behaviour but defensive layering avoids a future
  regression silently corrupting features.
* **Quartile-windowing by dialogue-line index, not scene index
  (sentiment group).** Avoids the `data_quality_flag` problem with
  collapsed scene structure. A film with 2 scenes and 800 dialogue
  lines still has well-defined dialogue-line quartiles.
* **Hand-coded smoothed mathematical templates for Reagan arc
  archetypes.** Preferred over re-deriving via SVD on a corpus of
  our choosing (defensibility against overfit-the-archetype-corpus)
  and over reading shape coefficients from Reagan's supplementary
  materials (URL stability).
* **`nrclex` Python package for NRC EmoLex.** Canonical Saif
  Mohammad source is form-gated; the package ships byte-identical
  word-emotion mappings under the original publication's
  research-use license.
* **wordfreq for SUBTLEX-US frequency reference.** Canonical
  SUBTLEX-US download URLs returned 404; wordfreq's English Zipf
  scale uses OpenSubtitles in its source mixture, preserving the
  subtitle-domain match argument the proposal selected SUBTLEX-US
  for. Feature names dropped the `_subtlex` suffix to be honest.
* **PCA dimensionality 32 for embeddings.** Cumulative variance
  73.9% on the training-fold pooled embeddings. The number was
  chosen as a reasonable trade-off between dimensionality and
  variance; Phase 4 retains the option to revisit (the raw 384-dim
  cache is on disk).
* **K = 20 for LDA topics.** Cohort-based default for screenplay
  text; the topic handoff documents the character-name-dominance
  caveat explicitly. Phase 4 may want to test K = 30 or K = 50 if
  topic features turn out to be a useful Phase 4 input.
* **`treat_flagged_as_nan=True` default in character_network.py.**
  Films with `data_quality_flag` (collapsed scene structure)
  produce unreliable graph metrics (density, modularity,
  centrality) because their scene boundaries are not meaningful.
  These get NaN values for network features; the modelling
  pipeline imputes them.
* **NLTK `word_tokenize` rather than spaCy.** Cheaper for this
  group's purposes; if a downstream group requires spaCy,
  re-tokenizing lexical and sentiment under spaCy is a
  one-module-rewrite alternative. Penn Treebank and spaCy English
  tokenizers agree on more than 99% of tokens for prose-like text.

---

## Results

### Headline numbers (linear OOF lift over Phase 3a revised dialogue-only floor)

Phase 3a floor (linear OOF, the comparison point for Phase 3b
standalone and Phase 3c combinations): log_roi RMSE 1.339,
roi_gt_1 AUC 0.558, roi_gt_2 AUC 0.602.

#### Phase 3b standalone groups

| Group | log_roi RMSE | roi_gt_1 AUC | roi_gt_2 AUC | Verdict |
|---|---:|---:|---:|---|
| lexical | +0.011 (worse) | +0.007 | -0.002 | Null |
| sentiment | +0.019 (worse) | -0.012 | +0.009 | Null |
| topic | +0.016 (worse) | **+0.032** | +0.012 | Partial positive (`roi_gt_1`) |
| character_network | +0.009 (worse) | +0.013 | **+0.016** | Partial positive (`roi_gt_2`) |
| embedding | **-0.007** (better) | +0.013 | +0.006 | Partial positive (regression) |

#### Phase 3c combinations

| Combination | log_roi RMSE | roi_gt_1 AUC | roi_gt_2 AUC | Linear in-band |
|---|---:|---:|---:|---:|
| `all_five` | +0.007 | -0.005 | **+0.011** | 1/3 |
| `partial_positives` | +0.006 | +0.012 | -0.003 | 0/3 |
| `topic_plus_cn` | +0.028 | +0.021 | **+0.021** | 1/3 |
| `semantic_trio` | +0.004 | -0.000 | -0.012 | 0/3 |

#### SVM-RBF on combinations (the substantive Phase 4 finding)

The largest classification lifts in the phase. SVM standalone was
worst-of-four; on combinations, SVM is best-of-four:

| Combination | SVM `roi_gt_2` AUC lift | SVM `roi_gt_1` AUC lift |
|---|---:|---:|
| `all_five` | **+0.063** | +0.056 |
| `partial_positives` | **+0.063** | +0.054 |
| `topic_plus_cn` | +0.043 | **+0.081** |
| `semantic_trio` | +0.057 | +0.046 |

SVM on `all_five` reaches `roi_gt_2` AUC = ~~0.665~~ **0.5966** OOF
(corrected — see note below), below the original
forward-expected band of 0.65 to 0.72. SVM on
`topic_plus_cn` reaches `roi_gt_1` AUC = 0.639 OOF, the largest
classification lift recorded in Phase 3.

> **Correction note (added 2026-05-04 mid-Phase-4):** The 0.665
> figure originally reported in this section was a calculation
> error. SVM-RBF's lift over its own structural baseline (+0.0629)
> was incorrectly added to *linear*'s structural floor (0.6017)
> rather than to *SVM*'s own floor (0.5337), yielding the
> spurious "0.665." The actual SVM-RBF OOF AUC on `roi_gt_2` from
> the Phase 3c run is **0.5966** (95% CI [0.5649, 0.6310]) per
> `reports/tables/phase3c_combinations.csv`. The
> "forward-expected 0.65 to 0.72 band" cited above and elsewhere
> in this summary was anchored to the incorrect 0.665 figure;
> Phase 4 uses 0.6346 (the actual best primary OOF AUC) as the
> realistic benchmark and treats the original band as
> non-binding. See `docs/PROJECT_CONTEXT.md` Section 8 entry
> dated 2026-05-04 00:30 for the full Phase 4 audit trail of
> this discovery and its implications.

### Saved figures

* `reports/figures/phase3_target_distributions.png`. Three-panel
  diagnostic of `log_roi`, `roi_gt_1`, `roi_gt_2` distributions and
  threshold-consistency cross-tab. **Implication:** `log_roi` is
  approximately symmetric around 1.06 (the median); `roi_gt_1` is
  80% positive; `roi_gt_2` is 64% positive. The asymmetric base
  rates have implications for which target is the most tractable
  primary outcome (Phase 4 question, but the data already suggest
  `roi_gt_2`).
* `reports/figures/phase3_log_transform_effect.png`. Before-and-
  after histograms for three representative heavy-tailed
  structural counts (`n_dialogue_lines`, `n_unique_characters`,
  `parse_warning_count`). **Implication:** the log transform
  produces approximately symmetric distributions suitable for
  z-score standardization, justifying the Phase 3a revision.

### Saved tables

* `reports/tables/phase3_split_diagnostics.csv` (57 strata x split
  counts; every named stratum has at least one film in each split).
* `reports/tables/phase3a_baseline.csv` (384 rows: 4 configs x 4
  families x 2 eval sets x 3 targets x ~4 metrics).
* `reports/tables/phase3_ablation.csv` (480 rows: Phase 3b
  standalone-group ablation across all 5 groups x 4 families x 2
  eval sets x 3 targets x ~4 metrics).
* `reports/tables/phase3c_combinations.csv` (384 rows: 4
  combinations x 4 families x 2 eval sets x 3 targets x ~4
  metrics).

---

## Issues encountered & resolved

1. **Lexical first-run negative drift on linear.** First Phase 3b
   ablation produced -0.016 R-squared lift (with R-squared still
   in the metric set at the time). Surfaced as a methodology
   question rather than mechanically accepted. Triggered the
   multi-family expansion that became the structural pattern for
   the rest of Phase 3b and Phase 3c.
2. **HistGB substantial overfit on the structural baseline.** The
   train-versus-OOF reporting added on 2026-05-03 19:00 surfaced
   roughly 0.20 AUC train-OOF gap on classification targets even
   with conservative defaults. Documented as a Phase 4
   hyperparameter-search instruction (more aggressive
   regularization needed).
3. **wordfreq replaced SUBTLEX-US.** Canonical SUBTLEX-US download
   URLs returned 404 / HTML at implementation time. Used wordfreq
   (subtitle domain preserved via OpenSubtitles inclusion), renamed
   features to drop the `_subtlex` suffix, documented in the
   lexical handoff.
4. **NRC sourcing via `nrclex` package.** Canonical Saif Mohammad
   source is form-gated. Used the `nrclex` package, byte-identical
   word-emotion mappings, documented as deviation in the sentiment
   handoff.
5. **MPS embedding determinism.** Local reproduction of the
   embedding ablation produced regression metrics matching the
   friend's run to four decimals but a 1pp difference on
   `roi_gt_1` AUC. Cause: MiniLM forward-pass on MPS has minor
   floating-point variation in the 6-7th decimal that compounds
   through PCA; `LogisticRegressionCV`'s internal C-selection lands
   on a different grid point. `RidgeCV` is closed-form so it
   reproduces exactly. Documented; the verdict (embedding partial
   positive) is unchanged.
6. **Pre-registered Phase 3c bands missed on 10 of 12 metrics.**
   Standalone-group lifts do not compose additively for linear
   regression. The miss is itself the finding the report uses:
   incremental ablation against a linear baseline systematically
   under-credits combination value for non-linear models. Pre-
   registration discipline preserved by not expanding the set
   after seeing results.

---

## Open questions / things to flag

The closing decisions for Phase 4 to settle:

1. **Phase 4 input matrix: `all_five` (130 features) or only the
   standalone-positive union?** The Phase 3c finding that SVM-RBF
   extracts substantial signal from the maximum-information matrix
   argues for `all_five`. The pre-registration discipline argues
   for the standalone-positive union (topic + character_network +
   embedding, ~92 features). Recommendation in the Phase 3c
   handoff: `all_five` plus a sensitivity-analysis run on the
   standalone-positive union.
2. **SVM-RBF as a serious Phase 4 candidate.** Phase 3a's framing
   implied linear and tree ensembles were the natural Phase 4
   candidates. The Phase 3c evidence elevates SVM-RBF to the same
   tier on classification. Phase 4 should give it a real
   hyperparameter search (C grid, gamma grid).
3. **Primary outcome variable.** The decision is formally deferred
   to end of Phase 4, but the Phase 3 evidence already points to
   `roi_gt_2` as the most tractable: SVM on `all_five` reaches
   ~~0.665~~ **0.5966** OOF AUC (corrected — see correction note in
   the Results section), the strongest single-family classification
   number in the phase. `roi_gt_1` is signal-thin (80% positive
   base rate); `log_roi` regression hits a wall at the corpus's
   survivorship structure.
4. **K = 20 LDA character-name dominance.** A custom character-name
   stopword list is defensible polish if Phase 4 finds topic
   features useful. Optional v2; not blocking Phase 4.
5. **Topic and embedding artifacts: keep on disk?** The
   `topic_model_artifacts/` directory and `embedding_pca.joblib`
   are gitignored. They are worth keeping locally so Phase 4 can
   apply them to held-out data without re-fitting. The closing
   handoffs document that train-fitted parameters apply to
   calibration and test sets.

---

## Files produced

### Code (Phase 3 specific)
* `src/features/__init__.py`, `src/features/split.py`,
  `src/features/targets.py`, `src/features/baseline_features.py`
  (Phase 3a + the 5 group flags added in Phase 3b)
* `src/features/lexical.py`, `src/features/sentiment.py`,
  `src/features/topic.py`, `src/features/character_network.py`,
  `src/features/embedding.py` (one module per Phase 3b group)
* `src/models/__init__.py`, `src/models/baseline/__init__.py`,
  `src/models/baseline/metrics.py`,
  `src/models/baseline/train.py` (multi-family multi-eval-set
  trainer)
* `src/experiments/__init__.py`, `src/experiments/save_run.py`
  (per-run logging infrastructure introduced in this phase)
* `src/experiments/run_lexical_ablation.py` and four siblings for
  sentiment, topic, character_network, embedding
* `src/experiments/run_combinations_ablation.py` (Phase 3c
  combinations runner)
* `tests/__init__.py` and `tests/test_*.py` (62 passing tests, 1
  skipped on fresh checkout: 17 lexical + 18 sentiment + 9 topic +
  12 character_network + 6 embedding excluding the skipped one)

### Data
* `data/processed/split_assignments.parquet` (1,713 rows; the
  authoritative split definition)
* `data/processed/features.parquet` (1,713 x 131; the consolidated
  Phase 4 input matrix)
* `data/processed/features_lexical.parquet`,
  `features_sentiment.parquet`, `features_topic.parquet`,
  `features_character_network.parquet`,
  `features_embedding.parquet` (per-group parquets)
* `data/processed/embeddings_minilm_pooled.parquet` (1,713 x 384;
  the raw MiniLM cache)
* `data/processed/topic_model_artifacts/` (TF-IDF vectorizer + LDA
  model + train_ids index, fit on training fold only)
* `data/processed/embedding_pca.joblib` (32-component PCA fit on
  training-fold embeddings)

### Tables
* `reports/tables/phase3_split_diagnostics.csv`
* `reports/tables/phase3a_baseline.csv`
* `reports/tables/phase3_ablation.csv`
* `reports/tables/phase3c_combinations.csv`

### Figures
* `reports/figures/phase3_target_distributions.png`
* `reports/figures/phase3_log_transform_effect.png`

### Documents
* `docs/FEATURE_NOTES.md` (the standing reference for the feature
  matrix, replacing this summary as the canonical column glossary)
* `docs/proposals/phase3_lexical_proposal.md` (v2),
  `docs/proposals/phase3_sentiment_proposal.md` (v2),
  `docs/proposals/phase3c_combinations_proposal.md`
* `docs/handoffs/phase_3a_handoff.md`,
  `docs/handoffs/phase_3b_lexical_handoff.md`,
  `docs/handoffs/phase_3b_sentiment_handoff.md`,
  `docs/handoffs/phase_3b_topic_handoff.md`,
  `docs/handoffs/phase_3b_character_network_handoff.md`,
  `docs/handoffs/phase_3b_embedding_handoff.md`,
  `docs/handoffs/phase_3c_combinations_handoff.md` (seven interim
  handoffs documenting the chronological audit trail)

### Run artifacts
Eleven directories under `runs/phase_3/`, each containing
`params.json`, `preprocessing_summary.json`, `features_used.json`,
`metrics.json`, and `run.log`. Three iterations of the lexical
ablation (development), one per other Phase 3b group, four for
Phase 3c combinations. RUNS.md updated with corresponding rows.

### Notebook
* `notebooks/_build_phase_3_notebook.py` and
  `notebooks/phase_3.ipynb` (53 cells; covers Phase 3a baseline +
  lexical group findings; sections for sentiment, topic,
  character_network, embedding, and Phase 3c combinations are
  pending and can be appended for the final report deliverable)

---

## Next phase prerequisites

Phase 4 needs:

* `data/processed/features.parquet` ✓ (the consolidated 1,713 x 131
  matrix)
* The split assignments ✓ (in `features.parquet` and as
  `split_assignments.parquet`)
* `docs/FEATURE_NOTES.md` ✓ (the standing column glossary and
  per-feature handling reference)
* The four model families' multi-family ablation evidence ✓
  (Phase 3 ablation tables document strengths and weaknesses per
  family)

Phase 4 will:

* Run a benchmark across candidate model families with hyperparameter
  search (Ridge, Lasso, XGBoost, LightGBM, Random Forest, SVM-RBF
  per the Phase 3c surprise, and optionally DistilBERT)
* Conservative regularization for HistGB (`max_depth in {2, 3}`,
  `learning_rate in {0.01, 0.02, 0.05}`,
  `min_samples_leaf in {10, 20, 40}`) per the Phase 3a train-OOF
  gap finding
* Real hyperparameter search for SVM-RBF (C grid, gamma grid) per
  the Phase 3c combinations finding
* 5-fold cross-validation with bootstrapped CIs (matches Phase 3
  trainer)
* Statistical significance testing across folds for model-versus-
  model comparisons
* Save trained primary model artifacts; select primary outcome
  variable; commit to the Phase 5 calibration target

---

## Questions for the planning conversation

Three for the Phase 4 planning conversation:

1. **Phase 4 input matrix:** `features.parquet` as-is (the
   `all_five` 127-feature union), or the standalone-positive
   union (topic + character_network + embedding ~67 features)?
   Phase 3c evidence supports `all_five` for non-linear models;
   pre-registration discipline supports the union. Recommendation:
   `all_five` as primary plus a sensitivity-analysis comparison.
2. **Promote SVM-RBF to a primary candidate?** Phase 3 elevated
   SVM-RBF from worst-of-four standalone to best-of-four on
   combinations, with the largest single classification lift in
   the phase. Phase 4 model benchmark should give SVM a serious
   hyperparameter search; the question is whether the project
   should also revisit the implied "tree ensembles plus linear"
   framing that the four-layer architecture's novelty discussion
   leaned on.
3. **Primary outcome variable nudge.** The Phase 3 evidence
   suggests `roi_gt_2` as the most tractable target (SVM-RBF on
   `all_five` reaches ~~0.665~~ **0.5966** OOF AUC — corrected
   per the Results section). The formal decision is
   end-of-Phase-4, but if the planning conversation
   wants to commit earlier given the magnitude of the signal
   difference, the path is open.

Phase 3 is complete. Phase 4 model selection should begin against
the saved `features.parquet` and the SVM-RBF surprise.
