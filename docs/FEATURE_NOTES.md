# FEATURE_NOTES: Phase 3 feature matrix reference

> Standing reference for the Phase 3 feature matrix. Future phases
> read from here rather than re-deriving the feature inventory.
> Last updated: 2026-05-03 (Phase 3 complete).

---

## 1. Feature matrix at a glance

* **Source files:**
  * `data/processed/features.parquet` (1,713 rows x 131 columns: 127
    feature columns + 3 target columns + 1 split-assignment column)
  * Per-group feature parquets (kept on disk for re-use and audit):
    `features_lexical.parquet`, `features_sentiment.parquet`,
    `features_topic.parquet`, `features_character_network.parquet`,
    `features_embedding.parquet`. Each group's parquet contains its
    model features plus any diagnostic-only columns (leading
    underscore convention).
  * Embedding cache: `embeddings_minilm_pooled.parquet` (1,713 x 384,
    the raw MiniLM pooled embeddings before PCA).
  * Topic-model artifacts: `topic_model_artifacts/` (TF-IDF
    vectorizer + LDA model + train_ids index, fit on training fold
    only).
  * PCA artifact for embeddings: `embedding_pca.joblib`.
* **One row per film**, keyed by `imdb_id` (the same key Phase 2's
  `films_joined.parquet` uses).

### Headline structure

| Block | Columns | Source |
|---|---:|---|
| Structural baseline (revised) | 26 | Phase 3a, log-transformed counts + log_runtime + genre dummies + era |
| Lexical | 13 | Phase 3b lexical group |
| Sentiment | 22 | Phase 3b sentiment group |
| Topic | 22 | Phase 3b topic group |
| Character network | 12 | Phase 3b character network group |
| Embedding | 32 | Phase 3b embedding group |
| **Total feature columns** | **127** | (matches the `all_five` Phase 3c combination) |
| Targets | 3 | `log_roi`, `roi_gt_1`, `roi_gt_2` from Phase 3 targets module |
| Split assignment | 1 | `split` column with values `train` / `cal` / `test` |

NaN cells in the feature block: 793 across 54 rows. These are
films with empty or near-empty dialogue (the lexical and sentiment
groups produce NaN on these). Phase 4 modelling pipelines apply
median imputation inside the cross-validation folds, which is
already established by the Phase 3 trainer.

---

## 2. Phase 3 ablation summary (the headline result)

Two ablation tables drive the Phase 4 feature decision:

* `reports/tables/phase3_ablation.csv` (480 rows): the Phase 3b
  standalone-group ablation. Each of the five groups added on top
  of the Phase 3a revised dialogue-only floor, evaluated under all
  four model families (linear, HistGB, KNN, SVM-RBF) on both train
  (in-sample) and OOF (cross-validation) eval sets.
* `reports/tables/phase3c_combinations.csv` (384 rows): the
  Phase 3c combinations sub-phase. Four pre-specified combinations
  (`all_five`, `partial_positives`, `topic_plus_cn`,
  `semantic_trio`) evaluated under the same harness.

### Phase 3b standalone verdict (linear OOF lift over the floor)

| Group | log_roi RMSE | roi_gt_1 AUC | roi_gt_2 AUC | Verdict |
|---|---:|---:|---:|---|
| lexical | +0.011 (worse) | +0.007 | -0.002 | Null |
| sentiment | +0.019 (worse) | -0.012 | +0.009 | Null |
| topic | +0.016 (worse) | **+0.032** | +0.012 | Partial positive (`roi_gt_1` dominant) |
| character_network | +0.009 (worse) | +0.013 | **+0.016** | Partial positive (`roi_gt_2` dominant) |
| embedding | **-0.007** | +0.013 | +0.006 | Partial positive (regression dominant) |

### Phase 3c combinations verdict (most informative results)

* `topic_plus_cn` (60 features) is the parsimonious winner on
  linear: only combination with both classification AUCs
  positive (+0.021 each).
* `all_five` (130 features) is the maximum-information matrix.
  SVM-RBF reaches `roi_gt_2` AUC 0.665 OOF (lift +0.063) and
  `roi_gt_1` AUC 0.614 OOF (lift +0.056) on it.
* SVM-RBF was the worst-of-four standalone; on combinations, it
  becomes the best-of-four on classification. The largest single
  classification lift in Phase 3 work is SVM on `topic_plus_cn`
  `roi_gt_1` AUC: +0.081.
* The pre-registered linear-OOF lift bands missed on 10 of 12
  headline metrics. Standalone lifts do not compose additively for
  linear regression at this corpus size; combinations help under
  non-linear kernels (SVM-RBF) but hurt under regularized linear.

This is documented in detail in the seven Phase 3 handoffs:

* `docs/handoffs/phase_3a_handoff.md`
* `docs/handoffs/phase_3b_lexical_handoff.md`
* `docs/handoffs/phase_3b_sentiment_handoff.md`
* `docs/handoffs/phase_3b_topic_handoff.md`
* `docs/handoffs/phase_3b_character_network_handoff.md`
* `docs/handoffs/phase_3b_embedding_handoff.md`
* `docs/handoffs/phase_3c_combinations_handoff.md`

---

## 3. Column glossary

All feature columns are float64. The genre one-hot dummies are 0/1
floats. NaN values appear in the lexical and sentiment groups for
films with empty or near-empty dialogue.

### 3.1 Structural baseline (26 columns)

Carried over from Phase 3a's revised dialogue-only configuration.
Six heavy-tailed structural counts have `log1p` applied before
z-scoring; the seventh structural feature (the bounded ratio
`dialogue_to_total_text_ratio`) is left untransformed.

| Column | Description |
|---|---|
| `log_n_scenes` | log1p of the per-screenplay scene count |
| `log_n_unique_characters` | log1p of distinct speaking characters (post Phase 2 Tier 1.3 filter) |
| `log_n_dialogue_lines` | log1p of dialogue tuple count |
| `log_total_dialogue_chars` | log1p of total dialogue text length |
| `log_total_action_chars` | log1p of total stage-direction + scene-description text length |
| `dialogue_to_total_text_ratio` | dialogue / (dialogue + action). Bounded in [0, 1] |
| `log_parse_warning_count` | log1p of parser-warning count |
| `release_year_parsed` | year of release as numeric feature |
| `log_runtime` | log1p of runtime in minutes |
| `genre_Action` ... `genre_Other` (13 dummies) | primary-genre one-hot. Genres with fewer than 30 films collapsed into `Other` per Phase 2 |

### 3.2 Lexical group (13 model features + 1 diagnostic-only)

Phase 3b group 1 of 5. Standalone verdict: null. Implementation in
`src/features/lexical.py`. Token-form matching via NLTK
`word_tokenize`; no lemmatization. Empty-text dialogue filter applied.

#### Vocabulary diversity (3)

| Column | Description |
|---|---|
| `mtld_dialogue` | Measure of Textual Lexical Diversity on dialogue text. Length-robust replacement for type-token ratio. McCarthy and Jarvis 2010 default threshold 0.72. NaN for sequences shorter than 50 tokens |
| `mtld_action` | Same measure on action text (stage_direction + scene_description). Captures dialogue-vs-action stylistic divergence |
| `hapax_ratio_dialogue` | Proportion of distinct dialogue tokens occurring exactly once. Captures vocabulary tail |

#### Lexical sophistication (2, dialogue only)

| Column | Description |
|---|---|
| `mean_log_frequency` | Mean wordfreq Zipf-scale log-frequency over dialogue tokens. Lower = rarer/more sophisticated vocabulary |
| `rare_word_proportion` | Proportion of dialogue tokens in the bottom Zipf-frequency quartile (corpus-wide cutoff at 5.110, computed from a 200-film sample) |

#### Readability (2)

| Column | Description |
|---|---|
| `flesch_kincaid_grade_dialogue` | Flesch-Kincaid grade level on dialogue. Compresses on fragmented dialogue (caveat documented in lexical proposal v2) |
| `flesch_kincaid_grade_action` | Same measure on action text |

#### Length statistics (3, dialogue only)

| Column | Description |
|---|---|
| `mean_dialogue_line_tokens` | Mean tokens per dialogue line |
| `std_dialogue_line_tokens` | Standard deviation of tokens per dialogue line |
| `short_line_proportion` | Proportion of dialogue lines under 5 tokens |

#### Punctuation and pronouns (3, dialogue only)

| Column | Description |
|---|---|
| `question_rate_per_1k_tokens` | Question marks per 1,000 dialogue tokens |
| `exclamation_rate_per_1k_tokens` | Exclamation marks per 1,000 dialogue tokens |
| `first_to_second_pronoun_ratio` | Count of first-person tokens divided by count of second-person tokens (epsilon 1.0 in denominator). Includes archaic forms (`thou`, `thee`, `thy`, `thine`, `thyself`) per the planning-conversation directive |

#### Diagnostic-only

| Column | Description |
|---|---|
| `_oov_rate_dialogue` | Per-film proportion of dialogue tokens not found in the wordfreq English vocabulary. Used in Section 9 diagnostic checks; not a model feature |

### 3.3 Sentiment group (22 model features + 2 diagnostic-only)

Phase 3b group 2 of 5. Standalone verdict: null. Implementation in
`src/features/sentiment.py`. VADER on dialogue lines, NRC token-form
matching on dialogue tokens (NLTK English stopwords removed before
NRC matching, no lemmatization).

#### VADER aggregates (3)

| Column | Description |
|---|---|
| `vader_compound_mean` | Mean VADER compound score across dialogue lines. In [-1, 1] |
| `vader_compound_std` | Standard deviation of VADER compound scores. Captures emotional volatility |
| `vader_compound_range` | Maximum minus minimum compound score. Captures emotional dynamic range |

#### NRC emotion proportions (8)

`nrc_anger_proportion`, `nrc_anticipation_proportion`,
`nrc_disgust_proportion`, `nrc_fear_proportion`, `nrc_joy_proportion`,
`nrc_sadness_proportion`, `nrc_surprise_proportion`,
`nrc_trust_proportion`. Each is the proportion of non-stopword
dialogue tokens whose surface form appears in the NRC lexicon and
is annotated with the given emotion. Multiple emotions per word are
allowed; each tag contributes once.

#### Sentiment quartile-trajectory (5, scene-windowed)

| Column | Description |
|---|---|
| `sentiment_q1_compound_mean` | Mean VADER compound for the first quartile of dialogue lines |
| `sentiment_q2_compound_mean` | Mean VADER compound for the second quartile |
| `sentiment_q3_compound_mean` | Mean VADER compound for the third quartile |
| `sentiment_q4_compound_mean` | Mean VADER compound for the fourth quartile |
| `sentiment_volatility_concentration` | Maximum minus minimum of the four per-quartile standard deviations. Captures whether emotional volatility concentrates in one quartile (e.g., a late climax) |

#### Reagan arc archetype similarities (6, arc-clustered)

`arc_similarity_rags_to_riches`, `arc_similarity_tragedy`,
`arc_similarity_man_in_a_hole`, `arc_similarity_icarus`,
`arc_similarity_cinderella`, `arc_similarity_oedipus`. Each is
the cosine similarity between the film's z-score-normalized per-line
sentiment trajectory (interpolated to length 100) and the named
canonical archetype template. Templates are hand-coded smoothed
mathematical shapes per proposal v2 Section 3.4 (linear ramps for
Rags-to-Riches and Tragedy, single-trough cosine for Man-in-a-Hole,
single-peak cosine for Icarus, two-trough rescaled cosine for
Cinderella, two-peak reflection for Oedipus).

#### Diagnostic-only

| Column | Description |
|---|---|
| `_nrc_oov_rate_dialogue` | Per-film proportion of dialogue tokens not in the NRC lexicon |
| `_vader_zero_compound_rate` | Per-film proportion of dialogue lines for which VADER returned a compound score of 0 (no detected sentiment cues) |

### 3.4 Topic group (22 model features)

Phase 3b group 3 of 5. Standalone verdict: partial positive
(`roi_gt_1` dominant; linear OOF AUC lift +0.032). Implementation
in `src/features/topic.py`. LDA fit on training-fold screenplays
only (no test/cal leakage), K = 20 topics, max_iter=10, learning
method=batch, train perplexity 8812.

| Column pattern | Description |
|---|---|
| `topic_00_proportion` ... `topic_19_proportion` | Per-film LDA topic proportions, one column per topic. Sum to 1.0 across the 20 columns |
| `topic_concentration_entropy` | Shannon entropy of the per-film topic distribution. Lower = more topically focused; higher = more topically diffuse |
| `topic_dominant_id` | Integer id [0, 19] of the topic with the highest proportion for the film |

A known caveat documented in the topic handoff: with K = 20 on
screenplay text, several topics are character-name-dominated (e.g.,
topic 02 contains `peter`, `bruce`, `elizabeth`, `charlie`, `dude`,
`movie`, `santa`). A custom character-name stopword list would
likely improve topic interpretability and is a defensible v2 polish
item; v1 keeps the topics as-is for honest standalone evaluation.

### 3.5 Character network group (12 model features + 1 diagnostic-only)

Phase 3b group 4 of 5. Standalone verdict: partial positive
(`roi_gt_2` dominant; linear OOF AUC lift +0.016, plus a
univariate r = -0.102 between `network_lead_role_count` and
`roi_gt_2`, the first cross-group |r| > 0.10 of the phase).
Implementation in `src/features/character_network.py`. Empty-text
filter applied; `data_quality_flag` films excluded from per-graph
metrics by default (treat_flagged_as_nan=True).

| Column | Description |
|---|---|
| `network_n_significant_characters` | Number of characters speaking at least 5 lines |
| `network_lead_role_count` | Number of characters whose dialogue share is at least 10% of the film's total dialogue |
| `network_dialogue_gini` | Gini coefficient of the per-character dialogue-line distribution. Measures concentration |
| `network_density` | Edge density of the character co-occurrence graph (characters share an edge if they appear in the same scene) |
| `network_n_components` | Number of connected components in the co-occurrence graph |
| `network_mean_clustering_coefficient` | Mean local clustering coefficient |
| `network_top1_dialogue_share` | Fraction of total dialogue lines spoken by the most-talkative character |
| `network_top3_dialogue_share` | Fraction of total dialogue lines spoken by the top 3 characters combined |
| `network_top1_eigenvector_centrality` | Eigenvector centrality of the most-central character in the co-occurrence graph |
| `network_modularity` | Greedy-modularity community-detection score on the co-occurrence graph |
| `network_max_betweenness_centrality` | Maximum betweenness centrality across all characters |
| `network_diameter` | Diameter of the largest connected component (longest shortest path) |

#### Diagnostic-only

| Column | Description |
|---|---|
| `_n_dropped_minor_characters` | Number of characters excluded by the 5-line minimum threshold |

### 3.6 Embedding group (32 model features)

Phase 3b group 5 of 5. Standalone verdict: partial positive
(regression-target dominant; the only group with negative log_roi
RMSE lift across families on standalone, plus 2 univariate
|r| > 0.10 features: `embed_pc_01` r = +0.114 with `log_roi`,
`embed_pc_04` r = +0.106 with `roi_gt_2`). Implementation in
`src/features/embedding.py`. MiniLM
(`sentence-transformers/all-MiniLM-L6-v2`) pooled per scene then
mean-pooled per film, producing a 384-dim per-film vector. PCA
fit on training-fold pooled embeddings only (32 components,
cumulative variance explained 73.9%); transform applied to all
1,713 films.

| Column pattern | Description |
|---|---|
| `embed_pc_00` ... `embed_pc_31` | The 32 principal-component projections of each film's mean-pooled MiniLM embedding |

The raw 384-dim pooled embeddings are kept on disk at
`embeddings_minilm_pooled.parquet` so Phase 4 can experiment with
different PCA dimensionalities or skip PCA entirely if a model
candidate handles 384-dim input well.

---

## 4. Per-feature handling of edge cases

### 4.1 `data_quality_flag` films (n = 30 corpus, 24 in train)

Phase 2 flagged 30 films with collapsed scene structure (fewer than
10 scenes containing more than 50,000 dialogue characters). Per
the planning-conversation directive (Phase 3 lexical handoff
Section 1), all 30 films stay in train/cal/test for sample size,
with per-feature handling that depends on whether the feature
needs scene-level integrity.

| Group | Handling | Why |
|---|---|---|
| Lexical | Use as-is | Whole-screenplay aggregates; collapsed scenes do not affect lexical metrics |
| Sentiment | Use as-is for whole-screenplay aggregates and quartile-windowed features (windowed by dialogue-line index, not scene index). Use as-is for arc archetypes (computed from dialogue-line trajectory) | Same robustness rationale as lexical |
| Topic | Use as-is | LDA operates on the bag of dialogue tokens; scene structure does not enter the model |
| Character network | **Excluded from graph computation** (treat_flagged_as_nan=True default) | Graph density, modularity, and centrality require meaningful scene boundaries. Flagged films get NaN for network features; the modelling pipeline imputes |
| Embedding | Use as-is | MiniLM pools across scenes; a film with all dialogue in 4 scenes is represented by 4 scene-level embeddings averaged |

### 4.2 Empty-text dialogue placeholders

Phase 2 Tier 1.3 fix: the parser inserts empty-text dialogue
placeholders into `dialogue_units` when scene structure breaks the
character->dialogue pairing rule. Every dialogue-derived feature
applies the empty-text filter defensively at module entry,
matching the lexical group's pattern.

### 4.3 NaN cells in the consolidated matrix

793 NaN cells across 54 rows in the feature block of
`features.parquet`. Source: lexical and sentiment groups produce
NaN on films with empty or near-empty dialogue (typically 6 to 10
films corpus-wide). The Phase 3 trainer's pipeline applies
`SimpleImputer(strategy="median")` to numeric features inside each
cross-validation fold, so the model never sees NaN. Phase 4 should
keep this same imputation pattern.

### 4.4 Wordfreq deviation (lexical sophistication features)

The lexical proposal v2 specified SUBTLEX-US (Brysbaert and New
2009) as the frequency reference for `mean_log_frequency` and
`rare_word_proportion`. Canonical SUBTLEX-US download URLs returned
404 at implementation time, so the implementation uses the
`wordfreq` Python package (Speer 2018), whose English Zipf-scale
log-frequencies are computed from a mixture INCLUDING OpenSubtitles
(subtitle-domain, similar to SUBTLEX in spirit). The mixture is
not pure SUBTLEX-US but the conceptual mechanism the proposal
selected SUBTLEX-US for (subtitle-derived frequencies match
dialogue input better than prose-derived) is preserved. Feature
names dropped the `_subtlex` suffix to be honest about the source.

### 4.5 NRC sourcing via `nrclex` package

The sentiment proposal v2 specified the NRC EmoLex lexicon
(Mohammad and Turney 2013). The canonical Saif Mohammad page is
form-gated, blocking automated download. The implementation uses
the `nrclex` Python package, which ships the same word-emotion
mappings under the original publication's research-use license.
The data is byte-identical to the canonical text-form file at the
word-emotion-tag level.

### 4.6 LDA character-name dominance (topic group, known limitation)

With K = 20 LDA on screenplay dialogue, several topics are
dominated by character names (e.g., topic 02 weights heavily on
`peter`, `bruce`, `elizabeth`, `charlie`). This is a property of
the source text (character names are high-frequency tokens in
dialogue) and is not corrected in v1 of the topic group. A custom
character-name stopword list (built from the per-film
`<character>` tag values) would likely improve topic
interpretability and is a defensible polish item for a v2 if the
report needs cleaner topic labels. The standalone topic group
ablation result is reported for the v1 topics as-is.

### 4.7 LDA fit on training fold only (no leakage)

The TF-IDF vectorizer and LDA model are fit on the 1,199-film
training split exclusively. The fitted artifacts are persisted to
`data/processed/topic_model_artifacts/` along with the
`train_ids.json` index for full audit-trail. The `transform` step
is applied to all 1,713 films (train + calibration + test) using
the train-fitted parameters. This is the same no-leakage pattern
the project applies to PCA fitting in the embedding group.

### 4.8 PCA fit on training fold only (no leakage)

The PCA reducing the 384-dim MiniLM embeddings to 32 components is
fit on the 1,199 training films' pooled embeddings. The fitted
PCA is persisted to `data/processed/embedding_pca.joblib`. The
transform is applied to all 1,713 films. The 73.9% cumulative
variance explained by 32 components is the train-fitted figure.

### 4.9 Redundancies known to exist

* `mean_log_frequency` and `rare_word_proportion` correlate at
  r = -0.939 on the corpus (planning-conversation v2 review noted
  this; both kept for Phase 4 optionality).
* `topic_dominant_id` is determined by the argmax of the 20
  per-topic proportion features and therefore carries no information
  beyond them. Phase 4 may drop it as redundant.
* The 4 sentiment-quartile means are linearly related to
  `vader_compound_mean` (their average approximates it). Modelling
  pipelines using strong regularization absorb this; tree models
  ignore it.
* Within the 6 Reagan arc similarities, Man-in-a-Hole and Icarus
  are reflections of each other by template construction, so their
  similarities are roughly inverse-correlated.

These redundancies are documented for transparency. The standalone
multi-family ablation evidence and the Phase 3c combinations
evidence both indicate that retaining the features rather than
preemptively dropping is the right call: no Phase 3 model family
showed performance loss attributable to redundancy specifically,
and Phase 4 candidate models will handle redundancy at training
time.

---

## 5. Train / calibration / test split

The split assignment carried in `features.parquet`'s `split`
column is the authoritative split definition for every Phase 4+
operation. Reading from `data/processed/split_assignments.parquet`
returns the same data; the `split` column in `features.parquet`
is a convenience copy.

* Train: 1,199 films (70%). Used for cross-validation in Phase 3
  ablations and in Phase 4 model selection.
* Calibration: 257 films (15%). Held out in Phase 3; reserved for
  Phase 5 conformal prediction.
* Test: 257 films (15%). Held out in Phase 3 and Phase 4; touched
  once in Phase 8 final evaluation.

Stratified by `(primary_genre_bucketed, decade_bucket)` with rare
cells (fewer than 5 films) pooled into a single `rare|rare`
stratum. 57 strata total; every named stratum has at least one
film in each split. Random seed 42.

---

## 6. Where to find more

* **Phase 3 ablation tables:** `reports/tables/phase3_ablation.csv`
  (Phase 3b standalone), `reports/tables/phase3c_combinations.csv`
  (Phase 3c combinations), `reports/tables/phase3a_baseline.csv`
  (the floor numbers).
* **Phase 3 handoffs:** `docs/handoffs/phase_3a_handoff.md` and
  `docs/handoffs/phase_3b_*_handoff.md` (5 group handoffs) and
  `docs/handoffs/phase_3c_combinations_handoff.md`.
* **Phase 3 final summary:** `docs/summaries/phase_3_summary.md`.
* **Per-group implementation:** `src/features/lexical.py`,
  `src/features/sentiment.py`, `src/features/topic.py`,
  `src/features/character_network.py`, `src/features/embedding.py`.
* **Multi-family trainer:** `src/models/baseline/train.py`.
* **Per-run metadata:** `runs/phase_3/<timestamp>_<name>/`.
  Each directory contains `params.json`,
  `preprocessing_summary.json`, `features_used.json`,
  `metrics.json`, `run.log`. The full Phase 3 work is reproducible
  from the recorded git SHA plus these per-run artifacts.
* **Decisions log:** `docs/PROJECT_CONTEXT.md` Section 8 has
  dated entries for every strategic decision Phase 3 surfaced.
