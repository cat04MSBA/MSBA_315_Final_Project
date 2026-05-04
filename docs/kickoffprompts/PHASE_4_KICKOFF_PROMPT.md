================================================================
Phase 4 kickoff prompt for the new chat (copy-paste this verbatim)
================================================================

I am continuing the MSBA315 ML course project. Phases 1, 2, and 3 are
complete. We are starting Phase 4 (Layer 1: Core Prediction Model).

This kickoff prompt walks you through everything you need to read and
do before any code work. The Phase 4 brief from the planning
conversation will land separately at
`docs/briefs/phase_4_brief.md` once Phase 4 strategy is settled; do
not start implementation work until you have read it.

================================================================
Required reading, in order
================================================================

The four foundation documents are in `docs/`:

1. `docs/PROJECT_CONTEXT.md` : project framing, methodology
   principles, current data status, and the rules in Section 11 for
   when you must escalate to me before acting unilaterally. The
   decisions log in Section 8 has nine dated entries from Phase 3
   that are required context.
2. `docs/CLAUDE_CODE_GUIDELINES.md` : engineering standards. Read
   the whole thing. Section 7 is the phase-summary template. Section
   9 documents the `save_run` per-run experiment-tracking
   infrastructure introduced in Phase 3 (you will use it for every
   model training run).
3. `docs/PROJECT_ROADMAP.md` : nine-phase outline. Phase 4 is Layer
   1 of the four-layer triage system: the core prediction model
   that subsequent layers (calibration, decision rule, scene-level
   SHAP) wrap.
4. `docs/briefs/phase_4_brief.md` : Phase 4 execution brief. Will
   be uploaded separately. Do not start implementation work until
   you have read it.

Phase 3 outputs you must read carefully (this is the input to your
work):

5. `docs/summaries/phase_3_summary.md` : the canonical Phase 3
   record. Replaces seven interim handoffs as the single source of
   truth for what Phase 3 produced and decided.
6. `docs/FEATURE_NOTES.md` : the standing column glossary for the
   feature matrix. Documents every Phase 3 feature, per-feature
   handling decisions, NaN policies, and the wordfreq /
   nrclex / LDA-character-name-dominance / `treat_flagged_as_nan`
   caveats. You will reference this constantly.
7. `docs/DATA_NOTES.md` : Phase 2 corpus reference. Survivorship
   bias, the `data_quality_flag` 30 films, the year and genre
   distributions, and edge cases (the `0`-as-missing convention,
   future-year noise, dedup policies).

Per-phase summaries (lighter read, but skim):

8. `docs/summaries/phase_1_summary.md` : corpus EDA findings.
9. `docs/summaries/phase_2_summary.md` : pipeline + parser audit.

Optional but useful for context:

10. The seven Phase 3 handoffs in `docs/handoffs/`:
    `phase_3a_handoff.md`, `phase_3b_lexical_handoff.md`,
    `phase_3b_sentiment_handoff.md`, `phase_3b_topic_handoff.md`,
    `phase_3b_character_network_handoff.md`,
    `phase_3b_embedding_handoff.md`, `phase_3c_combinations_handoff.md`.
    Each has the multi-family numbers, diagnostic results, and
    open-questions sections at the end. The Phase 3c handoff in
    particular has the SVM-RBF combinations finding that will
    reshape Phase 4 model selection.

Read items 1-7 before doing anything else. Items 8-10 you can read
as needed.

================================================================
Environment setup (one-time, before any code work)
================================================================

Confirm Python 3.11+ and the project's installed dependencies. From
the project root:

```bash
# Required Python packages (some were added during Phase 3):
pip install nltk wordfreq nrclex sentence-transformers scikit-learn pandas numpy scipy pytest

# NLTK data (download once):
python -c "import nltk; [nltk.download(r, quiet=True) for r in ('punkt', 'punkt_tab', 'vader_lexicon')]"

# Sanity check the test suite:
python -m pytest tests/ -v
```

Expected: 62 passed, 1 skipped (the embedding integration test that
needs the encoder cache from disk, which exists on this machine).

If any test fails on a fresh machine, surface it before doing any
modelling work.

================================================================
What Phase 3 left you (the inputs)
================================================================

The full Phase 3 inventory is in `docs/summaries/phase_3_summary.md`,
but the key inputs Phase 4 reads from:

* **`data/processed/features.parquet`** (1,713 rows × 131 columns).
  The consolidated feature matrix. Contains:
  * 127 feature columns (the `all_five` union: structural baseline +
    13 lexical + 22 sentiment + 22 topic + 12 character network +
    32 embedding PCA components)
  * 3 target columns (`log_roi`, `roi_gt_1`, `roi_gt_2`)
  * 1 `split` column (`train`, `cal`, or `test`)

* **`data/processed/split_assignments.parquet`** : the authoritative
  split definition. Same `split` column as above, plus the stratum
  label. 1,199 train, 257 calibration, 257 test. Stratified by
  primary genre and decade bucket. Seed 42. **Read from this; do
  not re-split.**

* **`data/processed/films_joined.parquet`** : Phase 2 master table.
  Source columns + derived columns (genre dummies, log monetary,
  data_quality_flag, etc.). Phase 4 should rarely touch this
  directly because `features.parquet` already has everything Phase 4
  needs, but it is available.

* **Per-group feature parquets** retained on disk:
  `features_lexical.parquet`, `features_sentiment.parquet`,
  `features_topic.parquet`, `features_character_network.parquet`,
  `features_embedding.parquet`. Phase 4 may want to use these for
  feature-subset experiments rather than the consolidated matrix.

* **Auxiliary artifacts:**
  * `data/processed/embeddings_minilm_pooled.parquet` : raw 384-dim
    MiniLM cache, in case Phase 4 wants to retest with a different
    PCA dimensionality.
  * `data/processed/topic_model_artifacts/` : TF-IDF vectorizer +
    LDA model + train_ids index, fit on training fold only.
  * `data/processed/embedding_pca.joblib` : 32-component PCA fit on
    training fold only.

* **Phase 3 ablation tables:**
  * `reports/tables/phase3a_baseline.csv` : the floor numbers
    against which Phase 3b lift was measured (384 rows).
  * `reports/tables/phase3_ablation.csv` : Phase 3b standalone-group
    results (480 rows).
  * `reports/tables/phase3c_combinations.csv` : Phase 3c
    combinations-sub-phase results (384 rows).

* **Multi-family trainer:** `src/models/baseline/train.py`. The
  4-family ablation harness Phase 3 used. Phase 4 will likely need
  to extend it for hyperparameter search and additional candidate
  model families (e.g., LightGBM, XGBoost), but the basic
  pipeline-construction patterns and the SimpleImputer/StandardScaler
  conventions are established.

* **Per-run experiment tracking:** `src/experiments/save_run.py`.
  Phase 4 must use this for every model training run. See Section 9
  of `CLAUDE_CODE_GUIDELINES.md` for the API and the per-run
  directory layout (`runs/<phase>/<timestamp>_<name>/`).

================================================================
Hands-on exploration (15-20 minutes after reading)
================================================================

Run these in order to confirm the environment is healthy and to get
a feel for the inputs you will be modelling against. **Do not re-run
any Phase 3 ablation runner; the artifacts are already on disk.**

### 1. Sanity-check the feature matrix loads

```python
import pandas as pd
from src.utils import paths

df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "features.parquet")
print(f"Shape: {df.shape}")
print(f"Split distribution: {df['split'].value_counts().to_dict()}")
print(f"NaN cells: {df.iloc[:, :-4].isna().sum().sum()} (in feature columns)")
print(f"Target distributions:")
for col in ('log_roi', 'roi_gt_1', 'roi_gt_2'):
    print(f"  {col}: {df[col].describe().to_dict() if col == 'log_roi' else df[col].mean()}")
```

Expected: shape (1713, 131); split distribution train 1199, cal 257,
test 257; ~793 NaN cells (in 54 rows; lexical and sentiment NaNs from
empty-dialogue films); `log_roi` median ~1.06; `roi_gt_1` ~80% positive;
`roi_gt_2` ~64% positive.

### 2. Inspect the feature blocks

```python
# Identify feature blocks by column-name pattern.
feature_cols = [c for c in df.columns if c not in ('log_roi', 'roi_gt_1', 'roi_gt_2', 'split')]
print(f"Total features: {len(feature_cols)}")
blocks = {
    'structural': [c for c in feature_cols if c.startswith('log_') or c == 'release_year_parsed' or c == 'dialogue_to_total_text_ratio' or c.startswith('genre_')],
    'lexical':    [c for c in feature_cols if c in ('mtld_dialogue', 'mtld_action', 'hapax_ratio_dialogue', 'mean_log_frequency', 'rare_word_proportion', 'flesch_kincaid_grade_dialogue', 'flesch_kincaid_grade_action', 'mean_dialogue_line_tokens', 'std_dialogue_line_tokens', 'short_line_proportion', 'question_rate_per_1k_tokens', 'exclamation_rate_per_1k_tokens', 'first_to_second_pronoun_ratio')],
    'sentiment':  [c for c in feature_cols if c.startswith('vader_') or c.startswith('nrc_') or c.startswith('sentiment_') or c.startswith('arc_')],
    'topic':      [c for c in feature_cols if c.startswith('topic_')],
    'character_network': [c for c in feature_cols if c.startswith('network_')],
    'embedding':  [c for c in feature_cols if c.startswith('embed_pc_')],
}
for name, cols in blocks.items():
    print(f"  {name}: {len(cols)} features")
```

Expected total: 127 features. Block counts: structural 26, lexical 13,
sentiment 22, topic 22, character_network 12, embedding 32.

### 3. Read the Phase 3 ablation results

```python
ab = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3_ablation.csv")
oof = ab[(ab['eval_set'] == 'oof') & (ab['model_family'] == 'linear')]
mask = (
    ((oof['target'] == 'log_roi') & (oof['metric'] == 'rmse'))
    | ((oof['target'].isin(['roi_gt_1', 'roi_gt_2'])) & (oof['metric'] == 'auc_roc'))
)
print("Phase 3b standalone-group lift over Phase 3a floor (linear OOF):")
print(oof[mask].pivot_table(index='feature_group', columns=['target', 'metric'], values='lift').round(4))
```

You should see the 5-group standalone pattern: lexical and sentiment
nulls; topic / character_network / embedding partial positives. Read
the Phase 3 summary for the verdict per group.

```python
comb = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3c_combinations.csv")
print("\nPhase 3c combinations: SVM-RBF lift on roi_gt_2 AUC (the surprise):")
svm_rg2 = comb[(comb['eval_set'] == 'oof') & (comb['model_family'] == 'svm') & (comb['target'] == 'roi_gt_2') & (comb['metric'] == 'auc_roc')]
print(svm_rg2[['feature_group', 'phase_3a_floor', 'phase_3c_actual', 'lift']].to_string(index=False))
```

You should see SVM-RBF lifting `roi_gt_2` AUC by +0.063 on `all_five`
and `partial_positives`, +0.043 on `topic_plus_cn`, +0.057 on
`semantic_trio`. SVM was the worst-of-four standalone; on
combinations, it is the best-of-four. **This is the most consequential
Phase 3 finding for Phase 4 model selection.**

### 4. Inspect a few raw screenplays (5 minutes)

For modelling-context understanding, spend a couple of minutes
looking at the underlying data:

```python
import pickle
from src.utils import paths

with open(paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl", "rb") as f:
    sp = pickle.load(f)

# A canonical film
imdb_id = next(iter(sp.keys()))
parsed = sp[imdb_id]
print(f"Film: {imdb_id}, scenes: {parsed.n_scenes}, dialogue lines: {parsed.n_dialogue_lines}")
for char, text in parsed.scenes[5].dialogue_units[:3]:
    print(f"  {char!r}: {text[:80]!r}")

# A data_quality_flag film for contrast
films_df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
flagged = films_df[films_df["data_quality_flag"]]
print(f"\nFlagged films (collapsed scene structure, n={len(flagged)}):")
print(flagged[['movie_name', 'n_scenes', 'n_dialogue_lines']].head(5).to_string(index=False))
```

Things to internalize from this:
* `dialogue_units` are 2-tuples `(character_name, dialogue_text)`. The
  Phase 2 Tier 1.3 fix means you will rarely see empty-text tuples,
  but the empty-text filter is mandatory at every dialogue-derived
  computation.
* `parsed.scenes` is a tuple of `Scene` dataclasses with
  `stage_direction`, `scene_description`, `dialogue_units`. Action
  text lives in the first two.
* `data_quality_flag` films (e.g., Elvis, 12 Angry Men, The Wizard of
  Oz) genuinely have collapsed scene structure (a few thousand
  dialogue lines packed into 2-9 scenes). Whole-screenplay aggregates
  work fine on them; scene-level / per-scene-graph features do not.
  The `treat_flagged_as_nan=True` default in
  `src/features/character_network.py` is the project's convention for
  this.

### 5. Look at one prior run directory

`runs/phase_3/20260503_2155_combinations_all_five/` (or any other) is
a complete `save_run` artifact. Open the five files (`params.json`,
`preprocessing_summary.json`, `features_used.json`, `metrics.json`,
`run.log`) to see what Phase 4 will produce per training run.

================================================================
Decisions already made : implement directly
================================================================

These are settled. Phase 4 inherits them:

1. **Three targets in parallel.** `log_roi` (regression), `roi_gt_1`
   and `roi_gt_2` (classification, threshold-consistent with the
   regression target). Phase 4 will benchmark candidate models on all
   three and pick the primary outcome variable for the cost-decision
   layer at end of phase. **Phase 3 evidence already strongly suggests
   `roi_gt_2`** (SVM-RBF on `all_five` reaches OOF AUC 0.665, well
   inside the project's forward-expected 0.65-0.72 band); the formal
   decision is yours.

2. **Train / calibration / test split is fixed.** 1,199 / 257 / 257
   films. Read split membership from
   `data/processed/split_assignments.parquet` or the `split` column
   in `features.parquet`. Do not re-split. The calibration set is
   reserved for Phase 5; the test set is touched once in Phase 8.

3. **Metric vocabulary.** Regression: MSE, RMSE, MAE, CVRMSE.
   Classification: AUC-ROC, PR-AUC, F1 (at 0.5 threshold), log-loss.
   Both in-sample (train) and out-of-fold (OOF) values reported per
   metric. R-squared is no longer in the reported set.

4. **Pre-registration discipline.** Every model-selection or
   feature-set decision Phase 4 makes must be pre-registered before
   running the benchmark on test data. Phase 3 followed this for
   feature groups and feature combinations; Phase 4 follows it for
   model families and hyperparameter ranges.

5. **`save_run` per-run logging is mandatory.** Every model training
   run wraps in a `save_run` block (see
   `CLAUDE_CODE_GUIDELINES.md` Section 9). Each run produces a
   directory with `params.json`, `preprocessing_summary.json`,
   `features_used.json`, `metrics.json`, `run.log`, and (Phase 4+ only)
   `model.joblib`. The `RUNS.md` index gets a new row per run.

6. **Held-out test set untouched until Phase 8.** This is
   non-negotiable per `PROJECT_CONTEXT.md` Section 6 and Section 11.
   Phase 4's model selection runs on the train split via 5-fold CV;
   the held-out 15% test set is the final evaluation surface in
   Phase 8.

7. **Calibration set untouched in Phase 4.** Reserved for Phase 5
   conformal prediction. Same discipline as the test set; Phase 4
   model selection does not see the calibration set.

================================================================
Tactical recommendations from the prior chat (you can override)
================================================================

These are tactical choices the Phase 4 brief will leave to you. The
prior chat's recommendations follow; if you disagree on any, justify
the deviation in the phase summary.

* **Phase 4 input matrix:** `features.parquet` as-is (the `all_five`
  127-feature union). The Phase 3c combinations evidence shows
  SVM-RBF extracts substantial signal from the maximum-information
  matrix that the standalone-positive union misses. Run a
  sensitivity-analysis comparison on `topic + character_network +
  embedding` (~67 features) so the report can show the trade-off,
  but the primary benchmark uses `all_five`.

* **Candidate model families** (in priority order):
  1. **SVM-RBF.** Phase 3c surprise: went from worst-of-four
     standalone to best-of-four on combinations. Largest single
     classification lift in Phase 3 (+0.081 AUC on `roi_gt_1` for
     `topic_plus_cn`). Give it a real hyperparameter search:
     C grid `[0.01, 0.1, 1, 10, 100]`, gamma grid
     `["scale", "auto", 0.001, 0.01, 0.1]`.
  2. **HistGradientBoosting** (sklearn) and / or **LightGBM**.
     Phase 3 used HistGB at conservative defaults (max_depth=4,
     learning_rate=0.05) and surfaced a roughly 0.20 train-OOF AUC
     gap on classification. Phase 4 should explore even more
     conservative regularization: `max_depth in {2, 3}`,
     `learning_rate in {0.01, 0.02, 0.05}`,
     `min_samples_leaf in {10, 20, 40}`,
     `max_iter in {100, 200, 500}` with early stopping.
  3. **Ridge / Logistic-L2** (the Phase 3 linear baseline). Phase 4
     should keep this as the constant comparator and confirm whatever
     primary candidate beats it across folds with statistical
     significance.
  4. **Random Forest.** The brief specified RF as a Phase 4
     candidate. Worth benchmarking even though HistGB is the more
     modern choice for the same paradigm.
  5. **XGBoost / LightGBM (other tree boosting frameworks).** If
     you want the report to claim a thorough modelling-side
     benchmark, include one of these in addition to HistGB.
     Otherwise HistGB is sufficient as the gradient-boosting
     representative.
  6. **DistilBERT or sentence-transformer fine-tune.** Brief mentions
     this as optional. Phase 3 already pre-pooled MiniLM embeddings
     into the feature matrix; Phase 4 could optionally fine-tune a
     small transformer head on top, but n = 1,199 makes this fragile.
     Defer unless the gradient-boosting candidates underperform.

* **Hyperparameter search:** sklearn's `RandomizedSearchCV` with 50
  iterations is the standard recommendation for the candidate-grids
  above. `GridSearchCV` is acceptable for the simpler grids. Each
  search runs inside the same 5-fold CV the Phase 3 trainer used;
  the hyperparameter selection happens on the inner CV folds, the
  reporting metric is OOF on the outer folds.

* **Significance testing** for model-versus-model comparisons:
  paired bootstrap on the OOF prediction series. 1,000 resamples,
  seed 42 (project standard). Two-sided p-value for "model A's OOF
  metric differs from model B's." The Phase 3 trainer's
  `bootstrap_ci` helper is the building block.

* **Primary outcome variable selection:** end-of-Phase-4 decision.
  Pre-register the criteria you will use to pick (e.g., "highest OOF
  AUC with confidence interval that does not overlap chance, with
  gap to second-best target meaningful at p < 0.05") before running
  the benchmark.

================================================================
Three things to be aware of from prior phases
================================================================

1. **HistGB substantially overfits at conservative defaults on this
   corpus.** Phase 3a established a roughly 0.20 train-OOF gap on
   classification AUC, even with `max_depth=4`,
   `learning_rate=0.05`, and `early_stopping=True`. Phase 3c
   confirmed: HistGB train-OOF gap on `roi_gt_2` AUC for `all_five`
   is approximately 0.27. Phase 4 hyperparameter search must
   explore aggressive regularization (lower learning rate, larger
   min_samples_leaf, smaller max_depth). The
   `phase_3a_handoff.md` Section 3.2 has the train-vs-OOF table
   that motivated this.

2. **Linear regression on `log_roi` is signal-limited.** No Phase 3
   feature group, no Phase 3c combination, beat the Phase 3a revised
   dialogue-only floor on linear OOF RMSE (1.339). The signal is in
   non-linear models on classification targets, not in linear
   regression on the regression target. Phase 4 should expect
   non-linear models to win on the regression target as well, or
   document a finding that linear regression has hit a corpus-
   structural ceiling.

3. **MPS encoder embeddings have minor floating-point variation.**
   The MiniLM forward pass on Apple Silicon MPS produces embeddings
   that differ by ~1e-7 across runs even with the same seed. This
   propagates through PCA into `LogisticRegressionCV`'s C-selection,
   producing roughly 1pp AUC variation on `roi_gt_1` between runs.
   Regression metrics (which use `RidgeCV` closed-form LOO-GCV)
   reproduce exactly. Documented in the embedding handoff.

================================================================
Workflow conventions (unchanged from Phase 3)
================================================================

* Code in `src/`. Narrative notebooks in `notebooks/` calling those
  functions. Don't define heavy logic inside the notebook.
* Each phase ends with a notebook generated from a builder script
  (`notebooks/_build_phase_4_notebook.py` produces
  `notebooks/phase_4.ipynb`). Easier to diff and regenerate. The
  Phase 3 notebook is the structural template.
* Use git as the code-history mechanism. No `_old.py` /
  `final_v2.py` / `_archive/` folders. Tag the milestone when
  Phase 4 lands (`phase-4-complete`).
* Logging: default level is INFO. Milestones-only at INFO; per-fold
  scores and per-iteration hyperparameter-search progress at DEBUG.
  `set_log_level("WARNING")` silences INFO during long benchmark
  runs. The `save_run` block captures both INFO and DEBUG to
  `run.log`.
* Notebook markdowns are submission-quality (formal academic
  register, no em dashes, no second-person pronouns, no process
  narrative about what was tried and reverted, no internal jargon
  like "the brief" or "the planning conversation"). The Phase 3
  notebook (`notebooks/phase_3.ipynb`) is the style reference.
* Tests: every new module gets a `tests/test_<module>.py` with at
  least a smoke test (compute on first 10 rows) plus a unit test
  per non-trivial helper.

================================================================
Mandatory escalation points (Phase 4 has the heaviest of the project)
================================================================

`PROJECT_CONTEXT.md` Section 11 is the source of truth. Phase 4
specifically requires escalation at:

* **Before fixing the candidate-model list and hyperparameter
  grids.** Pre-register the model families and hyperparameter ranges
  in the Phase 4 brief or in a Phase 4 plan document the planning
  conversation reviews. Same multiple-comparisons discipline that
  Phase 3 followed for feature groups.
* **Before any test-set evaluation.** Phase 4 does not touch the
  held-out test set. If a planning-conversation conclusion changes
  this, it gets surfaced and logged in the decisions log first.
* **End of phase (mandatory).** Per Section 11, Phase 4 ends with
  escalation: which model is primary, which outcome variable is
  primary for the cost-decision layer (Phase 6), and whether to
  invest in ensemble work before moving to Phase 5. Write the
  Phase 4 summary, prepare specific questions for the planning
  conversation, and tell me explicitly: "Phase 4 complete. Please
  bring the summary and these questions to the planning
  conversation before starting Phase 5."

================================================================
What is open for the planning conversation to settle
================================================================

The Phase 3 final summary lists three open questions that Phase 4
needs the planning conversation's input on. Read them in
`docs/summaries/phase_3_summary.md` Section "Questions for the
planning conversation":

1. Phase 4 input matrix: `features.parquet` (all_five, 127 cols) or
   the standalone-positive union (~67 cols)?
2. Promote SVM-RBF to a primary candidate? (The Phase 3c surprise
   argues yes; the original four-layer architecture discussion
   leaned toward tree ensembles plus linear.)
3. Primary outcome variable nudge toward `roi_gt_2` based on Phase 3
   evidence, or wait for Phase 4 benchmark to settle it?

The planning conversation will provide its answers in the Phase 4
brief. Do not start implementation work until you have the brief.

================================================================
Get started
================================================================

Once you have read the foundation documents, the Phase 3 summary,
the FEATURE_NOTES, the DATA_NOTES, and the Phase 4 brief (when it
arrives), walk me through your plan. Specifically:

1. Confirm you have read items 1-7 in "Required reading."
2. Confirm the test suite passes on your machine (`python -m
   pytest tests/`; expected: 62 passed, 1 skipped).
3. Run the hands-on exploration cells above (15-20 minutes).
4. Walk me through your interpretation of the Phase 3 summary's
   three open questions and how the Phase 4 brief settles them.
5. Once the Phase 4 brief is in hand, propose your candidate-model
   list and hyperparameter grids before running any benchmark.
   Pre-register, then implement.

After candidate models are pre-registered, you have the full
Phase 4 modelling work ahead: hyperparameter search per candidate,
significance testing across folds, primary-model selection, primary-
outcome-variable selection, then the Phase 4 summary and escalation
to the planning conversation before Phase 5 begins.

Stop and ask if anything is unclear.
