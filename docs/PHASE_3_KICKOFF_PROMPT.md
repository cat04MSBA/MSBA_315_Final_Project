================================================================
Phase 3 kickoff prompt for the new chat (copy-paste this verbatim)
================================================================

I'm continuing the MSBA315 ML course project. Phases 1 and 2 are complete
and pushed to git tagged `phase-2-complete`. We're now starting Phase 3.

================================================================
Required reading, in order
================================================================

The four foundation documents are in `docs/`:
  - docs/PROJECT_CONTEXT.md       -- project framing, methodology, decisions log,
                                     data summary, and the rules for when you
                                     must ask me before acting
  - docs/CLAUDE_CODE_GUIDELINES.md -- engineering standards, escalation rules,
                                     phase-summary template (Section 7 now
                                     includes a "Strategic decisions" section)
  - docs/PROJECT_ROADMAP.md       -- nine-phase outline
  - docs/briefs/phase_3_brief.md  -- Phase 3 execution brief

Plus what Phase 1 and 2 produced:
  - docs/DATA_NOTES.md                       -- column glossary for
                                                films_joined.parquet, edge
                                                cases, biases. Read this
                                                carefully; it documents the
                                                0-as-missing convention,
                                                survivorship bias, and the
                                                data_quality_flag column.
  - docs/summaries/phase_1_summary.md        -- corpus EDA findings
  - docs/summaries/phase_2_summary.md        -- pipeline + parser audit
                                                + Tier-1 enhancements

Read all of these before doing anything else.

================================================================
Hands-on exploration (5 minutes after reading)
================================================================

Run a quick scratch cell to confirm the artifacts load and to get a
feel for the data shape. Do not re-run `build_corpus.py` — the
artifacts are already on disk and pass the validator.

```python
import pandas as pd, pickle
from src.utils import paths

df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {sorted(df.columns)}")
print(df.dtypes.to_string())
print()
print("Sample rows:")
print(df[["imdb_id", "movie_name", "release_year_parsed", "primary_genre_bucketed",
         "budget", "revenue", "effective_rating", "n_scenes", "n_unique_characters",
         "n_dialogue_lines", "data_quality_flag"]].head(10))
print()
print(f"data_quality_flag count: {df['data_quality_flag'].sum()}  (informational; do not pre-filter)")

with open(paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl", "rb") as f:
    sp = pickle.load(f)
sample = sp[df.iloc[0]['imdb_id']]
print()
print(f"Sample ParsedScreenplay: {df.iloc[0]['movie_name']}")
print(f"  n_scenes: {sample.n_scenes}")
print(f"  First scene's first 3 dialogue_units:")
for char, line in sample.scenes[0].dialogue_units[:3]:
    print(f"    {char!r}: {line[:80]!r}")
```

To confirm pipeline health: `python -m src.data.validate_processed_corpus`
(takes 3 seconds, hard-asserts every invariant).

================================================================
Decisions already made — implement directly
================================================================

The Phase 3 brief Section 2 documents strategic decisions that came
from the planning conversation. The brief's only explicit question to
me ("regression target: log(ROI) or other?") has been answered: use
**`log_roi = log(revenue) - log(budget)`** (natural log).

Three reasons: (1) decomposable from the existing `log_revenue` and
`log_budget` columns; (2) approximately symmetric, satisfying standard
regression assumptions; (3) consistent with the two classification
targets — `log_roi > 0` is equivalent to `ROI > 1`, and
`log_roi > log(2) ≈ 0.69` is equivalent to `ROI > 2`.

`log10(ROI)` would also work; `log_revenue` alone (without budget
normalization) was rejected because it conflates success with
production scale.

================================================================
Tactical recommendations from the prior chat (you can override)
================================================================

These are tactical choices the brief leaves to you. The prior chat's
recommendations follow; if you disagree on any, justify the deviation
in the phase summary.

* **Train/cal/test split: 70/15/15**, stratified jointly by
  `primary_genre_bucketed` and a `decade_bucket` column where
  `decade_bucket = "pre-1980"` for pre-1980 films and the actual
  decade otherwise. Calibration set of ~257 is sufficient for stable
  conformal prediction in Phase 5; 70% training is generous for 1,713
  films.

* **Baseline model — regression: `RidgeCV`** with a reasonable alpha
  grid (e.g., `[0.01, 0.1, 1, 10, 100]`).
  **Baseline model — classification: `LogisticRegressionCV`** with
  L2 penalty, same alpha-grid logic.
  Both are interpretable "constant comparators" with regularization
  that handles the multicollinearity that will appear once embedding
  features arrive in Group 4.

* **Cross-validation: 5-fold stratified** within the train split. With
  ~1,200 training films and 5 folds, each fold has ~240 films.

* **Metrics — regression: R² (headline) + MAE in log units + RMSE.**
  R² for the report's headline number; MAE answers "how far off is a
  typical prediction in log-units."
  **Metrics — classification: AUC-ROC (headline) + PR-AUC.** AUC-ROC
  is threshold-free and standard; PR-AUC matters because Target 1
  (`ROI > 1`) is at ~80% positive (imbalanced) and Target 2
  (`ROI > 2`) at ~60% positive (closer to balanced).

* **Bootstrap CIs: 1,000 iterations, percentile method, seed 42.**
  Standard.

================================================================
Three things to be aware of from prior phases
================================================================

1. **Empty-text dialogue filter applies to every dialogue-derived
   feature.** Phase 2's Tier 1.3 already strips empty placeholders
   from `n_unique_characters`, but feature-engineering code that
   iterates `dialogue_units` must apply the same filter defensively
   (`name and text.strip()`). The brief's Constraints section
   reinforces this.

2. **`data_quality_flag` marks 30 films with degenerate source-XML
   scene structure** (Elvis, 12 Angry Men, Wizard of Oz, Manhattan
   Murder Mystery, etc.). For each feature you build, document how
   flagged films are handled in `FEATURE_NOTES.md`: drop, downweight,
   treat as missing, or use as-is. Per-scene features (Group 5)
   especially need to address this.

3. **`n_unique_characters` and `parse_warning_count` carry overlapping
   information** (Spearman ρ +0.39). Don't drop either preemptively;
   include both, document the redundancy in `FEATURE_NOTES.md`, and
   let Phase 4's regularization or feature selection sort them out.

================================================================
Workflow conventions (unchanged from Phase 2)
================================================================

* Code in `src/`, narrative notebooks in `notebooks/` calling those
  functions. Don't define heavy logic inside the notebook itself.
* For each notebook, keep its content in a small builder script
  (`notebooks/_build_<name>.py`) that emits the .ipynb. Easier to
  diff and regenerate. Each phase ends with a notebook.
* Use git as the code-history mechanism. No `_old.py` /
  `final_v2.py` / `_archive/` folders. Tag milestones (we have
  `phase-1-complete` and `phase-2-complete` already).
* Logging: default level is INFO. INFO at "main pipeline milestone"
  granularity (no paths, no per-item save loops). Per-item detail at
  DEBUG. The `set_log_level("WARNING")` runtime call silences INFO if
  you want it quiet during long feature-extraction runs.
* Notebook markdowns are submission-quality (formal academic
  register, no em dashes, no second-person pronouns, no process
  narrative about what was tried and reverted, no references to
  prior data sources we no longer use). The Phase 1 and Phase 2
  notebooks demonstrate the expected style.

================================================================
The brief's mandatory escalation points (read Section 6)
================================================================

You must escalate to me/the planning conversation at:

* **After Task 1 split design** — if any choice would materially
  affect Phase 5 calibration.
* **After Task 2 baseline (Phase 3a)** — if R² < 0.05 or AUC < 0.55
  across all targets, pause and escalate. Then return regardless to
  ask "next steps before Phase 3b" (the brief is explicit about this
  return).
* **Before each feature group implementation in Task 3** — write the
  proposal in `docs/proposals/phase3_<group>_proposal.md` and tell me
  to bring it to the planning conversation.
* **After Task 3 ablation** — if any group's actual lift is negative
  or near-zero.
* **Three-target divergence** — if the three targets show wildly
  different feature importance patterns.
* **End of phase** — full ablation table review before Phase 4 begins.

================================================================
Get started
================================================================

Once you've read the foundation documents, the data notes, the
phase-1 and phase-2 summaries, and the phase-3 brief, walk me through
your plan for Task 1 (the train/cal/test split). If you agree with
the 70/15/15 stratified-by-(genre, decade-bucket) recommendation,
implement it directly. If you'd do something different, explain
before implementing.

After Task 1 lands, proceed to Task 2 (Phase 3a baseline). After
Task 2, you have the brief's explicit instruction to return to me
with a Phase-3a summary before starting Phase 3b.

Stop and ask if anything is unclear.
