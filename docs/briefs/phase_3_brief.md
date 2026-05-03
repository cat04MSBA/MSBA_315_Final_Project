# Phase 3 Brief — Feature Extraction with Incremental Baselining

> **Read first:** `docs/PROJECT_CONTEXT.md`, `docs/CLAUDE_CODE_GUIDELINES.md`,
> `docs/DATA_NOTES.md`, and `docs/summaries/phase_2_summary.md`.
>
> Phase 2 built the data pipeline. Phase 3 builds the feature matrix that
> Phase 4 will model on. Methodologically, Phase 3 is structured around
> incremental ablation: establish a baseline on what we already have, then
> add feature groups one at a time and measure the lift each contributes.
> This produces a defensible feature set with a clean ablation table for
> the report.

---

## 1. Goal of This Phase

Two deliverables, both required:

1. **A feature matrix** saved as `data/processed/features.parquet`,
   keyed by `imdb_id`, containing all engineered features that earned
   their place in the ablation, plus the train/calibration/test split
   assignment per film, plus the prediction targets.
2. **An ablation table** (saved as `reports/tables/phase3_ablation.csv`)
   documenting baseline performance and the lift each feature group
   contributed. This becomes Phase 3's narrative and feeds Phase 4 model
   selection.

---

## 2. Strategic Decisions From the Planning Conversation

These are confirmed; implement directly.

### Targets — train all three in parallel throughout Phase 3

- **Regression target:** `log(ROI)` or other you may see fit (let me know in the chat) where `ROI = revenue / budget`
- **Classification target 1:** `ROI > 1` (gross profitable; ~80% of corpus)
- **Classification target 2:** `ROI > 2` (industry rule-of-thumb for net-profitable; closer to balanced)

All three are saved as columns in the feature matrix. All three are
modeled in every ablation step. The decision on which becomes the
primary outcome variable is deferred to the end of Phase 4 once
comparative results are in.

### Methodology — incremental ablation

Phase 3 has two sub-phases:

- **Phase 3a (baseline):** train baseline models on existing
  master-Parquet features only — no new feature engineering yet. Document
  the floor.
- **Phase 3b (incremental ablation):** add one feature group at a time,
  retrain baselines, log the lift each group contributes.

**Sub-phase boundary is mandatory.** Do not start Phase 3b until Phase 3a
is complete and the baseline numbers are documented. The Phase 3b lift
calculations reference Phase 3a baseline numbers.

### Calibration set required

Phase 5 (calibrated uncertainty) requires a held-out calibration set
separate from training and test. Carve it out as part of Phase 3's
single train/cal/test split. The split happens once, before any feature
fitting that depends on data distribution.

### No-leakage discipline

Per `PROJECT_CONTEXT.md` Section 6:

- The split happens before any feature engineering that involves data
  distribution.
- LDA topic models, sentence-embedder fine-tuning (if any), feature
  scalers, PCA, and any imputation parameters are fit on **training
  data only** and applied to the calibration and test sets.
- Pre-trained embeddings used as-is do not leak.
- Hyperparameter tuning happens within the training set (cross-validation
  within the train split). The test set is not touched until Phase 8.

### Pre-registration (epistemic discipline)

Before implementing each feature group in Phase 3b, write one
paragraph predicting:
- Which target(s) the group should help most
- Rough estimate of expected lift on each target
- The mechanism by which the features should carry signal

After implementation, log predicted vs. actual lift in the ablation
table. This is a small bookkeeping cost for a real epistemic benefit:
forces honest tracking and surfaces overfitting to single CV folds.

---

## 3. Tasks

### Task 1 — Train / calibration / test split

- Carve a single split from `films_joined.parquet`. Save the split
  assignment as a column in the feature matrix and as a separate
  reference file under `data/processed/split_assignments.parquet`
  (keyed by `imdb_id`).
- The calibration set must exist for Phase 5. Sizing is tactical —
  pick a reasonable balance. Justify in the phase summary.
- Stratification, time-aware vs. random, fold structure for CV — all
  tactical choices. Justify in the summary. If any choice would
  meaningfully affect downstream phases, ask in chat first.
- Set random seed (project standard `42`) and ensure reproducibility.

### Task 2 — Phase 3a: Baseline models on existing features

- Identify which features in `films_joined.parquet` are usable as
  baseline inputs. Likely candidates: the parser-derived structural
  metrics, `release_year_parsed`, `primary_genre_bucketed` (one-hot),
  `log_runtime`. Use your judgment on what to include and exclude;
  document the choice.
- Train a simple, interpretable baseline model on each of the three
  targets. The choice of model family is yours — pick something
  appropriate for the "constant comparator" role (linear / regularized
  linear is the obvious default but use your judgment).
- Cross-validate within the training set. Report performance metrics
  appropriate to each target with bootstrap confidence intervals.
- Train an additional **sanity-check baseline** with `log_budget` added
  to the feature set, separately reported. This shows what budget
  knowledge would buy you. NOT the headline; the dialogue-only baseline
  is what the deployed system would use. Document both.
- Save the baseline numbers to `reports/tables/phase3a_baseline.csv`.
  These become the floor that every Phase 3b addition compares against.

**Decision point at end of Task 2:** if the baseline R² (regression) is
below 0.05 or AUC (classification) is below 0.55 on cross-validation
across all three targets, pause and escalate to the planning
conversation. This would indicate the existing features don't carry
signal at all, which would be strategically important to know before
investing in Phase 3b.

Before moving to phase 3b, revert back to the planning conversation with a summary of 3a and ask on next steps.

### Task 3 — Phase 3b: Incremental feature engineering with ablation

Five feature groups, added one at a time. For each group:

**Step 3.a — Propose features (planning conversation review)**

Before implementing the group, write a short proposal containing:
- Which specific features you propose for this group, with rationale
  for each
- Expected lift on each of the three targets (the pre-registration
  prediction)
- Any feasibility concerns specific to the corpus

Save the proposal as `docs/proposals/phase3_<group>_proposal.md` and
tell the user explicitly:

> "Proposal for the [group] feature group is ready. Please bring it
> to the planning conversation before I implement."

Do not implement until the user returns with planning-conversation
input.

**Step 3.b — Implement**

Once the proposal is approved (possibly with edits from the planning
conversation), implement the features. Save the feature matrix
snapshot as `data/processed/features_<group>.parquet`.

**Step 3.c — Re-run baselines**

Retrain the same baseline models on the expanded feature set. Compute
lift over the Phase 3a numbers. Append a row to
`reports/tables/phase3_ablation.csv` with: feature group, predicted
lift (from pre-registration), actual lift on each target, total
features added, total features in matrix.

**Five feature groups, in suggested order:**

1. **Lexical features** — vocabulary diversity, readability,
   syntactic complexity, length statistics
2. **Sentiment features** — sentiment aggregates, emotional arc
   patterns, sentiment dynamics
3. **Topic features** — topic distributions over screenplay text
4. **Embedding features** — distributed text representations,
   appropriately pooled to film level given screenplay length exceeds
   transformer context windows
5. **Character network features** — graph metrics derived from
   character interaction structure

The planning conversation will provide a literature reference doc as
a second-lens check on each proposal. It is not a constraint on what
you propose — it's used after the fact to validate or extend.

### Task 4 — Final feature matrix and documentation

- Save the final feature matrix as `data/processed/features.parquet`
  with all engineered features that earned their place plus the
  three target columns plus the split assignment column.
- Write `docs/FEATURE_NOTES.md` (modeled on `docs/DATA_NOTES.md`)
  containing: full column glossary for the feature matrix, which
  features came from which group, the ablation summary, any features
  dropped after measurement and why.

### Task 5 — Phase summary

Use the standard template from `CLAUDE_CODE_GUIDELINES.md` Section 7.
Save to `docs/summaries/phase_3_summary.md`. Must include:

- Strategic decisions section (the targets, ablation-first methodology,
  any new decisions surfaced during the phase)
- The full ablation table
- Pre-registration vs. actual lift comparison for each group
- Recommendation on which feature groups should carry into Phase 4
  (and which to drop)
- Open questions for the planning conversation about target choice
  (regression vs. classification primary outcome)

---

## 4. Constraints From Prior Phases

These are non-negotiable:

- **Empty-text dialogue filter:** for any feature derived from
  `dialogue_units`, filter to entries with non-empty, non-whitespace
  dialogue text before computing. Phase 2's Tier 1.3 fix ensures this
  for the parser's own metrics; downstream feature engineering must
  apply the same filter defensively.
- **Data quality flag handling:** the `data_quality_flag` derived in
  Phase 2 cleanup marks films with degenerate source XML (e.g.,
  `Elvis_2022`). For each feature, document how flagged films are
  handled (drop, downweight, treat as missing, or use as-is).
- **Prefer `dialogue_to_total_text_ratio`** over `dialogue_to_action_ratio`.
  The latter is degenerate (~0.99 for most films) because
  `<stage_direction>` in MovieSum is usually slugline only.
- **Redundancy with `parse_warning_count`:** if both
  `n_unique_characters` and `parse_warning_count` are used as features
  (Spearman ρ +0.39), document the redundancy in `FEATURE_NOTES.md`.
  Phase 4 may want to use only one.
- **Pre-1980s decades:** if any feature uses decade as input, bucket
  pre-1980s into a single "older films" stratum to avoid noise from
  thin cells.

---

## 5. Definition of Done

- [ ] `data/processed/features.parquet` exists, validated, contains
      every engineered feature plus three targets plus split column
- [ ] `data/processed/split_assignments.parquet` exists with one row
      per film
- [ ] `reports/tables/phase3a_baseline.csv` documents Phase 3a baseline
      numbers (with and without `log_budget`)
- [ ] `reports/tables/phase3_ablation.csv` documents the full
      incremental ablation
- [ ] `docs/FEATURE_NOTES.md` exists with column glossary and ablation
      summary
- [ ] Each feature group has a proposal in `docs/proposals/` with
      pre-registration predictions, and the planning conversation
      reviewed each before implementation
- [ ] Phase summary written using the standard template
- [ ] User informed that Phase 3 is complete and the planning
      conversation should review the ablation table before Phase 4
      starts

---

## 6. When to Ask vs. When to Act

Per `CLAUDE_CODE_GUIDELINES.md` Section 6.

**Act on your own** — most tactical choices in this phase are yours:
- Choice of baseline model family
- Train/cal/test split sizing and stratification
- Cross-validation fold structure
- Specific metrics to report (regression: pick from R², MAE, RMSE;
  classification: pick from AUC-ROC, accuracy, F1, PR-AUC)
- Bootstrap iteration counts
- Library choices (gensim vs. sklearn for LDA, sentence-transformers
  variant, etc.)
- Feature engineering implementation details within an approved
  proposal
- Plot styles, file naming, refactoring patterns

Justify each non-obvious tactical choice briefly in the phase summary.

**Ask the user in chat** if:
- The brief is unclear or ambiguous on something specific
- A choice depends on user preference rather than methodology (e.g.,
  computational budget for embedding extraction)
- An environment issue blocks progress
- The data behaves in a way that contradicts what `DATA_NOTES.md`
  documented

**Tell the user to escalate to the planning conversation** for the
mandatory escalation points:
- After Task 1 split design — if the split decision will materially
  affect Phase 5 calibration, surface for review
- After Task 2 baseline — if R² < 0.05 or AUC < 0.55 across all
  targets, pause and escalate
- Before each feature group implementation in Task 3 — proposal
  step requires planning conversation review
- After Task 3 ablation — if any group's actual lift is negative or
  near-zero, surface for discussion (keep, drop, or investigate)
- If the three targets show wildly different feature importance
  patterns — surface for discussion of primary outcome choice
- End of phase — full ablation table review before Phase 4 begins

---

## 7. Anti-Goals

- Do not start Phase 4 modeling work (XGBoost, LightGBM, hyperparameter
  tuning, ensemble exploration) — that is Phase 4's job. Phase 3's
  models are intentionally simple "constant comparators."
- Do not touch the test set. Cross-validation happens within the train
  split.
- Do not implement calibration (conformal prediction). That is Phase 5.
- Do not skip the proposal step for any feature group. Even if you're
  confident about the features, the planning conversation needs the
  proposal in writing for the literature-reference comparison.
- Do not skip pre-registration. The point is to track honest expectation
  vs. reality.
- Do not over-engineer for production hardening (parse-quality scores,
  unknown-tag capture, etc.). This is a course project; ship the
  features, document them, move to Phase 4.

---

## 8. Self-Check Before Marking Phase 3 Complete

- [ ] All production code in `src/` (likely `src/features/` and
      `src/models/baseline/`)
- [ ] All functions have type hints and docstrings
- [ ] Logging in place; no stray `print()` calls
- [ ] Random seeds set; the phase reruns deterministically
- [ ] Train/test boundary respected throughout — no test-set touches,
      no leakage in feature fitting
- [ ] Calibration set carved out and preserved for Phase 5
- [ ] Empty-text dialogue filter applied at every dialogue-derived
      feature
- [ ] Data quality flag handling documented per feature
- [ ] All saved figures have written interpretations in the summary
- [ ] All saved tables documented (what each column means)
- [ ] Phase summary written using standard template
- [ ] `docs/FEATURE_NOTES.md` written
- [ ] User informed Phase 3 complete; planning conversation should
      review ablation table before Phase 4 begins
