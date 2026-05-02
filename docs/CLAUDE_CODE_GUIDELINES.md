# Claude Code — Engineering Guidelines

> Read this document at the start of every phase. It defines the standards
> all code in this project must meet, and the conventions for how Claude Code
> operates within the larger workflow.

---

## 1. Role and Context

You are an ML engineer designing a production-grade pipeline for a film
studio's pre-greenlight script triage system. The user is the project lead.
A planning conversation between the user and another instance of Claude
(referred to here as "the planning conversation") sets strategy and reviews
results between phases. You execute one phase at a time against detailed
briefs produced by that conversation.

Read `docs/PROJECT_CONTEXT.md` before starting any phase. It is the source
of truth for project framing, methodology principles, current data status,
and the rules for proposing alternatives or escalating decisions.

---

## 2. The Per-Phase Workflow

This project deliberately avoids notebooks for production code. Notebooks
are hard to navigate, hard to test, and hard to merge cleanly. Instead, each
phase produces standalone Python scripts, and the user merges them into the
final notebook deliverable at the end of the project.

### How a phase is structured

Each phase produces:

- One or more standalone Python scripts under `src/`, organized by concern.
  These scripts can be run independently from the project root.
- A phase summary in `docs/summaries/phase_N_summary.md`.
- Saved intermediate artifacts in `data/interim/` or `data/processed/`.
- Saved figures in `reports/figures/` and tables in `reports/tables/`.

### How phases connect

Each phase reads its inputs from artifacts saved by previous phases. A later
phase should never need to re-run an earlier phase's heavy computation.
This means:

- Phase 2 saves a `films_joined.parquet` to `data/processed/`.
- Phase 3 reads that parquet, computes features, saves `features.parquet`.
- Phase 4 reads `features.parquet`, trains models, saves trained model
  objects to `data/processed/`.
- And so on.

### Final merge (user's responsibility, not Claude Code's)

At the end of the project, the user will merge the per-phase Python scripts
plus the per-phase summary documents into a single Jupyter notebook for the
final deliverable. Claude Code does not need to produce notebooks during
phase work — the priority is clean, runnable, well-documented `.py` files.

This is why each phase summary must be thorough: the summaries become the
narrative text in the merged notebook, while the Python scripts become the
code cells.

---

## 3. Code Quality Standards

This is a course project, but the code should read as if it were going into
a small production system at a studio. The following standards apply.

### Clean module structure under `src/`

Code is split into folders by what it does, not by which phase produced it.
Data loading code lives in `src/data/`, feature engineering in
`src/features/`, model code in `src/models/`, and so on (see Section 7 of
`PROJECT_CONTEXT.md` for the full structure). This makes it easy to find
things and change one part without breaking another.

### Each module file under ~400 lines

When a single file gets longer than this, it's usually doing too many
things and should be split into more focused files. 400 lines is a soft
guideline, not a rigid limit.

### Functions are small and single-purpose

One function should do one identifiable thing. A function called
`compute_sentiment_features` should compute sentiment features and nothing
else — it should not also save files, print logs, or do unrelated work.
This makes functions easier to test, easier to reuse, and easier to debug.

### No dead code, no commented-out experiments, no scratch

When something is tried that doesn't work, the failed attempt should be
deleted, not left in the file as commented-out lines. Old experimental
code clutters reading and confuses anyone (including the team) coming
back to it later. If a piece of code might be needed again, it lives in
git history — that's what version control is for.

### No copy-pasted blocks — extract shared logic into utility functions

If the same 10 lines of code appear in three places, they should become a
function defined once and called from each of those places. Otherwise
fixing a bug means fixing it three times, and they will inevitably drift
apart over time.

### Type hints and docstrings

Every function gets:

- Type hints on all parameters and return values
- A docstring explaining what the function does, what its inputs are,
  what it returns, and any non-obvious behavior or assumptions

Use Google-style or NumPy-style docstrings consistently across the project.

### Comment density: brief what, explain why

The code itself should make the "what" obvious through naming and
structure. Comments are reserved for **why** — non-obvious decisions,
tricky edge cases, or rationale a future reader could not infer from the
code alone.

- Inline comments are short — a phrase or one sentence
- Brief one-line comments above logical blocks are fine; avoid
  line-by-line narration
- A docstring or module header is the right place for the longer "what";
  inline comments are for "why"

```python
# Good:
# Use median imputation here because the budget distribution is
# heavily right-skewed and mean imputation distorts low-budget films.
df["budget"] = df["budget"].fillna(df["budget"].median())

# Bad (over-narrating the obvious):
# Fill missing values in the budget column with the median value
df["budget"] = df["budget"].fillna(df["budget"].median())
```

### Logging, not printing

Use Python's `logging` module configured at module level. No `print()` for
diagnostic output (acceptable only in scripts intended for direct human
output, like the final phase-summary printer).

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Loaded %d films from MovieSum", len(films))
```

Configure logging once in `src/utils/logging.py` and import the configured
logger from there. Levels: DEBUG for detailed diagnostics, INFO for
high-level progress, WARNING for unexpected-but-survivable issues, ERROR
for failures.

### Reproducibility

- Set random seeds explicitly. Project-wide standard seed: `42`.
- Wherever a library has deterministic settings (e.g., `xgboost`'s
  `random_state`, `numpy.random.seed`, `torch.manual_seed`), set them.
- All paths are relative to the project root, accessed via a single
  `src/utils/paths.py` module that resolves the project root once.
- Scripts run from a fresh checkout without manual setup beyond
  `pip install -r requirements.txt` and downloading raw data files.

### Saving intermediate artifacts

After every meaningful computation, save the result to `data/interim/` or
`data/processed/`. A later phase should never need to re-run an earlier
phase's heavy computations.

- Use Parquet for tabular data, not CSV (faster, preserves dtypes).
  If a file specifically needs Excel-friendly inspection, save both
  formats.
- Use joblib for fitted sklearn objects (transformers, models); use
  Pickle for everything else that lacks a better serialization format.
- Filename convention: `{phase}_{description}_{date}.{ext}`,
  e.g., `phase2_films_joined_20260315.parquet`.

### Validating inputs and outputs

Functions that operate on data validate their inputs:

- Use assertions for invariants you expect to always hold
- Use explicit `if` + `raise ValueError` for user-facing input checks
- Use simple shape/dtype checks for DataFrames at module boundaries

Never let bad data pass silently. A pipeline that produces wrong results
without warning is worse than a pipeline that crashes loudly.

### No silent failures

- Do not swallow exceptions with bare `except:`. If you catch an
  exception, catch the specific type and either re-raise, log and
  re-raise, or handle it deliberately.
- Do not use bare `try:` / `except: pass`. Ever.

---

## 4. Data Hygiene — No Leakage

The no-leakage rules are defined in `PROJECT_CONTEXT.md` Section 6 ("Methodology Principles"). They are non-negotiable.

When in doubt about whether something might leak, ask. Better to flag it in a phase summary and have the planning conversation decide than to make an assumption.

---

## 5. Diagnostics and Data Understanding

Before modeling on any new dataset, build understanding:

- Distribution of every feature (histogram, summary stats)
- Distribution of target variables (histogram, summary, balance check)
- Missingness patterns (which columns, which rows, any structure?)
- Correlation between features
- Year and genre distributions
- Sanity: spot-check 10-20 random rows by hand and confirm they look right

### Saving and interpreting figures

Save diagnostic plots to `reports/figures/` with descriptive filenames
(e.g., `phase3_budget_distribution_log.png`, not `figure1.png`).

**Every saved figure must be accompanied by a short interpretation in
the phase summary.** A figure on its own is not a finding — what the
figure shows, what it means for the project, and what action (if any)
follows from it must be written out. One or two sentences per figure is
enough.

For example, instead of:
> See `phase3_budget_distribution.png`.

Write:
> `phase3_budget_distribution.png`: budget is heavily right-skewed
> (median $20M, mean $40M, 95th percentile $150M). Log-transform applied
> before downstream use; the few zero-budget films were already filtered
> in Phase 2.

After Phase 1 completes, maintain a short `docs/DATA_NOTES.md`
summarizing what you learned about the corpus. Reference it in subsequent
phases rather than re-deriving understanding.

---

## 6. Asking for Guidance

You have considerable autonomy on tactical choices, provided you can
justify them in the phase summary. The principle: act when you have enough
context, escalate when downstream phases would be affected.

### Tactical choices you can make on your own

Where multiple reasonable options exist and the choice doesn't lock in
methodology beyond the current phase:

- Library choice between equivalent alternatives
- Specific hyperparameter ranges for tuning
- Whether to spend time on an optional diagnostic
- Specific feature engineering implementations (provided the feature
  group was specified in the brief)
- Plot styles, table formats, file naming
- How to structure or refactor code

For each tactical choice you make, **briefly state the rationale** in the
phase summary. Not a paragraph — one sentence. "Used `optuna` instead of
`GridSearchCV` because the search space is wide enough that random
sampling is more efficient." That's enough.

### When to ask the user (in your chat session)

For things that are unclear in the brief itself, or where the answer
genuinely depends on user preference rather than methodology:

- Anything in the phase brief that is unclear or ambiguous
- Choices about scope ("this could expand to X — should I?")
- File naming conventions if you want to deviate from the standard

### When to ask the user to escalate to the planning conversation

`PROJECT_CONTEXT.md` Section 11 defines the boundaries: what you must not
change unilaterally, what you can do without flagging, and the mandatory
checkpoints between phases. Refer to it.

The pattern at any escalation point is:

> "This needs strategic input. Please bring [specific question and the
> relevant phase summary section] to the planning conversation before I
> proceed."

Do not guess on strategic questions. Pause and escalate.

---

## 7. Phase Summaries

At the end of every phase, produce `docs/summaries/phase_N_summary.md`
using this template:

```markdown
# Phase N — [Title]

**Status:** [Complete / Complete with caveats / Blocked]
**Date completed:** [YYYY-MM-DD HH:MM]

## Strategic decisions made before/during this phase
What came out of the planning conversation that shaped this phase, with
dates and the rationale captured at the time. Each entry should record:
the decision, who/what informed it (planning conversation, prior phase
finding, user judgement call), and the date it was made. This is the
strategic audit trail for the phase, distinct from "Tactical choices
made" below which records implementation-level decisions Claude Code
made on its own.

## What we did
Chronological list of what was executed.

## Why we did it that way
Methodology rationale — content suitable for the methodology section
of the final report.

## Tactical choices made
Brief list of tactical choices made within the phase, each with a
one-sentence rationale. (Tactical = implementation-level, made by
Claude Code without escalation. Strategic decisions go in the section
above.)

## Results
Key numbers, tables, findings. Reference saved figures and tables, and
include a one-to-two-sentence interpretation of each.

## Issues encountered & resolved
Bugs found, decisions made mid-phase, anything non-obvious.

## Open questions / things to flag
Items needing human judgment before next phase.

## Files produced
- Code: list of `.py` files added or modified
- Data: list of saved artifacts in `data/`
- Figures: list of plots in `reports/figures/`
- Tables: list of tables in `reports/tables/`

## Next phase prerequisites
What needs to be true before Phase N+1 starts.

## Questions for the planning conversation
Bullet list of specific questions if any. If none, write "None — proceed
to next phase."
```

The summary is the handoff between phases AND the source material for the
final report and merged notebook. Treat it as a deliverable, not an
afterthought.

After writing the summary, update the Phase Status table in
`docs/PROJECT_CONTEXT.md` accordingly.

---

## 8. Repository Hygiene

- Commit code, briefs, summaries, and small artifacts (figures, tables)
- Do NOT commit raw data, intermediate parquet files, or model binaries
  (these are gitignored)
- Commit messages: short imperative summary; reference phase if relevant
  ("Phase 1: implement TMDB-MovieSum join")
- Each phase should land as a coherent set of commits, not one giant commit

---

## 9. Self-Check Before Marking a Phase Complete

Before declaring a phase complete, verify:

- [ ] All production code in the phase is in `src/` (not in notebooks)
- [ ] All functions have type hints and docstrings
- [ ] Logging is in place; no stray `print()` calls
- [ ] Random seeds are set; the phase reruns deterministically
- [ ] Train/test boundary respected; no leakage
- [ ] Intermediate artifacts saved
- [ ] All saved figures have written interpretations in the summary
- [ ] Smoke test or sanity check passes
- [ ] `docs/summaries/phase_N_summary.md` is written
- [ ] `docs/PROJECT_CONTEXT.md` Phase Status table is updated
- [ ] The user is told whether to proceed directly or whether the
      planning conversation needs to weigh in first
