# Phase 2 Brief — Data Pipeline

> **Read first:** `docs/PROJECT_CONTEXT.txt` (project framing, decisions
> log, methodology principles), `docs/CLAUDE_CODE_GUIDELINES.txt`
> (engineering standards, escalation rules, summary format), and
> `docs/summaries/phase_1_summary.txt` (Phase 1 results, EDA
> observations, deferred decisions).
>
> Phase 1 was exploratory. Phase 2 is production. The deliverable is a
> clean, reproducible data pipeline that produces the master working
> dataset that every downstream phase consumes.

---

## 1. Goal of This Phase

Build the production-quality data pipeline that turns raw downloaded
data into the canonical processed artifact:
`data/processed/films_joined.parquet`. This artifact is the single
input that Phase 3 (feature extraction) and all later phases read.
Subsequent phases must not re-implement loading, joining, dedup, or
filtering — they read this file.

The pipeline must be reproducible from a fresh checkout with one
script invocation, given the raw data files in `data/raw/`.

---

## 2. Locked-in Decisions From the Planning Conversation

These decisions are confirmed; implement directly without re-asking.

### Strategic
- **Survivorship bias treatment:** Path (a) confirmed. Keep the corpus
  as-is, document the ~80% gross-profitable rate prominently in the
  report's limitations section, run sensitivity analysis across
  multiple cost matrices in Phase 6 to test robustness. Do not
  reweight at training time. If Phase 4 reveals the model is wildly
  overconfident on flops, we can revisit as a Phase 8 robustness
  experiment, but Phase 2 does not pre-build for reweighting.
- **Outcome variables:** Both rating and box-office outcomes are
  carried forward through Phase 2 (no decision needed yet — Phase 4
  trains both and compares).

### Corpus filters (apply at corpus-build time)
- **Pre-1995 cutoff.** Drop films with parsed release year < 1995.
  Rationale to record in code comments and report: 30 films across
  60 years is statistically meaningless for any temporal
  generalization claim; cleaner CV folds; matches the deployment
  framing of contemporary script triage.
- **Drop the 2 unmatched MovieSum films** (no entry in ratings
  dataset).
- **Dedup ratings dataset on `imdb_id`,** keeping the row with the
  higher `vote_count`. (Phase 1 already implemented `dedupe_ratings()`;
  Phase 2 calls it.)
- **MovieSum dedup per the user's review file.** Read
  `reports/tables/phase1_moviesum_duplicates_review.csv` and apply
  the user's `keep`/`drop` decisions. If any row in the review file is
  blank or contradictory, escalate (don't fall back to the longest-
  script heuristic silently).
- **Clip `release_year_parsed` to 1900–2025** to handle future-year
  noise from the ratings dataset.

### Column conventions
- **`effective_rating`** is a single derived column: use `IMDB_Rating`
  when present and non-null, else fall back to `vote_average`. Save
  both source columns alongside `effective_rating` for traceability.
- **0-is-missing for budget, revenue, runtime.** Treat 0 as a missing
  value sentinel for these monetary/numeric fields throughout the
  pipeline. Filtering to the four-signal working corpus uses `> 0`
  conditions, not non-null conditions, for budget and revenue.
- **`log_budget` and `log_revenue`** are derived columns, computed as
  `log1p(budget)` and `log1p(revenue)`. Save both raw and
  log-transformed columns. The log transform is a config knob; the
  default is `log1p`.

### Genre handling
- **Bucket genres with fewer than 30 films in the working corpus into
  `Other`.** Save the original `genres_parsed` (multi-label) alongside
  `genres_bucketed` (multi-label, with thin genres replaced by
  `Other`). The bucketing threshold (30) is a config knob.
- **Primary genre column.** Derive `primary_genre` and
  `primary_genre_bucketed` columns by taking the first listed genre
  per film, for use as a stratification variable in Phase 4 CV.

### Deferred to later phases (Phase 2 does NOT do these)
- **Imputation:** deferred to Phase 3 (must happen after train/cal/test
  split to avoid leakage).
- **Scaling:** deferred to Phase 4 (model-specific; tree-based models
  don't need it, linear models do; applied inside sklearn Pipelines).
- **Outlier capping:** deferred. `log1p` handles right-skew adequately;
  if Phase 4 surfaces specific outlier issues, address then.
- **Train/cal/test split:** deferred to Phase 3.

---

## 3. Tasks

This phase is production work, not exploration. All code must meet the
standards in `CLAUDE_CODE_GUIDELINES.txt` Section 3 (type hints,
docstrings, logging, no leakage, validation, etc.).

### Task 1: Refactor Phase 1 loaders into production form

Phase 1's loaders (`load_ratings.py`, `load_moviesum.py`) work but
were exploratory. Refactor them as needed to meet production standards:

- Type hints and docstrings on every function
- Input validation (file exists, expected columns present, expected
  shape)
- No silent failures; specific exception types where appropriate
- Logging at INFO for high-level progress, DEBUG for detail

If the existing loaders already meet these standards (they likely do),
note that and move on. Don't refactor for refactoring's sake.

### Task 2: Build the screenplay parser

Phase 1 confirmed MovieSum's XML structure but did NOT parse it.
Phase 2 builds the parser that extracts structured content from each
screenplay. Save under `src/data/parse_screenplay.py`.

The parser should produce, for each screenplay, a structured object
(dataclass or pydantic model) containing:

- `imdb_id` (string)
- `scenes`: ordered list of scenes, each with:
  - `scene_number` (int, 1-indexed)
  - `stage_direction` (string, may be empty)
  - `scene_description` (string, may be empty)
  - `dialogue_units`: ordered list of (character_name, dialogue_text) tuples

Edge cases to handle gracefully (don't crash, log a warning, record in
diagnostics):
- Scenes with no dialogue
- Dialogue with no character attribution (uncommon but possible)
- Scenes with empty stage directions and empty descriptions
- Malformed XML in individual screenplays

The parser must be deterministic (same input → same output).

After parsing, add to the master Parquet:
- `n_scenes`: integer count of scenes
- `n_unique_characters`: integer count of distinct character names
- `n_dialogue_lines`: total count of dialogue units across all scenes
- `total_dialogue_chars`: sum of dialogue text lengths
- `total_stage_direction_chars`: sum of stage direction lengths
- `dialogue_to_action_ratio`: `total_dialogue_chars /
  (total_dialogue_chars + total_stage_direction_chars)`, with a sane
  default for the edge case of zero total

The full structured screenplay (the dataclass) should ALSO be
serialized — recommend a separate file
`data/processed/screenplays_parsed.pkl` or `.parquet` keyed by
`imdb_id`, since the structured object is too rich to denormalize into
the master film table. Phase 3 reads both: the master table for
metadata and outcomes, the structured screenplays for feature
extraction.

### Task 3: Build the master pipeline orchestrator

Write `src/data/build_corpus.py` — a single entry-point script that:

1. Loads the ratings dataset (calls Task 1 loader)
2. Dedupes the ratings dataset
3. Loads MovieSum (calls Task 1 loader)
4. Applies the MovieSum dedup policy from the user's review CSV
5. Joins on `imdb_id`
6. Applies the corpus filters (pre-1995 cutoff, drop unmatched, etc.)
7. Computes derived columns (`effective_rating`, `log_budget`,
   `log_revenue`, `primary_genre`, `genres_bucketed`,
   `primary_genre_bucketed`)
8. Parses every screenplay (calls Task 2 parser), adds the
   screenplay-derived columns to the master table
9. Validates the output (expected shape, no nulls in critical columns,
   year range correct)
10. Saves `data/processed/films_joined.parquet` and
    `data/processed/screenplays_parsed.{pkl|parquet}`

This script must be idempotent: running it twice produces the same
output. It must be runnable as `python -m src.data.build_corpus` from
the project root.

### Task 4: Validate and profile the processed corpus

Write `src/data/validate_processed_corpus.py` (or similar). This
script:

- Loads `data/processed/films_joined.parquet`
- Verifies expected film count (target: ~1,660 after pre-1995 cutoff
  applied to the 1,713 working corpus; flag if off by more than 20)
- Verifies year range (1995–2023)
- Verifies all rows have non-null `effective_rating`, `budget > 0`,
  `revenue > 0`
- Verifies the screenplay-derived columns are non-null and in expected
  ranges (e.g., `n_scenes > 0`, `dialogue_to_action_ratio` between 0
  and 1)
- Re-profiles the corpus: counts, year/genre distributions, missingness
  patterns, summary statistics. Save as updated CSV tables under
  `reports/tables/phase2_*.csv`.
- Updates the diagnostic plots from Phase 1 with the post-filter corpus
  (year, genre, budget/revenue distributions, rating, screenplay
  length, the new screenplay-structural features). Save to
  `reports/figures/phase2_*.png`.

Each saved figure needs a one-to-two-sentence interpretation in the
phase summary, per `CLAUDE_CODE_GUIDELINES.txt` Section 5.

### Task 5: Write `docs/DATA_NOTES.md`

A short standing reference document (~1-2 pages) describing the
processed corpus characteristics. Future phases reference this rather
than re-deriving understanding. Suggested sections:

- Corpus size and composition (year range, genre distribution,
  budget/revenue/rating ranges)
- Known biases (survivorship bias, year-density skew, genre skew)
- Column glossary (every column in `films_joined.parquet`, what it is,
  derived from what)
- Edge cases and how the pipeline handles them (0-as-missing, future
  years, MovieSum duplicates)
- File locations (`data/processed/`)

### Task 6: Update foundation documents

- **`PROJECT_CONTEXT.txt` Section 8 (Decisions Log):** Append a dated
  entry documenting the dataset swap from TMDB 5000 to the IMDb-TMDB
  1M dataset. Use the content of the planning-conversation handoff
  note as the source. Include the comparison numbers (1,019 → 1,713
  films, year coverage 2016 → 2023, 86% → 80% gross-profitable). The
  audit trail belongs in the Decisions Log even though the rest of the
  foundation docs treat the new dataset as if it had always been the
  source.
- **`PROJECT_CONTEXT.txt` Section 5 (Data Summary):** Update with
  post-filter numbers (final corpus size, year range, genre counts).
- **`PROJECT_CONTEXT.txt` Section 9 (Phase Status):** Mark Phase 2
  complete, link to the summary.
- **`CLAUDE_CODE_GUIDELINES.txt` Section 7 (phase summary template):**
  Add a new section near the top of the template called
  "Strategic decisions made before/during this phase." This captures
  what came out of the planning conversation that affected the phase,
  with dates. Going forward, every phase summary uses this updated
  template.

### Task 7: Write the phase summary

Use the updated template in `CLAUDE_CODE_GUIDELINES.txt` Section 7
(after Task 6 update). Save to `docs/summaries/phase_2_summary.txt`.

The summary must include:
- Strategic decisions section (Path-a survivorship bias treatment,
  pre-1995 cutoff, etc., with dates)
- Final corpus profile after Phase 2 filters
- Validation results from Task 4
- All saved figures with written interpretations
- Open questions for the planning conversation (if any)

---

## 4. Definition of Done

- [ ] `data/processed/films_joined.parquet` exists, validates cleanly,
      contains the expected ~1,660 films
- [ ] `data/processed/screenplays_parsed.{pkl|parquet}` exists, contains
      a structured screenplay for every film in the master table
- [ ] `src/data/build_corpus.py` is idempotent and runs cleanly from
      the project root
- [ ] `src/data/parse_screenplay.py` parses MovieSum XML deterministically
      and handles documented edge cases
- [ ] `docs/DATA_NOTES.md` exists and documents corpus + columns
- [ ] `PROJECT_CONTEXT.txt` Sections 5, 8, and 9 updated
- [ ] `CLAUDE_CODE_GUIDELINES.txt` Section 7 updated with the new
      strategic-decisions section
- [ ] Phase 2 summary written using the updated template
- [ ] All figures have written interpretations in the summary
- [ ] All code passes the self-check in `CLAUDE_CODE_GUIDELINES.txt`
      Section 9 (type hints, docstrings, logging, leakage check, etc.)

---

## 5. When to Ask vs. When to Act

Per `CLAUDE_CODE_GUIDELINES.txt` Section 6.

**Act on your own** for tactical implementation choices: parser
implementation details, file format choices (pickle vs parquet for the
structured screenplays), specific dataclass design, plot styling,
logging messages, refactoring patterns. Briefly state rationale for
non-obvious choices in the phase summary.

**Ask the user in chat** if:
- The MovieSum review CSV has missing or contradictory entries
- The screenplay XML structure differs from what's documented and the
  parser can't determine the right interpretation
- The processed corpus size is wildly off from ~1,660 (e.g., below
  1,400 or above 1,800) — investigate why before saving
- An environment issue blocks progress (missing library, version
  conflict)

**Tell the user to escalate to the planning conversation** if:
- A new strategic question surfaces during execution that wasn't
  anticipated
- A discovery in the data fundamentally changes what the project can
  claim
- You believe a different data structure or pipeline shape would serve
  the project significantly better than what this brief specifies

This phase has no mandatory checkpoint at the end (Phase 2 is
infrastructure; the next checkpoint is at end of Phase 4).

---

## 6. Anti-Goals for This Phase

- Do not extract dialogue features. That is Phase 3.
- Do not split into train/cal/test. That is Phase 3.
- Do not impute missing values. That is Phase 3, post-split.
- Do not scale or normalize features. That is Phase 4, model-specific.
- Do not train any models, baseline or otherwise. That is Phase 4.
- Do not implement reweighting for survivorship bias. The decision is
  Path (a), no reweighting.
- Do not chase the genre long tail with imputation or augmentation;
  bucketing into `Other` is the chosen approach.

---

## 7. Self-Check Before Marking Phase 2 Complete

From `CLAUDE_CODE_GUIDELINES.txt` Section 9, with phase-specific
additions:

- [ ] All production code in `src/`, organized by concern
- [ ] All functions have type hints and docstrings
- [ ] Logging in place; no stray `print()` calls
- [ ] Random seeds set where applicable
- [ ] No leakage — no train/test split happens here, but verify no
      operation conditioned on outcomes
- [ ] `films_joined.parquet` saved and validated
- [ ] `screenplays_parsed` artifact saved
- [ ] `build_corpus.py` is idempotent
- [ ] Parser handles documented edge cases without crashing
- [ ] All saved figures have written interpretations in the summary
- [ ] Smoke test: load the processed parquet from a fresh Python
      session, confirm it loads cleanly and has expected shape
- [ ] `docs/summaries/phase_2_summary.txt` written using the updated
      template (with Strategic decisions section)
- [ ] `docs/DATA_NOTES.md` written
- [ ] `PROJECT_CONTEXT.txt` updated (Sections 5, 8, 9)
- [ ] `CLAUDE_CODE_GUIDELINES.txt` Section 7 template updated
- [ ] User informed that Phase 2 is complete and Phase 3 can proceed
      directly (no mandatory checkpoint)
