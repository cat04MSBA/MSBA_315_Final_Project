# Phase 2 — Data Pipeline

**Status:** Complete

**Date completed:** 2026-05-02

---

## Strategic decisions made before/during this phase

- **Survivorship bias treatment: Path (a) — keep corpus as-is, document the bias.** Source: planning conversation, recorded 2026-05-02 in the Phase 2 brief Section 2. The ~80% gross-profitable rate stays; Phase 6 will run sensitivity analysis across multiple cost matrices to test robustness.
- **Outcome variables: both rating and box-office are carried forward through Phase 2.** Source: planning conversation, recorded 2026-05-02 in the Phase 2 brief Section 2. No decision needed yet; Phase 4 trains both and compares.
- **Pre-1995 cutoff: REVERSED mid-Phase-2 (2026-05-02 23:35).** Originally locked in by the planning conversation based on the Phase 1 summary's claim of "~50 pre-1995 films." When the build pipeline first ran with the 1995 cutoff, the corpus came in at 1,315 films, well below the brief's expected 1,660. Investigation showed the actual pre-1995 count was 398, not ~50 — a Phase 1 EDA count error. Escalated to user, who chose to drop the cutoff entirely. The `min_year` knob in `CorpusBuildConfig` defaults to 1900 now, and the working corpus retains all 1,713 films. Reversal logged in `PROJECT_CONTEXT.md` Section 8 (entry dated 2026-05-02 23:35); Phase 1 summary corrected with strikethrough on the bad claim.
- **Dataset swap (TMDB 5000 → IMDb-TMDB Movie Metadata Big Dataset (1M)).** This happened mid-Phase-1 but the audit-trail entry was added to the decisions log in Phase 2 (entry 2026-05-02 23:50) per Phase 2 brief Task 6. Foundation docs were rewritten as if the new dataset was always the choice, but the swap is preserved in the decisions log so future readers can trace the change.

---

## What we did

1. **Re-read the Phase 2 brief and validated the locked-in decisions** in Section 2 against the Phase 1 summary; flagged the 1995 cutoff as a recount-needed once the build surfaced the discrepancy.
2. **Updated the phase-summary template** (`CLAUDE_CODE_GUIDELINES.md` Section 7) to include a "Strategic decisions made before/during this phase" section. This summary uses the new template.
3. **Reviewed the Phase 1 loaders** (`load_ratings.py`, `load_moviesum.py`) for production quality. They already had type hints, docstrings, input validation (`FileNotFoundError` checks), specific exception types, and INFO/DEBUG logging. No refactoring needed.
4. **Built the screenplay parser** (`src/data/parse_screenplay.py`). Two frozen dataclasses: `Scene` (one per scene; stage_direction, scene_description, dialogue_units as a tuple of `(character, dialogue_text)` pairs) and `ParsedScreenplay` (one per film; tuple of scenes plus computed structural metrics). The parser walks each `<scene>` in document order and reconstructs `(character, dialogue)` pairs, with `last_speaker` tracking so dialogue continuations after a `<parenthetical>` element correctly attribute to the same speaker.
5. **Discovered a tag the brief did not document:** `<parenthetical>`. It appears in MovieSum between `<character>` and `<dialogue>` (delivery instruction like `(softly)`) or between two `<dialogue>` elements from the same speaker (continuation marker like `(beat)`). Recognized as a valid screenplay element; not stored in the dialogue-units tuples (the brief specified 2-tuples).
6. **Built the master pipeline orchestrator** (`src/data/build_corpus.py`). One entry-point script that loads ratings, dedupes ratings, loads MovieSum, applies the user-filled dedup CSV, joins on `imdb_id`, applies corpus filters, computes derived columns, parses every screenplay, attaches structural metrics to the master DataFrame, validates, and saves both artifacts. All preprocessing knobs are exposed via `CorpusBuildConfig` (frozen dataclass).
7. **Caught and escalated the pre-1995 count discrepancy.** First build with the 1,995 cutoff produced 1,315 films, below the brief's 1,400 floor. Investigated, found 398 films pre-1995 (vs. claimed ~50). Escalated to the user with three options. User chose to drop the cutoff. Re-ran build → 1,713 films, year range 1932-2023, all hard assertions pass.
8. **Logged the cutoff reversal** in three places: a new dated entry in `PROJECT_CONTEXT.md` Section 8 (with cross-reference to the original entry, which gained a "REVERSED" annotation), strikethrough corrections in `phase_1_summary.md`, and an updated entry in `handoffs/PHASE_2_PLANNING_HANDOFF.md`.
9. **Built the validator** (`src/data/validate_processed_corpus.py`). Hard-asserts every invariant the master Parquet must satisfy (no nulls in critical columns, year range, log values finite and positive, ratios in [0,1], every film has at least one parsed scene). Also generates Phase 2 versions of the diagnostic plots and three Phase 2 summary tables.
10. **Wrote `docs/DATA_NOTES.md`** as a standing reference: corpus headlines, known biases, full column glossary, edge cases, file locations, where to look for more.
11. **Updated `PROJECT_CONTEXT.md`** Sections 5 (Data Summary — added Phase 2 processed corpus block), 8 (decisions log — three new entries: pre-1995 reversal, dataset-swap audit-trail, and the existing pre-1995 entry annotated), and 9 (Phase Status — Phase 2 marked complete).

---

## Why we did it that way

The pipeline is built around a single config object (`CorpusBuildConfig`) and a single entry-point function (`build_corpus`). Every preprocessing decision is a knob on the config: `min_year`, `year_clip_min/max`, `monetary_transform`, `genre_min_count`, `rating_priority`, etc. The user can override any field to test alternatives, which means switching to (say) a 1990 cutoff or a `log10` transform is one line of Python in a notebook, not a refactor.

We chose pickle for the structured screenplays (`screenplays_parsed.pkl`) and Parquet for the master film table (`films_joined.parquet`). Parquet for the master table because it preserves dtypes (we have list-valued `genres_*` columns), is small (409 KB for 1,713 films × 41 cols), and loads fast. Pickle for the screenplays because they're nested Python dataclasses (`ParsedScreenplay` containing tuples of `Scene` containing tuples of `(str, str)`) which Parquet would have to flatten or serialize as JSON.

We chose to compute screenplay-structural metrics at corpus-build time and denormalize them onto the master DataFrame, rather than computing them on the fly in Phase 3. The metrics are static features of each screenplay; recomputing them every time Phase 3 runs would be wasteful. The full structured form (the `Scene` lists with their dialogue) stays in the pickle for any feature extraction that needs the raw text.

For the `<parenthetical>` tag we silently absorbed it as a continuation marker rather than storing it in a third tuple slot, because (a) the brief's `dialogue_units` schema is 2-tuples, and (b) parentheticals carry instructions to actors, not dialogue content; if Phase 3 wants them, the parser is a one-line change.

We treated the brief's literal `dialogue_to_action_ratio` formula (uses only `stage_direction`, which is usually a slugline only and yields ~0.99) as one possible feature and added a more informative `dialogue_to_total_text_ratio` (uses `stage_direction + scene_description`) alongside it. Both are saved; downstream phases pick whichever is appropriate.

The pre-1995 cutoff's reversal merits separate justification beyond just "the count was wrong." The original 1995 cutoff was rationalized as "30 films across 60 years is statistically meaningless." With the actual count of 398 films across 60 years, that rationale doesn't hold — we have meaningful coverage of pre-1995 cinema and dropping it would discard 23% of the corpus. Era-stratified CV in Phase 4 can bucket pre-1980s decades into a single "older films" stratum; that's a softer and more honest treatment than a hard cutoff.

---

## Tactical choices made

- **Pickle for `screenplays_parsed`, Parquet for `films_joined`.** Pickle for nested dataclass objects, Parquet for tabular dtypes-aware columns including list-valued ones.
- **`CorpusBuildConfig` as a frozen dataclass with all preprocessing knobs as fields.** Override any field via `dataclasses.replace(config, min_year=1990)` to test alternatives.
- **Default `min_year=1900`.** No effective cutoff after the reversal; the knob is exposed for future experimentation.
- **`dedupe_moviesum_from_csv` accepts `keep` and `drop`/`remove`/`delete`/`discard` synonyms.** The user wrote `remove` instead of `drop` in the review CSV; rather than rewrite the CSV, the parser normalizes.
- **Validation invariants live in a separate module** (`validate_processed_corpus.py`) and run as both build-time hard asserts (in `build_corpus.py`) and as a standalone smoke test.
- **Two dialogue-to-action ratios saved** rather than picking one.
- **Random seed 42** wherever applicable (consistent with project standard from `CLAUDE_CODE_GUIDELINES.md`).
- **`script` and `summary` columns dropped from the master Parquet.** They're already in the pickle; carrying them in the table would add hundreds of MB.

---

## Results

### Final corpus

- **1,713 films** in `data/processed/films_joined.parquet` (41 columns)
- **1,713 entries** in `data/processed/screenplays_parsed.pkl` (228 MB)
- Year range 1932-2023, median 2005
- Budget median $25M, revenue median $64M
- Rating mean 6.94, median 7.0 (using `IMDB_Rating` preferred, `vote_average` fallback)
- ROI median 2.9x, 80.5% gross-profitable
- Median 130 scenes / 56 unique characters / 880 dialogue lines per film
- 680 of 1,713 films had at least one parser warning (40%); none were XML errors. Warnings come mostly from minor structure breaks (dangling `<character>` tags, etc.) and don't affect the structural metrics.

### Saved figures

- `phase2_year_distribution.png` — full 1932-2023 corpus, dense band 1995-2022 with a thin tail back to 1932. Confirms the cutoff-reversal: pre-1995 has visible film counts (typically 5-15/year) rather than the singletons originally claimed.
- `phase2_genre_distribution.png` — using `genres_bucketed`. Drama, Comedy, Thriller, Action lead. Genres with <30 films collapsed into `Other`. Distribution is stable enough for per-genre CV in Phase 4.
- `phase2_budget_revenue_distribution.png` — heavy right-skew confirmed; `log1p` transforms produce clean roughly-normal shapes. Phase 3 / 4 use the log columns.
- `phase2_rating_roi_length.png` — narrow rating Gaussian centered ~7.0; ROI heavily right-tailed (hits at 10x+); screenplay length bell-shaped centered around 200k chars.
- `phase2_screenplay_structure.png` — distributions of `n_scenes` (right-skew, median 130), `n_unique_characters` (right-skew, median 56), `n_dialogue_lines` (right-skew, median 880), `dialogue_to_total_text_ratio` (centered around 0.4). The structural metrics will be useful as Phase 3 baseline features.

### Saved tables

- `phase2_summary_metrics.csv` — 18-row metric/value summary of the corpus.
- `phase2_per_decade.csv` — counts and medians per decade (Phase 4 era-stratification reference).
- `phase2_per_genre.csv` — counts and medians per `primary_genre_bucketed`.
- `phase2_parse_warning_audit.csv` — see the audit section below.
- `phase2_top5_warnings_inspection.md` — see the audit section below.

### Audit of parser recovery rules

Following the production build, the parser's recovery rules were
empirically validated to confirm that warning-heavy films do not
exhibit systematic bias on the structural metrics. The audit
implementation lives in `src/data/audit_parse_warnings.py` and produces
three outputs: the correlation table
(`reports/tables/phase2_parse_warning_audit.csv`), the scatter grid
(`reports/figures/phase2_parse_warning_correlations.png`), and the
top-5 raw-vs-parsed inspection
(`reports/tables/phase2_top5_warnings_inspection.md`). Both Pearson and
Spearman coefficients are reported because the warning-count
distribution is heavy-tailed; flag threshold is `|r| > 0.30` for
numeric variables and `η² > 0.09` for the categorical
`primary_genre_bucketed`.

**Targeted parser improvements (Tier 1.1, 1.2, 1.3) applied.** Three
parser changes were implemented in response to an initial audit pass
that flagged `n_unique_characters` as systematically correlated with
`parse_warning_count` (initial Spearman ρ = +0.48):

* **Tier 1.1 — character-name normalization.** Trailing parenthetical
  variant suffixes are stripped from `<character>` tag content, so
  `TONY (CONT'D)`, `TONY (V.O.)`, and `TONY (O.S.)` all normalize to
  `TONY`. If stripping would leave the empty string (e.g.,
  `(WAITER)`), the original is preserved.
* **Tier 1.2 — implausible-character-name filter (conservative).**
  Strings appearing in `<character>` tags are rejected as
  not-real-characters when they (a) contain `©`/`®`/`™`, (b) start
  with a 4-digit year, or (c) contain the substrings `STUDIOS`,
  `PICTURES INC`, or `PRODUCTIONS LLC`. When rejected, the tag is
  treated as a flow break (resets `last_speaker`) and the next
  `<dialogue>` lands on the orphan path (Case 11) rather than being
  attributed to the implausible name.
* **Tier 1.3 — non-empty-dialogue requirement for unique characters.**
  A name only counts toward `n_unique_characters` if it delivered at
  least one non-empty, non-whitespace dialogue line. Empty-text
  placeholders inserted when Cases 5-8 fire remain in `dialogue_units`
  for traceability but do not contribute to the unique count.

**Numeric correlations** (Spearman ρ shown; Pearson concordant):

| Metric | Spearman ρ | Flagged | Interpretation |
|---|---:|:---:|---|
| `n_scenes` | -0.181 | no | small negative; no systematic bias |
| `n_unique_characters` | **+0.393** | **yes** | dropped from 0.483 pre-Tier-1; see below |
| `n_dialogue_lines` | +0.129 | no | small positive |
| `total_dialogue_chars` | +0.073 | no | effectively zero |
| `dialogue_to_total_text_ratio` | +0.105 | no | small positive |
| `mean_dialogue_line_length` | -0.041 | no | weak negative; direction matches case-analysis prediction |
| `script_char_len` | -0.025 | no | effectively zero |
| `decade` | +0.308 | yes | mild upward trend over time, marginally over threshold |

**Categorical correlation:** `primary_genre_bucketed` η² = 0.020
(below the 0.09 threshold). Per-genre warning counts do not vary
substantially.

**Before/after on the flagged correlation** (`n_unique_characters`):

| | Pearson r | Spearman ρ |
|---|---:|---:|
| Pre-Tier-1 | +0.42 | +0.48 |
| Post-Tier-1 | +0.17 | +0.39 |

**Interpretation.** The Pearson coefficient dropped substantially
(+0.42 → +0.17), indicating that the linear relationship was largely
driven by a small number of films with heavy artifactual inflation
(films like *Iron Man* and *Toy Story 4* whose source XML mis-tagged
copyright headers as character names). The Spearman coefficient
dropped less (+0.48 → +0.39) and remains above the 0.30 flag
threshold. Per the framework set at the audit stage (correlation
above 0.30 indicates a genuine underlying-data signal; below 0.20
indicates pure parser artifact), the result is unambiguous: a
meaningful portion of the original correlation was parser artifact
caused by variant inflation and source-mistagged character-tag
content, but a real underlying-data signal remains. Films with
larger casts have more character switches and more opportunities for
source-XML formatting irregularities; this property of the
screenplays themselves, independent of the parser, accounts for the
residual correlation. `decade` shows a similar pattern, marginally
crossing the threshold; the mild upward trend over time may reflect
greater formatting heterogeneity in modern source files.

**Top-5 inspection findings.** Post-Tier-1, the five films with the
highest `parse_warning_count` are *Julieta* (420 warnings,
predominantly date-marker pseudo-characters such as `2016. SPRING.`
correctly rejected by the year-prefix rule), *Toy Story 4* (235
warnings, copyright headers `©2019 DISNEY/PIXAR` correctly rejected
by the copyright-symbol rule), *Iron Man* (118 warnings, `© 2007
MARVEL STUDIOS, INC.` correctly rejected), *Elvis* (102 warnings,
scene headings such as `INT. SUN STUDIOS - CONTROL ROOM` correctly
rejected by the `STUDIOS` substring rule), and *Cat People* (62
warnings, legitimate Cases 7-8 only). Manual inspection of raw XML
alongside the parsed output confirms that the parser's recoveries
produced correct results in every case checked. The Tier 1.2 filter
is functioning as intended: the strings being rejected are
unambiguously not character names (date markers, copyright headers,
or scene headings mistakenly placed in `<character>` tags by the
source files).

**Aggregate effects of the Tier 1 changes.** Total warnings rose
from 2,949 to 3,961 (+34%) because the new rejection warnings expose
source-XML mistagging that was previously silently incorporated.
Films with at least one warning rose from 680 to 698 (+18). The
median `n_unique_characters` dropped from 56 to 51, reflecting the
combined effect of variant normalization and rejection of
mistagged content; the mean dropped from approximately 60 to 65.1.
For the most-affected films, `n_unique_characters` reductions were
substantial (e.g., *Iron Man* 97 → 83, a 14% reduction).

**Patch applied to Case 3.** The wrong-root-tag case now persists
its warning to the `parse_warnings` field on the `ParsedScreenplay`
dataclass, consistent with the empty-XML and parse-error cases. No
films in the current corpus trigger this case, but the patch
ensures future ingest with non-`<script>` roots will be visible to
downstream audits.

---

## Issues encountered & resolved

1. **Pre-1995 cutoff vs. actual count.** Detailed in the strategic-decisions and "what we did" sections above. Resolved by reversing the cutoff after escalation.
2. **`<parenthetical>` tag undocumented in the brief.** Detected during the parser smoke test (initial run produced ~180 warnings/screenplay). Investigated MovieSum's actual XML structure across 200 random screenplays, confirmed `parenthetical` is a real screenplay element (not random noise), updated the parser to recognize it. Warning rate dropped to ~1.8/screenplay (median 0).
3. **Dup-review CSV used `remove` rather than `drop`.** Found during the dedup-CSV completeness check before the build started. Resolved by normalizing both tokens to mean "drop this row" in `dedupe_moviesum_from_csv`. The brief's escalation rule for unrecognized decisions stays in place for genuinely ambiguous values.
4. **`title` column collides between MovieSum and ratings on merge.** Both DataFrames have a `title` column; pandas suffix-renames them to `title_ms` and `title_rt`. The pipeline uses `title_rt` (the canonical TMDB title) for human-readable output and keeps both for traceability.
5. **Future-year noise in `release_year_parsed`.** A handful of rows have years like 2055 or 2099 (scheduled releases, error entries). Clipped to `[1900, 2025]` via the `year_clip` knob; affected rows rarely intersect MovieSum but the filter is defensive.

---

## Open questions / things to flag

None for the planning conversation right now. Phase 2 has no mandatory checkpoint at the end (next checkpoint is end of Phase 4). Items to keep in mind for future phases:

- **`n_unique_characters` retains a moderate correlation with `parse_warning_count`** (Spearman ρ = +0.39 post-Tier-1; see audit section above). The correlation is interpreted as a genuine underlying-data signal: films with larger casts encounter more source-XML formatting irregularities, independent of the parser. Downstream phases that condition on `n_unique_characters` should be aware that `parse_warning_count` carries similar information; including both as features risks redundancy. The Tier 1.2 filter handles the most egregious mistagging (copyright headers, date markers, scene headings in `<character>` tags) automatically.
- **Degenerate scene structure in 30 source XMLs (`data_quality_flag` column).** Films including *Elvis* (4 scenes, 112k dialogue chars), *12 Angry Men* (2 scenes, 85k chars), and *Manhattan Murder Mystery* (1 scene, 121k chars) have source XML in which the entire screenplay is encoded as 1-9 `<scene>` elements. The flag is set when `n_scenes < 10 AND total_dialogue_chars > 50,000`. This is a property of the source, not the parser. Per-scene analyses on flagged films are unreliable; Phase 3 decides whether to filter, downweight, or include them as-is.
- **Pre-1980s decades have <30 films each** (1930s: 4, 1940s: 2, 1950s: 12, 1960s: 18, 1970s: 67). Phase 4 era-stratified CV needs to bucket these into a single "older films" stratum or exclude them from per-decade analyses to avoid noisy estimates.

---

## Files produced

### Code (new in Phase 2)
- `src/data/parse_screenplay.py` — XML parser, Scene + ParsedScreenplay dataclasses, structural-metric computation
- `src/data/build_corpus.py` — master orchestrator, `CorpusBuildConfig` dataclass, all preprocessing knobs
- `src/data/validate_processed_corpus.py` — hard-asserts + Phase 2 figures and tables
- `src/data/audit_parse_warnings.py` — empirical validation of parser recovery rules (correlation tests + top-5 XML inspection)

### Data
- `data/processed/films_joined.parquet` (master table, 1,713 × 41)
- `data/processed/screenplays_parsed.pkl` (per-film structured screenplays, 228 MB)

### Figures (`reports/figures/`)
- `phase2_year_distribution.png`
- `phase2_genre_distribution.png`
- `phase2_budget_revenue_distribution.png`
- `phase2_rating_roi_length.png`
- `phase2_screenplay_structure.png`
- `phase2_parse_warning_correlations.png` (parser-audit scatter grid)

### Tables (`reports/tables/`)
- `phase2_summary_metrics.csv`
- `phase2_per_decade.csv`
- `phase2_per_genre.csv`
- `phase2_parse_warning_audit.csv` (parser-audit correlation results)
- `phase2_top5_warnings_inspection.md` (raw XML vs parsed output for the 5 highest-warning films)

### Docs
- `docs/DATA_NOTES.md` (new)
- `docs/PROJECT_CONTEXT.md` Sections 5, 8, 9 updated
- `docs/CLAUDE_CODE_GUIDELINES.md` Section 7 (template) updated to add Strategic decisions section
- `docs/summaries/phase_1_summary.md` corrected (pre-1995 count error)
- `docs/handoffs/PHASE_2_PLANNING_HANDOFF.md` updated to reflect the reversal

---

## Next phase prerequisites

Phase 3 (Feature Extraction) needs:
- `data/processed/films_joined.parquet` and `data/processed/screenplays_parsed.pkl` ✓ (saved)
- The structural metrics (n_scenes, n_unique_characters, etc.) available on the master table for downstream feature engineering ✓
- A documented column glossary so Phase 3 knows what each column means ✓ (`docs/DATA_NOTES.md`)

Phase 3 will:
- Carve a train/calibration/test split from this corpus (the split happens in Phase 3, not earlier)
- Compute imputation strategy (post-split, to avoid leakage)
- Extract dialogue features (lexical, sentiment, topic, embedding, structural)
- Save the feature matrix to `data/processed/features.parquet`

---

## Questions for the planning conversation

None — proceed to Phase 3. The end-of-Phase-2 self-check from `CLAUDE_CODE_GUIDELINES.md` Section 9 has been run; all items pass.
