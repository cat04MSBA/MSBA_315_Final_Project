# DATA_NOTES — Processed Corpus Reference

> A standing reference for the master working dataset built by Phase 2.
> Future phases reference this rather than re-deriving understanding.
> Last updated: 2026-05-02 (Phase 2 complete).

---

## 1. Corpus at a glance

- **Source files:**
  - `data/processed/films_joined.parquet` (1,713 rows × 41 columns)
  - `data/processed/screenplays_parsed.pkl` (228 MB; pickle of `dict[imdb_id, ParsedScreenplay]`)
- **Built by:** `python -m src.data.build_corpus` (idempotent)
- **Validated by:** `python -m src.data.validate_processed_corpus`
- **One row per film**, keyed by `imdb_id`.

### Headline numbers

| | Value |
|---|---|
| Total films | 1,713 |
| Year range | 1932-2023 |
| Year median | 2005 |
| Budget median | $25M |
| Revenue median | $64M |
| Rating median (`effective_rating`) | 7.0 |
| ROI median | 2.9x |
| Films with revenue > budget | ~80% |
| Median scenes per film | 130 |
| Median unique characters | 56 |
| Median dialogue lines | 880 |
| Films with parse warnings | 680 (40%) — minor issues, no failures |

---

## 2. Known biases and caveats

These shape what the model can claim, not what it can do. They live in
the report's limitations section.

- **Survivorship bias.** ~80% of the corpus is gross-profitable, well
  above the industry's roughly 50% net-profitable rate. The corpus
  contains films that were produced AND known well enough to land in a
  major metadata aggregator. Phase 6 cost-decision tuning must use
  industry base rates rather than corpus rates, and Phase 8 will run
  sensitivity analysis across alternative cost matrices.
- **Year-density skew.** Most of the corpus sits in the 2000s and 2010s.
  Pre-1980 has fewer than 30 films per decade; pre-1960 has only a
  handful. Era-stratified CV in Phase 4 should bucket pre-1980s decades
  into a single "older films" stratum rather than treat them as
  independent strata.
- **Genre skew.** Drama, Comedy, Thriller, Action dominate. The
  `genres_bucketed` column collapses any genre with fewer than 30 films
  in the corpus into `Other` to keep per-genre splits well-conditioned.
- **No reweighting applied.** The Phase 1 strategic decision (planning
  conversation 2026-05-02) chose to keep the corpus as-is and document
  the bias rather than reweight at training time.

---

## 3. Column glossary

### Identity and source columns
| Column | Type | Source | Notes |
|---|---|---|---|
| `imdb_id` | str | MovieSum | `tt`-prefixed, unique key |
| `id` | int | ratings dataset | TMDB id |
| `movie_name` | str | MovieSum | Original "Title_YYYY" string |
| `title` (`_ms`, `_rt`) | str | both | `title_ms` from MovieSum (year stripped), `title_rt` from ratings |
| `original_title` | str | ratings | TMDB original-language title |
| `year_in_title` | Int64 | MovieSum | Year parsed from `movie_name` suffix |
| `release_date` | str | ratings | YYYY-MM-DD |
| `release_year_parsed` | Int64 | derived | Authoritative year (from `release_date`) |
| `origin_split` | str | MovieSum | `train` / `val` / `test` (MovieSum's upstream summarization split, not ours) |

### Outcome / financial columns
| Column | Type | Notes |
|---|---|---|
| `budget` | int | USD; **0 was a missing-value sentinel** in source, all rows here are > 0 |
| `revenue` | int | USD; same convention as `budget` |
| `runtime` | int | Minutes; 0-as-missing convention in source |
| `vote_average` | float | TMDB's 0-10 user rating |
| `vote_count` | int | Vote count (popularity proxy) |
| `IMDB_Rating` | float | IMDb's 0-10 rating; 0-as-missing |
| `AverageRating` | float | Smoothed/external rating |
| `Meta_score` | float | Metacritic 0-100; sparse |
| `popularity` | float | TMDB popularity score |

### Derived columns (added by `build_corpus.py`)
| Column | Type | Definition |
|---|---|---|
| `effective_rating` | float | `IMDB_Rating` if > 0, else `vote_average`. Always in [0, 10]. |
| `log_budget` | float | `log1p(budget)` (`monetary_transform` knob, default `log1p`) |
| `log_revenue` | float | `log1p(revenue)` (same knob) |
| `genres_parsed` | list[str] | Parsed from `genres_list` raw string |
| `primary_genre` | str | First entry of `genres_parsed`; `"Unknown"` if empty |
| `genres_bucketed` | list[str] | `genres_parsed` with sub-`genre_min_count` genres → `"Other"` |
| `primary_genre_bucketed` | str | First entry of `genres_bucketed` |

### Screenplay-structural columns (computed by parser)
| Column | Type | Definition |
|---|---|---|
| `n_scenes` | int | Number of `<scene>` elements |
| `n_unique_characters` | int | Distinct character names across all scenes |
| `n_dialogue_lines` | int | Total `(character, dialogue)` pairs |
| `total_dialogue_chars` | int | Sum of dialogue text length |
| `total_stage_direction_chars` | int | Sum of `<stage_direction>` text (usually slugline only) |
| `total_scene_description_chars` | int | Sum of `<scene_description>` text (action paragraphs) |
| `total_action_chars` | int | `stage_direction + scene_description` |
| `dialogue_to_action_ratio` | float | `dialogue / (dialogue + stage_direction)`. Tends to ~0.99; brief-literal formula. |
| `dialogue_to_total_text_ratio` | float | `dialogue / (dialogue + total_action)`. More informative; mean ~0.40. |
| `parse_warning_count` | int | Number of warnings raised while parsing this screenplay (most are minor: dangling `<character>` tags, unexpected sub-tags) |

---

## 4. Edge cases the pipeline handles

- **`0` as missing-value sentinel for `budget` / `revenue` / `runtime`.**
  TMDB CSVs use 0 instead of NaN for missing financial data because INT
  columns can't hold NaN. The pipeline treats `> 0` as the inclusion
  test, not `notna()`. Hard assertion in `validate_processed_corpus.py`.
- **Future-year noise (e.g., 2055, 2099) in raw `release_date`.**
  Clipped via `year_clip` knob to `[1900, 2025]` before any year-based
  filter or plot.
- **MovieSum same-IMDb-ID duplicates (12 pairs of alternate titles or
  alternate drafts).** Resolved by reading the user-filled review CSV
  at `reports/tables/phase1_moviesum_duplicates_review.csv`. The
  pipeline accepts `keep`/`drop`/`remove`/`delete`/`discard` synonyms;
  unrecognized or contradictory entries raise loudly.
- **Ratings dataset duplicates on `imdb_id` (alternate cuts, regional
  releases).** Deduped by keeping the row with higher `vote_count`
  (`dedupe_ratings()` in `src.data.join_corpus`).
- **`<parenthetical>` tags in MovieSum XML** (e.g., `(softly)`, `(beat)`).
  Recognized as continuation markers; dialogue following a parenthetical
  with no intervening `<character>` is attributed to the previous
  speaker. Not stored in the dialogue tuples (they're 2-tuples per the
  brief).
- **Malformed XML in any individual screenplay.** Returns a degenerate
  `ParsedScreenplay` with empty scenes and the error in
  `parse_warnings`. Does not crash. None observed in the current
  corpus (0 XML errors out of 1,713).

---

## 5. File layout

```
data/processed/
├── films_joined.parquet          # master table (this corpus)
└── screenplays_parsed.pkl        # dict[imdb_id, ParsedScreenplay]
```

**Loading from a notebook:**

```python
import pandas as pd, pickle
df = pd.read_parquet("data/processed/films_joined.parquet")
with open("data/processed/screenplays_parsed.pkl", "rb") as f:
    screenplays = pickle.load(f)  # dict[imdb_id, ParsedScreenplay]
```

The parquet has every metadata + outcome + screenplay-structural-metric
column. The pickle has the full per-film scene structure (Phase 3
feature extraction reads both).

---

## 6. Where to find more

- **EDA narrative + interpretations:** `notebooks/phase_1.ipynb` and
  `docs/summaries/phase_1_summary.md`.
- **Pipeline implementation:** `src/data/build_corpus.py` (orchestrator)
  and the loaders / parser it calls.
- **Pipeline knobs:** `CorpusBuildConfig` dataclass in `build_corpus.py`.
  Each preprocessing default has a knob; override the field to test
  alternatives.
- **Validation invariants:** `src/data/validate_processed_corpus.py`
  `hard_asserts()`.
- **Phase 2 figures and tables:** `reports/figures/phase2_*.png`,
  `reports/tables/phase2_*.csv`.
- **Strategic-decisions audit trail:** `docs/PROJECT_CONTEXT.md`
  Section 8 (decisions log) and `docs/PLANNING_HANDOFF.md`.
