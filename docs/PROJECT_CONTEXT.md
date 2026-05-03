# MSBA315 Project — Project Context

> This document is the source of truth for the project. Read it at the start
> of every phase. Update it (in the "Decisions Log" and "Data Summary" sections)
> when material decisions are made or when data understanding evolves.
> Do not delete content — append and date.

---

## 1. Project Framing

### The problem

Movie studios receive thousands of unproduced scripts each year. They must
decide which to greenlight, knowing that:

- Greenlighting a flop costs ~$50M (lost production budget)
- Passing on a hit costs ~$100-200M (foregone revenue)
- Human script readers are expensive but limited in throughput

At the moment of decision (pre-greenlight), the strongest predictors of film
success — budget, cast, marketing spend — do not yet exist. The only signal
available is the **script itself**.

### The user

A development executive or script reader at a film studio, performing
pre-greenlight triage on incoming screenplays.

### What the system does

Given a script's dialogue, the system outputs a triage decision —
**Greenlight / Pass / Refer to human reader** — together with a calibrated
confidence interval and scene-level explanations of what drove the decision.

### Why this framing is defensible

- The dialogue-only constraint is **operationally justified**, not arbitrary:
  at pre-greenlight, no other signal exists.
- The user, the moment of use, and the artifact are all clearly identified.
- The cost asymmetry is real and quantifiable, which makes the decision
  layer methodologically meaningful rather than decorative.

---

## 2. The Four-Layer System

The system is a single pipeline with four layers stacked on top of one core
predictive model. Each layer takes the previous layer's output and adds
something. They are not four independent models.

```
Layer 4: Actionable feedback (which scenes drag prediction down/up)
   ↑
Layer 3: Asymmetric-cost decision (Greenlight / Pass / Refer)
   ↑
Layer 2: Calibrated uncertainty (confidence intervals via conformal prediction)
   ↑
Layer 1: Core prediction model (XGBoost on dialogue features)
   ↑
Data: Movie dialogues + outcomes (IMDb rating, box office)
```

### Layer 1 — Core prediction
Standard ML: features extracted from dialogue → predicted outcome
(IMDb rating and box office tier). Phase 4 benchmarks several candidate
models and selects the primary one collaboratively.

### Layer 2 — Calibrated uncertainty
Wraps the model's output with calibrated confidence intervals. Uses
conformal prediction (`mapie` library) and temperature scaling. Validates
calibration empirically via reliability diagrams.

### Layer 3 — Asymmetric-cost decision
Takes the calibrated prediction and applies a cost matrix reflecting actual
studio risk asymmetries. Outputs one of three actions. Includes sensitivity
analysis across cost matrices.

### Layer 4 — Actionable feedback
Uses SHAP (TreeExplainer) for feature attribution. Where possible, attributes
at the scene/chunk level so that explanations are actionable to writers.

---

## 3. Novelty Claim

The combination of these four layers, applied to script triage, has not been
published. The novelty is **architectural**, not algorithmic — known
techniques assembled coherently for a use case that has not been served:

- Cost-sensitive decision-making is standard in fraud / medical diagnosis
  but not applied to script triage.
- Calibrated uncertainty for film prediction is absent in the existing
  literature (Eliashberg 2014, Mariani 2020, ScriptBook, Forecasting Film
  Audience Ratings 2025).
- Scene-level SHAP as actionable feedback to writers is novel for this
  domain.

The contribution is defensible as "methodological maturity in combining
established techniques for an underserved use case."

---

## 4. Data Sources

### Primary: MovieSum (Saxena & Keller, ACL 2024)
- 2,200 movie screenplays, manually formatted in structured XML
- Each screenplay tagged with IMDb ID natively — direct join to other datasets
- XML structure preserves scene boundaries, character names, dialogue lines,
  and stage directions / scene descriptions
- Range: 1930–2023, all genres, English-language
- Source: Hugging Face (`rohitsaxena/MovieSum`)
- License: research use; cited paper available
- Size: ~50-100MB total

XML structure of each screenplay:
```xml
<script>
  <scene>
    <stage_direction>...</stage_direction>
    <scene_description>...</scene_description>
    <character>NAME</character>
    <dialogue>...</dialogue>
    ...
  </scene>
  ...
</script>
```

### Primary: IMDb-TMDB Movie Metadata (Kaggle, 1M-row release)
- File: `data/raw/ratings_data/IMDB TMDB Movie Metadata Big Dataset (1M).csv`
- ~1.07 million films, ~13,800 with both budget and revenue
- Carries `imdb_id` (`tt`-prefixed) **and** TMDB `id` natively, so the
  MovieSum→ratings join is a direct exact-ID merge (no fuzzy matching,
  no external bridge required)
- Source for: budget, revenue, runtime, genre list, release date,
  TMDB rating (`vote_average`), IMDb rating (`IMDB_Rating`), Metacritic
  score (`Meta_score`), director, production companies, etc.
- License: see Kaggle dataset page (research / non-commercial use)

### Fallback / Supplementary: IMSDb Kaggle script dumps
- Used only if MovieSum + ratings join yields fewer than ~1,000 films
- Several available on Kaggle (Ismael 2024 has 1,172 scripts)
- Lower quality than MovieSum (raw scrapes, inconsistent formatting)
- Would require deduplication against MovieSum by IMDb ID
- (Phase 1 result, 1,713 four-signal films, makes invoking this fallback
  unnecessary.)

### Cornell Movie-Dialogs Corpus (held-out validation, optional)
- 617 films with cleaned dialogue
- Used as an out-of-distribution validation set in Phase 8 if useful
- Decision deferred until Phase 4 results are in

### Why screenplays over subtitles

The pre-greenlight framing assumes the system consumes what studios actually
receive: screenplays. Subtitles (OpenSubtitles) would have given a larger
corpus (~3,000 films) but with significant downsides:

- Subtitles lack speaker attribution — features like number of speakers,
  per-character dialogue volume, character interaction graphs become
  impossible
- Subtitles lack scene structure — Layer 4 (scene-level SHAP) would
  degrade to coarse time-segment attribution
- Subtitles include song lyrics, sound descriptions, and translation
  artifacts as noise
- Most importantly: training on subtitles when the deployed system
  consumes screenplays creates a methodological gap that would need to
  be defended in the report

MovieSum's smaller-than-OpenSubtitles corpus (~1,700 expected after the
ratings join) is offset by significantly higher data quality and exact
format alignment with the deployment target.

---

## 5. Data Summary

> Updated by Claude Code as Phase 1 progresses. Add timestamps, do not
> overwrite prior entries.

**Status as of 2026-05-02:** Phase 1 complete; corpus size lands above
the 1,500-film "green band" threshold (1,713 four-signal films). Numbers
below are the actual measured values; see
`docs/summaries/phase_1_summary.txt` for plots and interpretation.

### Ratings dataset (Phase 1, complete — 2026-05-02)
- Total films: 1,072,255
- Films with `imdb_id` (`tt`-prefixed): ~589,000
- Films with budget > 0: ~54,400
- Films with revenue > 0: ~20,300
- Films with both: ~13,800
- Year coverage: dense 1995–2023; thin tail back to ~1900;
  a handful of nonsensical future years (2055, 2099) — scheduled
  releases or noise, dropped at plot time.

### MovieSum (Phase 1, complete — 2026-05-02)
- Total screenplays: 2,200 (1,800 train + 200 val + 200 test;
  matches published documentation)
- IMDb ID coverage: 100% well-formed (`tt\d{7,10}`); **2,188 unique IDs**
  due to 12 same-IMDb-ID duplicate pairs (alternate titles or alternate
  drafts of the same film). Dedup keeps the longest script per ID;
  Phase 2 will formalize the policy after the user reviews the pairs
  via `src/data/review_duplicates.py`.
- Year range: 1931–2023, median 2007, ~21% post-2016
- Mean screenplay length: ~207k characters (~34k tokens — matches the
  README's claim)
- XML structure verified: `<script><scene>{<stage_direction>,
  <scene_description>, <character>, <dialogue>}*</scene>*</script>`,
  parses cleanly with `xml.etree.ElementTree`

### MovieSum × ratings join (Phase 1, complete — 2026-05-02)
- Strategy: direct exact-ID merge on `imdb_id` (both datasets carry the
  IMDb ID natively). No fuzzy matching, no external bridge.
- Films matched: **2,186 / 2,188 (99.9%)** — only 2 MovieSum films are
  absent from the ratings dataset entirely.
- **Working corpus (matched + budget>0 + revenue>0 + rating>0):
  1,713 films.**
- Year distribution: 1932–2023, median 2005, dense 1995–2022.
- Genre distribution: Drama / Comedy / Thriller / Action lead heavily;
  small genres and documentaries are sparse.
- Budget median $25M; revenue median $64M; rating mean ~7.1 (using
  `IMDB_Rating` preferred over `vote_average`); ROI median ~2.9x,
  **~80% gross-profitable** (still survivor-biased, but less than
  alternative ratings sources would have produced).
- Working dataset saved to
  `data/interim/phase1_joined_corpus.parquet` (2,188 rows; the 1,713
  four-signal subset is filtered at load time downstream).

### Decision criteria after Phase 1
- ≥1,500 films joined: proceed with MovieSum-only as primary corpus
- 1,000-1,500 films joined: proceed with MovieSum primary, document
  smaller corpus as a limitation
- <1,000 films joined: add IMSDb Kaggle scripts as supplementary,
  dedupe by IMDb ID, expand corpus to ~1,500

### Phase 2 processed corpus (complete — 2026-05-02)
- **Final master corpus: 1,713 films** in
  `data/processed/films_joined.parquet`. Same scale as the Phase 1
  working corpus (the planned pre-1995 cutoff was reversed mid-phase
  after a Phase 1 EDA recount; see decisions log).
- Year range 1932-2023, median 2005.
- Budget median $25M, revenue median $64M, rating mean ~6.94,
  ROI median ~2.9x, ~80% gross-profitable.
- 41 columns: source columns + derived columns
  (`effective_rating`, `log_budget`, `log_revenue`, `primary_genre`,
  `genres_bucketed`, `primary_genre_bucketed`) + screenplay-structural
  metrics (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
  character / action / dialogue char counts, two ratios).
- Per-screenplay structured form saved separately to
  `data/processed/screenplays_parsed.pkl` (228 MB,
  `dict[imdb_id, ParsedScreenplay]`); Phase 3 reads both.
- Median 130 scenes / 56 unique characters / 880 dialogue lines per
  film.
- See `docs/DATA_NOTES.md` for the full column glossary, edge-case
  documentation, and biases-to-remember.

---

## 6. Methodology Principles

These are non-negotiable across all phases. Apply them consistently.

### No data leakage
- Train/test split happens **once**, before any feature engineering or
  model fitting.
- Test set is touched **once**, at final evaluation.
- Cross-validation folds respect temporal ordering where relevant.
- Calibration set is separate from training and test sets.
- Features computed from the full corpus (e.g. LDA topics, TF-IDF
  vocabularies) are fit on the **training fold only** and applied to test.

### Reproducibility
- All random seeds set explicitly. Standard seed: `42`.
- Deterministic library settings where possible.
- All scripts run from project root with relative paths.
- Saved intermediate artifacts are timestamped and version-tagged in
  filename when format changes.

### Honest reporting
- Negative results are reported as findings, not hidden.
- Modest accuracy is reported honestly; the asymmetric-cost framing is
  what makes the system useful, not headline R² numbers.
- All limitations (survivorship bias, corpus skew, etc.) are explicit
  in the report.

### Statistical rigor
- 5-fold cross-validation for model comparison.
- Bootstrapped confidence intervals for performance metrics.
- Multiple-comparison correction when testing many hypotheses
  (Bonferroni or Benjamini-Hochberg).

### Survivorship bias acknowledgment
- The corpus contains films that got produced. The system predicts
  among production-ready scripts, not among all possible scripts.
- This is documented in every report section that discusses scope.

---

## 7. Project Structure

```
project_root/
├── docs/
│   ├── PROJECT_CONTEXT.md          # this file
│   ├── CLAUDE_CODE_GUIDELINES.md   # engineering standards
│   ├── PROJECT_ROADMAP.md          # phase-by-phase outline
│   ├── briefs/
│   │   ├── phase_1_brief.md        # execution brief per phase
│   │   ├── phase_2_brief.md
│   │   └── ...
│   └── summaries/
│       ├── phase_1_summary.md      # postmortem per phase
│       ├── phase_2_summary.md
│       └── ...
├── src/
│   ├── data/                       # data loading & joining
│   ├── features/                   # feature extraction
│   ├── models/                     # Layer 1 model code
│   ├── calibration/                # Layer 2
│   ├── decision/                   # Layer 3
│   ├── explanation/                # Layer 4
│   ├── evaluation/                 # metrics, error analysis
│   └── utils/                      # logging, paths, seeds
├── data/
│   ├── raw/                        # downloaded source files (gitignored)
│   ├── interim/                    # intermediate artifacts (gitignored)
│   └── processed/                  # final working datasets (gitignored)
├── notebooks/                      # exploratory notebooks (minimal)
├── reports/
│   ├── figures/                    # plots saved during phases
│   └── tables/                     # tables saved during phases
├── runs/                           # per-run experiment artifacts
│   ├── RUNS.md                     # human-readable index, newest first
│   └── <phase>/<YYYYMMDD_HHMM>_<name>/
│       ├── params.json             # hyperparameters
│       ├── preprocessing_summary.json
│       ├── features_used.json
│       ├── metrics.json
│       ├── run.log                 # full INFO/DEBUG trace per run
│       └── model.joblib            # gitignored; the rest is tracked
├── tests/                          # smoke tests for critical functions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. Decisions Log

> Append to this section, with date and time, for when any methodology choices or decisions
> are made. This acts as a history log, to track all changes and decisions made. Do not delete prior entries. 
> The decisions log is append-only. Old entries are never deleted or rewritten, even if a later decision reverses them — a > reversal is logged as a new entry referencing the old one. These changes should be succint with max 2 sentences
> explaining the decision and big picture justification. The more detailed explanation and rationale should always be
> detailed in the summary.

Example of format to follow
Format: one entry per decision, newest first
## YYYY-MM-DD HH:MM — [Short decision title]

**Phase:** [Phase number and name]
**Decision:** [One or two lines stating what was decided.]
**See also:** `docs/summaries/phase_N_summary.txt` for full rationale.


## 2026-05-03 17:20 — Phase 3a/3b: baseline expanded from 1 to 4 model families

**Phase:** Phase 3 — Feature Extraction (cross-cutting; affects 3a and 3b)
**Decision:** The original Phase 3a baseline used a single model family (L2-regularized linear: RidgeCV for regression, LogisticRegressionCV for classification) per the brief's "linear / regularized linear is the obvious default" guidance. The first Phase 3b ablation (lexical group) produced a negative lift on the linear baseline, raising the question of whether the issue was the features themselves or a peculiarity of the linear model family. To disambiguate, the Phase 3a baselines and every subsequent Phase 3b ablation now run across **four model families** with distinct inductive biases: linear (L2), histogram gradient boosting, k-nearest-neighbours, and SVM with RBF kernel. The linear family remains the historical reference for the brief's escalation threshold check; the other three families are reported alongside as a diagnostic. The 4-family run reveals (a) HistGB exceeds linear on the structural baseline alone (R² 0.069 vs 0.052, `roi_gt_2` AUC 0.610 vs 0.602), suggesting Phase 4 should expect tree-based models to outperform linear ones on this corpus; (b) the lexical group's negative lift is consistent across all four families (HistGB drops R² 0.009 and `roi_gt_1` AUC 0.041 with lexical added), strengthening rather than weakening the original "no signal" interpretation. KNN and SVM are weak baselines on this corpus (likely due to the high feature-to-sample ratio); their values are reported for completeness.
**See also:** `docs/handoffs/phase_3a_handoff.md` for the multi-family floor numbers; `docs/handoffs/phase_3b_lexical_handoff.md` for the lexical verdict.

## 2026-05-03 14:10 — Phase 3a: baseline feature parameterization revised (log-transforms + log_runtime added)

**Phase:** Phase 3 — Feature Extraction (sub-phase 3a)
**Decision:** The Phase 3a baseline as originally implemented placed point estimates just above the brief's escalation thresholds (R² 0.051, AUC 0.559 / 0.582 dialogue-only) but with 95% CI lower bounds dipping below the floors on the two weaker targets, and omitted `log_runtime` despite the brief listing it as a likely candidate. Two changes adopted: (1) `log1p` applied to the six heavy-tailed structural counts (`n_scenes`, `n_unique_characters`, `n_dialogue_lines`, `total_dialogue_chars`, `total_action_chars`, `parse_warning_count`) before z-scoring; (2) `log_runtime` added to the deployable baseline (runtime is leak-free pre-greenlight, page count to minutes is industry convention). Both changes implemented as flags on `BaselineFeatureConfig`; the trainer now writes 28 rows covering the four feature configurations. Original (un-logged, no-runtime) rows preserved in `phase3a_baseline.csv` for the report's before/after comparison. `roi_gt_2` is the only target whose 95% CI fully clears the 0.55 floor post-revision.
**See also:** `docs/handoffs/phase_3a_handoff.md` for the headline numbers, comparison tables, and resolved-questions audit.

## 2026-05-02 23:50 — Phase 2: dataset swap audit-trail entry (TMDB 5000 → IMDb-TMDB 1M)

**Phase:** Phase 2 — Data Pipeline (audit entry; the swap itself happened mid-Phase-1)
**Decision:** Phase 1 began with TMDB 5000 (~4,800 films, hard 2016 cutoff) as the planned ratings source. After the MovieSum × TMDB 5000 join capped at 1,019 four-signal films (in the brief's "document as a limitation" band), the user swapped the ratings source to the **IMDb-TMDB Movie Metadata Big Dataset (1M)** (~1.07M films, native `imdb_id` + `id` columns, year coverage through 2023). Phase 1 working corpus jumped from 1,019 to 1,713 films (+68%); year coverage extended from 2016 to 2023; gross-profitable rate dropped from 86% to 80%, slightly less survivor-biased. The swap was applied retroactively to the foundation docs (Section 4 data sources, Section 5 data summary, Phase 1 brief and summary) so they read as if the new dataset was always the choice. This entry exists in the decisions log to preserve the audit trail per Phase 2 brief Task 6.
**See also:** `docs/handoffs/PHASE_2_PLANNING_HANDOFF.md` for the full numbers and rationale, `docs/summaries/phase_1_summary.md` for what Phase 1 actually produced.

## 2026-05-02 23:35 — Phase 2: pre-1995 cutoff REVERSED

**Phase:** Phase 2 — Data Pipeline
**Decision:** The pre-1995 cutoff committed in the 2026-05-02 15:30 entry below is reversed. The Phase 1 EDA's claim of "~50 pre-1995 films" was a count error: the actual number is 398 films (about 23% of the 1,713-film working corpus). The 1995 cutoff was based on a faulty premise and would have shrunk the corpus to 1,315 films, below the brief's 1,400 floor. Working corpus retains all 1,713 films, year range 1932-2023; Phase 4 era-stratified CV will bucket pre-1980s decades into a single "older films" stratum rather than enforce a hard cutoff. The `min_year` knob in `src.data.build_corpus.CorpusBuildConfig` is exposed for future experimentation.
**See also:** `docs/summaries/phase_1_summary.md` (corrected via strikethrough on the original claim), and the upcoming `docs/summaries/phase_2_summary.md`.

## 2026-05-02 15:30 — Phase 1: pre-1995 cutoff for the working corpus

**Phase:** Phase 1 — Data Feasibility Verification
**Decision:** Working corpus will be restricted to films released ≥ 1995 starting in Phase 2. This drops the long thin tail of pre-1995 films (~50 films, mostly singletons per year) so era-based generalization claims in later phases are clean and the dense band of the corpus dominates training. Other end-of-Phase-1 questions (corpus configuration, survivorship-bias framing, MovieSum dedup policy) remain open.
**Note added 2026-05-02 23:35:** REVERSED — see entry above. The "~50 films" basis was a count error; actual is 398.
**See also:** `docs/summaries/phase_1_summary.md`.


### [PROPOSAL_DATE] — Project framing
- Adopted the four-layer pre-greenlight triage framing.
- Dropped the original Bechdel test / representation analysis after
  literature review showed the gap was thinner than expected.
- Dropped the dialogue-only success-prediction framing (Framing 2)
  in favor of the pre-greenlight framing (Framing 3) for clearer
  problem definition.

### [PROPOSAL_DATE] — Data sources committed: MovieSum + IMDb-TMDB Movie Metadata Big Dataset (1M)
- Selected MovieSum (Saxena & Keller 2024) as the primary screenplay
  source. Reasons: structured XML with scene/character/dialogue tags;
  IMDb IDs included natively; manually formatted; published precedent
  (Gross 2025 used it for Oscar nomination prediction).
- Rejected OpenSubtitles for the primary corpus despite larger potential
  size: subtitles lack speaker attribution and scene structure, which
  are essential for the features and SHAP granularity Layer 4 requires.
  Training on subtitles when the deployed system consumes screenplays
  also creates a domain mismatch.
- Selected the IMDb-TMDB Movie Metadata Big Dataset (1M) as the primary
  ratings source. Reasons: carries `imdb_id` and TMDB `id` natively
  (direct exact-ID join with MovieSum, no fuzzy matching); contains
  budget, revenue, runtime, multiple rating fields, director, and other
  metadata; year coverage extends through 2023, so MovieSum's post-2016
  films are matchable.
- Documented IMSDb Kaggle dumps as fallback if MovieSum × ratings join
  yields too few films (<1,000). Phase 1 result of 1,713 four-signal
  films makes invoking this fallback unnecessary.
- The 2,200-screenplay base of MovieSum is comparable to or larger than
  established screenplay-prediction work (ScriptBase-j: 917; Eliashberg
  2014: ~300).

### [PROPOSAL_DATE] — Model choice (deferred)
- The primary Layer 1 model is not yet selected.
- Phase 4 begins with a simple baseline (Linear / Ridge Regression) to
  establish that dialogue features carry signal at all.
- A range of candidates are then trained and benchmarked: tree-based
  (XGBoost, LightGBM, Random Forest), linear (Ridge, Lasso), and
  optionally deep (DistilBERT fine-tune) if corpus size and compute permit.
- Ensemble approaches (stacking, weighted averaging) are considered after
  individual models are evaluated, if they offer meaningful improvement
  and remain compatible with the calibration and SHAP layers.
- Final selection happens after Phase 4 cross-validation results are in.
  Selection criteria: predictive performance, stability across folds,
  compatibility with downstream layers (calibration, SHAP), and
  interpretability.
- Claude Code should run the baseline first, report results, and request
  guidance from the user before proceeding to ensemble work.

### [PROPOSAL_DATE] — Outcome variables (both, then compare)
- Phase 4 trains separate models for two outcomes:
  - IMDb rating (continuous regression)
  - Box office tier (Flop / Modest / Hit, three-class classification)
- After Phase 4 completes, we compare which outcome the system predicts
  more reliably from dialogue alone. The stronger outcome becomes the
  primary; the weaker one becomes a secondary or supporting analysis.
- The asymmetric-cost decision layer (Layer 3) requires box office to
  define dollar costs. If box office prediction is too weak, we adapt
  by deriving expected revenue from IMDb-rating predictions plus
  rating→revenue regression on the training set.

### [Add future entries here]

---

## 9. Phase Status

> Update after each phase completes.

| Phase | Title | Status | Summary doc |
|---|---|---|---|
| 1 | Data feasibility verification | Complete | `docs/summaries/phase_1_summary.md` |
| 2 | Data pipeline | Complete | `docs/summaries/phase_2_summary.md` |
| 3 | Feature extraction | In progress (3a multi-family floor complete; lexical group complete with null verdict; sentiment proposal next) | `docs/handoffs/phase_3a_handoff.md`, `docs/handoffs/phase_3b_lexical_handoff.md` |
| 4 | Layer 1: Core prediction | Not started | — |
| 5 | Layer 2: Calibration | Not started | — |
| 6 | Layer 3: Decision | Not started | — |
| 7 | Layer 4: SHAP explanations | Not started | — |
| 8 | Integration & evaluation | Not started | — |
| 9 | Report & presentation | Not started | — |

---

## 10. References

### Foundational
- Eliashberg et al. 2014. "Assessing the Future Box Office Performance of Movies Before Their Production." *Management Science.*
- Hamilton et al. 2016. "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change." *ACL.* (background, not used directly)

### Most relevant prior work
- Saxena & Keller 2024. "MovieSum: An Abstractive Summarization Dataset for Movie Screenplays." *Findings of the ACL.*
- Gross 2025. "Predicting Oscar-Nominated Screenplays with Sentence Embeddings." (uses MovieSum for prediction tasks)
- Forecasting Film Audience Ratings 2025 (Sciencedirect) — closest methodological neighbor.
- Mariani et al. 2020. (Nature HSSC) — large-scale subtitle-based prediction.
- Del Vecchio et al. 2018 — emotional arcs from screenplays.

### Methodology references
- Guo et al. 2017. "On Calibration of Modern Neural Networks." *ICML.*
- Angelopoulos & Bates 2021. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." (tutorial)
- `mapie` library documentation for conformal prediction implementation.
- `shap` library documentation, especially TreeExplainer.

### Domain references
- ScriptBook (commercial deployment, used by Fox on *Logan* in 2017).
- Cinelytic, ScriptHop (industry tooling).

---

## 11. Flexibility & Alternatives

This document encodes current decisions but is not a contract. Methodology
choices made here can and should be revisited if execution reveals better
options.

### Brief upload cadence

Phase briefs are uploaded by the user one at a time, as each phase begins.
Do not look for future phase briefs in `docs/briefs/` — they don't exist
yet. Each brief is drafted in the planning conversation after the previous
phase's results are reviewed, then uploaded before the next phase starts.
This keeps methodology choices flexible based on actual results rather
than committed in advance.

### When Claude Code should propose alternatives

Claude Code is encouraged to suggest deviations from this plan when:

- A library, technique, or approach better suited to the corpus or task
  becomes apparent during implementation.
- A planned step turns out to be unnecessary or redundant.
- Diagnostics reveal a problem (data quality, leakage risk, distributional
  issue) that the original plan does not address.
- An unplanned analysis would meaningfully strengthen the report
  (e.g., an additional baseline, a useful ablation, a robustness check).

### How to propose alternatives

When proposing a deviation, Claude Code should:

1. State the proposed change concretely.
2. Explain the rationale — what problem it solves or what it improves.
3. Estimate the cost (additional time, complexity, risk).
4. Compare against the current plan honestly, including downsides.
5. Flag the proposal in the phase summary AND tell the user to bring it
   to the planning conversation before executing it.

### What Claude Code should NOT do unilaterally

- Change the four-layer system architecture
- Change the primary problem framing (pre-greenlight script triage)
- Change which datasets are used as primary sources
- Skip phases or merge phases
- Make decisions that affect downstream phases without flagging them

### What Claude Code can do without flagging

- Choose specific library implementations (e.g., `optuna` vs `GridSearchCV`
  for hyperparameter tuning) provided behavior is equivalent
- Add diagnostic checks, assertions, and validation steps not in the brief
- Refactor code for clarity or efficiency
- Add comments and docstrings beyond what was specified
- Save additional intermediate artifacts that aid reproducibility

### Decision points where Claude Code MUST request guidance

Some choices are too consequential to make unilaterally and too
context-dependent to lock in advance. At each of these points, Claude Code
must complete the relevant phase work, report results, and explicitly
request that the user discuss with the planning conversation before
proceeding:

- **End of Phase 1:** corpus size and quality after all joins.
  Decision needed: do we proceed with the planned scale, expand, or
  reduce scope?
- **Phase 2 preprocessing decisions are Claude Code's to make**, but
  with two structural requirements:
    (a) Each choice (imputation, scaling, feature transforms,
        bucketing, outlier handling, etc.) must be grounded in the
        Phase 1 EDA observations and explained in the Phase 2 summary
        so the user can review the rationale.
    (b) The preprocessing pipeline must be implemented as
        configurable knobs (e.g., a config dict or function arguments
        with sensible defaults), NOT as hardcoded values. This way
        testing alternative techniques later is a one-line change,
        not a refactor.
  Claude Code does NOT have to ask before implementing tactical
  preprocessing choices; the user reviews them in the phase summary
  and can override knob values if they disagree. Strategic choices
  that affect the project's framing (survivorship-bias treatment,
  primary outcome variable, train/calibration/test ratio, etc.) are a
  separate matter and DO require escalation before implementation.
- **End of Phase 4:** baseline + candidate models are trained and
  benchmarked. Decisions needed: which model is primary, which outcome
  variable is primary, do we add ensemble work or move to Phase 5?
- **End of Phase 5:** calibration is empirically validated.
  Decision needed: is calibration good enough for Layer 3 to use, or
  does the model need rework?
- **End of Phase 7:** SHAP results are computed.
  Decision needed: are scene-level explanations meaningful, or do we
  fall back to feature-level only?

At each of these checkpoints, Claude Code writes the phase summary,
prepares a short list of questions for the planning conversation, and
tells the user: *"Phase N complete. Please bring the summary and these
questions to the planning conversation before starting Phase N+1."*
