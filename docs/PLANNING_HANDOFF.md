================================================================
Phase 1 -> Planning Conversation Handoff
================================================================

Use this when bringing the Phase 1 result to the parallel planning
conversation. The foundation docs (`PROJECT_CONTEXT.txt`,
`PROJECT_ROADMAP.txt`, briefs, summary) have been rewritten as if the
current data sources were the original plan, so this note is the
*only* place that records what changed mid-phase.

================================================================
Headline result
================================================================

Working corpus: 1,713 films with all four signals (screenplay, budget,
revenue, rating). That's well above the 1,500-film threshold in the
corpus-size decision criteria, so corpus size is no longer a
constraint on Phase 2.

================================================================
Mid-phase change: ratings source
================================================================

When Phase 1 started, the data plan was MovieSum x TMDB 5000 (the
original Kaggle ~5,000-film dataset). Mid-phase, that turned out to
be the binding constraint on corpus size:

- TMDB 5000 has only ~4,800 films total, ~3,200 with both budget and
  revenue.
- It effectively cuts off at 2016 (1 film in 2017, none after).
  About 21% of MovieSum is post-2016, so those films were
  unmatchable no matter what join strategy we used.
- The MovieSum x TMDB 5000 join, even with a hybrid IMDb-to-TMDB ID
  bridge via MovieLens, capped at 1,019 four-signal films -- in the
  middle band of the corpus-size decision criteria, requiring a
  "document as a limitation" caveat in the report.

Replacement: IMDb-TMDB Movie Metadata Big Dataset (1M) from Kaggle.
~1.07M rows, ~950 MB CSV. Carries `imdb_id` (`tt`-prefixed) AND TMDB
`id` natively, so the join is a direct exact-ID merge -- no fuzzy
matching, no external bridge, no normalization. Year coverage extends
through 2023, recovering the post-2016 era that TMDB 5000 couldn't
supply.

================================================================
Comparison of the two ratings sources
================================================================

(Compared head-to-head before adopting the swap.)

  Metric                          TMDB 5000 (hybrid)   New 1M dataset
  ------------------------------  -------------------  ----------------
  MovieSum match rate             1,130 / 2,188 (52%)  2,186 / 2,188 (99.9%)
  Four-signal working corpus      1,019                1,713
  Year coverage (max)             2016                 2023
  Films only in OLD corpus        --                   0  (strict superset)
  Films only in NEW corpus        --                   +694
  Median budget                   $30M                 $25M
  Median revenue                  $91M                 $64M
  % gross-profitable              85.9%                80.5%

The new dataset is a strict superset: every film we'd have had with
TMDB 5000 is also present, plus 694 additional films (mostly
post-2016, plus older mainstream titles TMDB 5000 simply didn't
include). The slight downward shift in median budget and revenue is
explained by including more recent indie films and films from
under-represented eras that pull the medians down. The drop from 86%
to 80% gross-profitable is welcome: the new corpus is slightly less
survivor-biased, which makes the Phase 6 cost matrix more honest.

================================================================
Foundation-doc state
================================================================

Per user instruction, the foundation docs were rewritten as if the
IMDb-TMDB 1M dataset had always been the ratings source. The
decisions log entry that mentions the swap was removed; the
decisions-log entries from Phase 1 / early Phase 2 are: the
pre-1995 cutoff (2026-05-02 15:30) and its REVERSAL
(2026-05-02 23:35) after a Phase 1 EDA recount found 398
pre-1995 films (not the "~50" originally claimed). The corpus is
1,713 films, year range 1932-2023. This is intentional: the
foundation docs guide execution, this handoff note carries the
strategic context.

The Phase 1 summary (`docs/summaries/phase_1_summary.txt`) is the
authoritative record of Phase 1 work and includes the consolidated
EDA observations + Phase 2 actions checklist. Read it for the full
picture.

================================================================
Phase 2 preprocessing approach
================================================================

Tactical preprocessing decisions are Claude Code's to make based on
the Phase 1 EDA, with two structural requirements: (a) document each
choice and rationale in the Phase 2 summary, and (b) implement the
pipeline as configurable knobs so testing alternatives later is a
one-line change. The preprocessing decisions Claude Code is making:

  Item                               | Phase 2 default            | Knob name (suggested)
  -----------------------------------|----------------------------|----------------------
  Year cutoff (pre-1995 reversed)    | No cutoff (min_year=1900)  | `min_year`
  MovieSum dedup                     | Per filled CSV decisions   | (CSV-driven)
  Drop the 2 unmatched MovieSum films| Drop                       | `require_ratings_match`
  Ratings-dataset dedup              | Keep highest `vote_count`  | `ratings_dedup_strategy`
  Future-year clipping               | Clip to 1900-2025          | `year_range`
  Effective rating column            | IMDB_Rating else vote_avg  | `rating_priority`
  Budget / revenue transform         | Save raw + log1p columns   | `monetary_transform`
  Genre long-tail bucketing          | Bucket genres with N<30    | `genre_min_count`
  Imputation (numeric / categorical) | Defer to Phase 3 (post-split) | (n/a in Phase 2)
  Scaling                            | Defer to Phase 4 (model-specific) | (n/a in Phase 2)
  Outlier handling for $ values      | None; rely on log1p        | `outlier_strategy`

The user reviews the chosen knob values in the Phase 2 summary and
can override anything they disagree with.

================================================================
Strategic items that DO need user input before Phase 2 implements
================================================================

  1. Survivorship-bias treatment. ~80% of the working corpus is
     gross-profitable, well above industry's ~50% net-profitable
     rate. Two options:
       (a) Keep the corpus as-is, document explicitly as a
           limitation in the report, run sensitivity analysis in
           Phase 6 with multiple cost matrices to test robustness.
       (b) Reweight at training time toward an industry base-rate
           profitability distribution.
     Recommendation: (a). (b) requires industry base-rate data we
     don't have at hand. User decision needed before Phase 4.

(All other items previously listed as "open" have been moved into
the Claude-decides table above — they're tactical and the EDA gives
us enough basis to choose defaults.)
