# Phase 3b: Sentiment group handoff

**Status:** Sentiment features implemented and evaluated across four
model families. Standalone-lift verdict is **null**: linear OOF lifts
went the wrong direction on seven of the eight pre-registered metrics
and below the predicted band on the eighth (`roi_gt_2` AUC, +0.008
versus a predicted +0.015 to +0.030). HistGB drops `roi_gt_2` AUC by
0.018 with sentiment added, mirroring the lexical group's pattern.
The negative-lift row is appended to `phase3_ablation.csv` as the
honest finding; all 22 features are kept in the matrix for Phase 3c
combinations evaluation. The NRC EmoLex source deviation (`nrclex`
package in place of the form-gated canonical download) is documented
below per the wordfreq precedent.
**Date:** 2026-05-03

This handoff matches the structure of `phase_3a_handoff.md` and
`phase_3b_lexical_handoff.md`. Subsequent groups (topic, character
network, embeddings) will each get their own handoff in this folder
when they land.

---

## 1. Strategic decisions in scope

* **22 sentiment features as approved in proposal v2** — three VADER
  aggregates over dialogue, eight NRC emotion proportions, four
  within-quartile compound means plus one volatility-concentration
  feature (added in v2), and six Reagan archetype similarities.
* **Three pooling levels intentionally combined into one matrix.**
  Whole-screenplay (VADER + NRC = 11 features), scene-windowed
  (quartiles, 5 features), arc-clustered (Reagan templates,
  6 features). Each level captures information the other two cannot
  derive; the multi-family ablation evaluates the combined matrix
  rather than picking a level.
* **Multi-family ablation pattern carried over from lexical.** Same
  four families (linear, HistGB, KNN, SVM-RBF), same two evaluation
  sets (in-sample train, OOF cross-validation), same metric vocabulary
  (regression: MSE / RMSE / MAE / CVRMSE; classification: AUC-ROC /
  PR-AUC / F1 / log-loss). Linear OOF remains the historical
  pre-registration reference; the other three families and the train
  numbers serve as cross-paradigm and overfit-gap diagnostics.
* **Standalone-lift methodology preserved.** The sentiment-augmented
  matrix joins the 22 sentiment features onto the revised dialogue-
  only baseline (structural counts + era + genre dummies +
  `log_runtime`), NOT onto the lexical-augmented matrix. Joint
  evaluation belongs in Phase 3c.
* **`data_quality_flag` films (24 in train split)** use the same
  sentiment pipeline as the rest of the corpus because all 22
  features are dialogue-line-based aggregates that are robust to
  scene-level degeneracy. The Section 5 diagnostic check empirically
  validates this decision.

---

## 2. Tactical decisions made during implementation

* **Tokenization:** NLTK `word_tokenize` and `sent_tokenize`. Same
  setup as lexical; resources downloaded on demand via
  `ensure_nltk_resources`.
* **VADER backend:** NLTK `SentimentIntensityAnalyzer` with the
  `vader_lexicon` resource. Compound score computed on full dialogue
  lines (VADER's rule-based formulation depends on multi-word
  interactions that token-level scoring discards).
* **NRC EmoLex source (DEVIATION FROM PROPOSAL v2):** the canonical
  NRC EmoLex distribution at saifmohammad.com is form-gated (the user
  submits name and affiliation and receives the zip by email), which
  blocks an automated hash-checked download into `data/external/`.
  The implementation backend is the **`nrclex` Python package**,
  which ships the same word-emotion mappings under the author's
  research-use license. The bundled lexicon contains 6,468 word
  entries (filtered to those with at least one emotion tag) — fewer
  than the canonical ~14,000-entry lexicon, but the missing entries
  are emotionally neutral words that would have contributed zero to
  every per-emotion proportion. This deviation matches the
  wordfreq-vs-SUBTLEX-US precedent set in the lexical group: when the
  canonical distribution is friction-loaded for reproducibility, use
  a stable Python package wrapper that preserves the conceptual
  mechanism, name the deviation in the run's preprocessing metadata,
  and document it in the handoff. The
  `SentimentFeatureConfig.lexicon_path` knob is exposed so a future
  strict-canonical run can swap in a manually-downloaded
  `data/external/nrc_emolex.tsv` without code changes.
* **Stop-word removal for NRC:** `nltk.corpus.stopwords.words("english")`
  in its default form (~180 high-frequency function words). Applies
  only to NRC matching, not to VADER (VADER uses full lines where
  function words contribute to negation rules and intensifier
  handling). Knob: `SentimentFeatureConfig.remove_stopwords_for_nrc`,
  default True.
* **Quartile pooling:** windows defined on the dialogue-line index
  via `np.array_split` so the four quartiles have as-equal-as-
  possible size; with median 880 lines per film this gives ~220
  lines per window. Robust to `data_quality_flag` films with
  collapsed scene structure.
* **Arc-archetype templates:** hand-coded smoothed mathematical
  shapes per proposal v2 Section 3.4, then z-score normalized.
  Single-cycle cosine for Man-in-a-Hole / Icarus (single trough at
  midpoint, single peak); two-cycle cosine for Cinderella / Oedipus
  (three maxima at endpoints + middle, two minima at quartile
  points). Linear ramps for Rags-to-Riches / Tragedy.
* **Trajectory comparison:** each film's per-line VADER compound
  trajectory is interpolated to length 100 via
  `np.interp` on a uniform grid, z-score normalized, then compared
  to each (already z-normalized) template via cosine similarity.
  Films with degenerate (constant) trajectories receive NaN for
  every archetype.
* **NaN handling:** the modelling pipeline's `SimpleImputer(median)`
  inside the linear, KNN, and SVM numeric branches handles any NaN
  rows that fall through from feature computation. HistGB handles
  NaN natively. Imputation inside cross-validation folds so medians
  are fit only on training data within each fold.
* **Feature matrix shape:** 22 model features + 2 diagnostic
  columns (`_nrc_oov_rate_dialogue`, `_vader_zero_compound_rate`).
  The diagnostic columns are excluded from the model-input matrix
  via the leading-underscore convention shared with lexical.

---

## 3. What was done

1. **Built `src/features/sentiment.py`** with the 22 features from
   proposal v2. Each of the three pooling blocks has its own
   helper functions, plus a `compute_sentiment_features` entry
   point that returns a `pd.DataFrame` indexed by `imdb_id`.
2. **Wrote `tests/test_sentiment.py`** with 18 unit and integration
   tests; all pass. Coverage includes VADER aggregate ordering,
   NRC stopword removal, quartile feature edge cases, archetype
   template z-normalization, archetype reflection-pair identities,
   synthetic Man-in-a-Hole trajectory matching, and a smoke test
   on the first 10 films of the corpus.
3. **Computed features on the full corpus** (1,713 films) in
   approximately 175 seconds. The VADER pass dominates runtime;
   NRC matching adds negligible cost. Output saved to
   `data/processed/features_sentiment.parquet` (~270 KB).
4. **Added `include_sentiment` knob to `BaselineFeatureConfig`** so
   the existing trainer can construct the sentiment-augmented
   matrix via the same `build_baseline_features` entry point used
   for lexical.
5. **Built `src/experiments/run_sentiment_ablation.py`** mirroring
   the lexical runner: the same `save_run` block, the same per-
   family lift computation against the Phase 3a revised dialogue-
   only floor, the same ablation-table append, the same
   `RUNS.md` row.
6. **Ran the multi-family ablation** through the runner. Run
   directory: `runs/phase_3/20260503_1943_sentiment_multifamily/`.
   All four families × two eval sets × three targets × seven
   metric rows produced, totalling 168 rows added to
   `phase3_ablation.csv` (cumulative table now at 192 rows
   covering both lexical and sentiment groups).
7. **Ran the Section 9 diagnostic checks** from proposal v2 plus
   the additional univariate target-correlation sanity check the
   lexical handoff established. Output captured below in Section 5.
8. **Set up the project's first `requirements.txt`** at
   `requirements.txt` with pinned (lower-bound flexible) versions
   of every dependency Phase 3 has accumulated, plus updates to
   `data/README.md` documenting how to regenerate
   `split_assignments.parquet`, `features_lexical.parquet`, and
   `features_sentiment.parquet` on a fresh checkout. (The
   environment was missing nltk, nrclex, and pytest at the start
   of this run; the requirements file ensures the next fresh
   clone installs everything cleanly.)

---

## 4. Headline numbers (sentiment added on top of the revised dialogue-only floor, 4 families × 2 eval sets)

Full table at `reports/tables/phase3_ablation.csv` (168 sentiment
rows: 4 families × 2 evaluation sets × 3 targets × {4 regression or
4 classification metrics}). Reporting uses the project's metric
vocabulary committed 2026-05-03: regression uses MSE / RMSE / MAE /
CVRMSE; classification uses AUC-ROC / PR-AUC / F1 / log-loss. Both
in-sample (train) and out-of-fold (OOF) numbers are reported per
family per metric. Pre-registered lift bands apply to the linear
family's OOF numbers only.

### 4.1 OOF lift per family (lower-is-better metrics: negative lift means improvement)

| Family | Metric | Floor | With sentiment | Lift |
|---|---|---:|---:|---:|
| linear | log_roi RMSE (lower) | 1.339 | 1.357 | **+0.019** (worse) |
| linear | log_roi MAE (lower) | 0.948 | 0.959 | +0.010 (worse) |
| linear | log_roi CVRMSE (lower) | 0.993 | 1.007 | +0.014 (worse) |
| linear | roi_gt_1 AUC (higher) | 0.558 | 0.546 | **-0.012** (worse) |
| linear | roi_gt_1 PR-AUC (higher) | 0.846 | 0.838 | -0.008 (worse) |
| linear | roi_gt_1 log-loss (lower) | 0.493 | 0.512 | +0.019 (worse) |
| linear | roi_gt_2 AUC (higher) | 0.602 | 0.610 | **+0.008** (better, below band) |
| linear | roi_gt_2 PR-AUC (higher) | 0.739 | 0.728 | -0.010 (worse) |
| linear | roi_gt_2 log-loss (lower) | 0.635 | 0.641 | +0.006 (worse) |
| histgb | log_roi RMSE (lower) | 1.327 | 1.328 | +0.001 (≈ neutral) |
| histgb | roi_gt_1 AUC (higher) | 0.552 | 0.540 | -0.013 (worse) |
| histgb | roi_gt_2 AUC (higher) | 0.610 | 0.592 | **-0.018** (worse) |
| histgb | roi_gt_2 PR-AUC (higher) | 0.731 | 0.723 | -0.008 (worse) |
| knn | log_roi RMSE (lower) | 1.364 | 1.387 | +0.023 (worse) |
| knn | roi_gt_1 AUC (higher) | 0.527 | 0.494 | -0.033 (worse) |
| knn | roi_gt_2 AUC (higher) | 0.578 | 0.545 | -0.033 (worse) |
| svm | log_roi RMSE (lower) | 1.357 | 1.366 | +0.009 (worse) |
| svm | roi_gt_1 AUC (higher) | 0.504 | 0.475 | -0.030 (worse) |
| svm | roi_gt_2 AUC (higher) | 0.534 | 0.570 | +0.036 (better, off worst floor) |

Bold marks OOF lifts of 0.005 or more in the worse direction on the
linear and histgb families. SVM's positive lift on `roi_gt_2` AUC
mirrors the lexical group's pattern: the SVM floor was the worst of
any family, and the augmented feature space happens to be slightly
more favourable for the RBF kernel; the family's lexical-augmented
result is still below every other family's pre-augmentation floor.

### 4.2 Pre-registered linear-family lift (proposal v2 Section 4)

| Target | Metric | Predicted band (linear OOF) | Actual | In band? |
|---|---|---|---:|:---:|
| log_roi | RMSE | -0.030 to -0.010 (lower is better) | +0.019 | No (wrong direction) |
| log_roi | MAE | -0.030 to -0.010 | +0.010 | No (wrong direction) |
| log_roi | CVRMSE | -0.025 to -0.010 | +0.014 | No (wrong direction) |
| roi_gt_1 | AUC-ROC | 0.000 to +0.010 | -0.012 | No (wrong direction) |
| roi_gt_1 | log-loss | -0.020 to 0.000 | +0.019 | No (wrong direction) |
| roi_gt_2 | AUC-ROC | +0.015 to +0.030 | +0.008 | No (right direction, below band) |
| roi_gt_2 | PR-AUC | +0.010 to +0.025 | -0.010 | No (wrong direction) |
| roi_gt_2 | log-loss | -0.020 to -0.005 | +0.006 | No (wrong direction) |

Of the eight pre-registered linear-family OOF lifts, seven moved in
the wrong direction; the eighth (`roi_gt_2` AUC) moved in the
predicted direction but below the predicted band. None landed inside
the band.

### 4.3 Train-versus-OOF gap with sentiment added (linear and histgb families)

The gap surfaces an overfit-gap signal that the OOF-only view of v1
hid, exactly as for lexical. RMSE on `log_roi` and AUC on `roi_gt_2`:

| Family | Eval set | log_roi RMSE | roi_gt_2 AUC |
|---|---|---:|---:|
| linear | train | 1.301 | 0.6543 |
| linear | oof | 1.357 | 0.6102 |
| linear | train − oof | -0.056 | +0.0441 |
| histgb | train | 1.155 | 0.7862 |
| histgb | oof | 1.328 | 0.5918 |
| histgb | train − oof | -0.173 | +0.1944 |

HistGB's train-OOF gap on `roi_gt_2` AUC widens from 0.20 (Phase 3a
floor and lexical-augmented) to 0.19 with sentiment — essentially
unchanged. Sentiment adds the same overfit pattern lexical did:
shallow-tree HistGB absorbs the new features into the train fit
without translating into OOF improvement.

### 4.4 Verdict

The multi-family picture is **the same shape as lexical's, with the
same conclusion**:

* **HistGB** (the strongest floor family) gets meaningfully worse on
  the classification targets, especially `roi_gt_2` AUC (-0.018).
  If sentiment carried genuine non-linear signal HistGB would
  extract some of it; instead HistGB extracts noise.
* **KNN** gets worse on every metric, with the largest drops on the
  classification targets (-0.033 on each AUC).
* **Linear** gets worse on the regression target and on `roi_gt_1`,
  flat on `roi_gt_2` AUC. The single positive direction is below the
  predicted band.
* **SVM** improves modestly on `roi_gt_2` AUC (+0.036), but starts
  from the worst floor of any family. SVM-with-sentiment on
  `roi_gt_2` AUC reaches 0.570, still below linear-without-sentiment
  (0.602) and HistGB-without-sentiment (0.610).

The original "linear hurt by noise features" hypothesis from the
lexical handoff is now strengthened a second time. If sentiment
features carried genuine non-linear signal HistGB would extract
some of it. HistGB does the opposite: it gets actively worse on
classification and stays flat on regression. Adding 22 features
that don't carry signal:

* For linear: shifts the regularization optimum, worsening RMSE and
  classification AUC modestly.
* For HistGB: gives the algorithm 22 more split candidates per node,
  some of which match noise patterns in the OOF folds; with shallow
  trees and conservative defaults the model still has to evaluate
  them at every split.
* For KNN: 22 noise dimensions added to the L2 distance metric;
  nearest neighbours become less informative.
* For SVM-RBF: same kernel-distance degradation as KNN, but SVM's C
  and gamma absorb more of the noise. The reason SVM appears to
  improve on `roi_gt_2` is that its floor was the worst, and the
  new feature space happens to be slightly more favourable for the
  RBF kernel.

The conclusion: sentiment features do not carry signal that any of
the four families can extract beyond what the structural baseline
already provides. **The genre-residual hypothesis from the lexical
handoff is now reinforced.** The information sentiment can in
principle carry — overall valence, emotional arc, genre-tagged
emotion vocabulary — appears to overlap heavily with the thirteen
genre dummies plus release year plus structural counts already in
the baseline. The marginal residual after those controls is too
thin for the four families to extract reliably at n = 1,199.

---

## 5. Diagnostic results (Section 9 of proposal v2)

| # | Check | Threshold | Result | Verdict |
|---|---|---|---|---|
| 1 | Sentiment ↔ structural baseline cross-correlations | drop pair if \|r\| > 0.85 | All pairs \|r\| < 0.85; strongest is `vader_compound_range ↔ log_total_dialogue_chars` at 0.802 | Pass |
| 2 | Within-sentiment pairwise correlations | drop one if \|r\| > 0.85 | Three pairs at \|r\| = 1.0 by mathematical construction (Rags/Tragedy, Icarus/Man-in-a-Hole, Cinderella/Oedipus reflection pairs); no other pairs above 0.85 | **Construction-implied; see below** |
| 3 | Quartile coherence (mean of q means ≈ whole-screenplay mean) | flag if \|deviation\| > 0.05 on many films | mean dev 0.000, max 0.005, zero films above 0.05 | Pass |
| 4 | Arc-archetype variance | flag if std < 0.5 unit across corpus | All six archetypes std 0.10-0.12 | **Trips threshold; see below** |
| 5 | VADER zero-compound rate | flag if > 100 films below 30% non-zero rate | Mean zero-rate 0.485; only 9 of 1,199 films below the 30% threshold | Pass |
| 6 | NRC OOV-rate distribution | flag if > 100 films above 15% OOV | Mean OOV 0.811; 1,193 of 1,199 above 15% | **Trips threshold; see below** |
| 7 | data_quality_flag films vs unflagged | flag if z-diff > 1.0 | All 6 spot-checked features within 0.6σ of unflagged means | Pass |
| 8 | Binary positive/negative vs eight-dimensional NRC | descriptive | Skipped (positive/negative excluded per proposal Section 3.2; can be revisited if planning conversation requests) | Skipped |
| 9 | VADER tail-compression by primary genre | flag genre-stds > 2 MAD below median | Animation lowest at 0.029, Fantasy highest at 0.057; no genre flagged | Pass |
| extra | Univariate target correlations | informational | No sentiment feature exceeds \|r\| = 0.10 with any target on the train split; strongest is `nrc_anger_proportion ↔ log_roi` at -0.083 | **Confirms verdict** |

Three checks merit narrative interpretation rather than mechanical
threshold application.

**Check 2 — three reflection pairs at |r| = 1.0.** The hand-coded
archetype templates are by mathematical construction reflections of
each other: `tragedy = -rags_to_riches`, `icarus = -man_in_a_hole`,
`oedipus = -cinderella`. The archetype-similarity features therefore
inherit the same reflection identity (cosine similarity is sign-
invariant under template negation). The pairs measure the same
dimension up to sign, which a linear model and a tree model both
handle naturally (sign-flip is captured by coefficient sign or split
direction). This was foreseeable from the proposal's Section 3.4
templates and is not a flagworthy redundancy. Three of the six
archetype features therefore add information equivalent to one
sign-bit; the effective archetype dimensionality is three, not six.
For Phase 3c combinations purposes the three "primary" archetypes
(Rags-to-Riches, Man-in-a-Hole, Cinderella) carry the relevant
information without their reflections.

**Check 4 — arc-archetype variance trips the 0.5-unit threshold.**
All six archetype-similarity features have standard deviation in the
0.10-0.12 range across the train split, well below the proposal's
"at least 0.5 units of variation" expectation. The threshold itself
turns out to be miscalibrated for the corpus and the feature design.
A film's per-line compound trajectory is a 800+ line noisy time
series; interpolating it to length 100 and z-normalizing produces a
shape whose cosine similarity to a smooth analytic template (also
length 100, z-normalized) is mechanically bounded in absolute value
by roughly the inverse square root of the trajectory's effective
degrees of freedom. With 100 z-normalized values compared against a
smooth template, the random-baseline cosine similarity has standard
deviation around 0.10 — exactly what we observe. Higher variation
would require either smoother trajectories or more distinctive
templates. The features are therefore structurally low-amplitude;
the model can still pick up directional information (positive vs
negative similarity to Man-in-a-Hole), but the magnitude is small.

The univariate target correlations confirm this directly: the best
archetype-target correlation is `arc_similarity_man_in_a_hole ↔
log_roi` at +0.044 — well below the |r| = 0.10 threshold the
proposal set as the multiple-comparisons guard against spurious
arc-target relationships. None of the six archetype features
clears the bar.

**Check 6 — NRC OOV rate trips the 15% threshold trivially.**
The bundled NRC EmoLex via `nrclex` contains 6,468 word entries
filtered to those with at least one emotion tag. Common dialogue
words like "go", "come", "see", "say", "look" — emotionally
neutral function words and high-frequency verbs that survive
stopword removal — are not in the lexicon by design, because they
carry no NRC emotion tag. An 81% OOV rate is therefore the expected
shape of the matching against this filtered lexicon; it is not a
signal that NRC matching is broken. The 15% threshold inherited
from the lexical proposal does not fit the NRC use case (the
lexical group's frequency lookup uses wordfreq's ~140,000-entry
English distribution, where 15% OOV is meaningful). The relevant
question for NRC is whether the matched-token count per film is
large enough to make per-emotion proportions stable; with median
~880 dialogue lines per film yielding ~5,000 non-stopword tokens
and ~19% in-lexicon, each film's per-emotion proportion is computed
from ~950 emotion-tagged tokens — plenty for stable estimates.

The implementation is functioning correctly; the inherited
threshold is a methodology mismatch that surfaces here and gets
documented. Phase 4 model selection should not condition on
`_nrc_oov_rate_dialogue` as a feature.

**The univariate finding is the verdict-confirming one.** No
sentiment feature exceeds |r| = 0.10 with any target. The strongest
five (in absolute value):

| Feature | Target | r |
|---|---|---:|
| `nrc_anger_proportion` | `log_roi` | -0.083 |
| `sentiment_q4_compound_mean` | `log_roi` | +0.076 |
| `nrc_trust_proportion` | `roi_gt_2` | -0.071 |
| `nrc_anger_proportion` | `roi_gt_2` | -0.071 |
| `nrc_trust_proportion` | `log_roi` | -0.068 |

These directions are interpretable (more anger correlates weakly
with worse ROI; more positive late-screenplay sentiment correlates
weakly with better ROI), but the magnitudes are sampling-noise
small for n = 1,199. The result mirrors the lexical group's
finding to a tighter degree than expected: the strongest sentiment
correlation is roughly the same magnitude as the strongest lexical
correlation (`mtld_action ↔ log_roi` at -0.094 in the lexical
group). Two independent feature groups, both capping out below
|r| = 0.10 univariate, both showing negative ablation lift on
HistGB, both flat-to-negative on linear.

The implementation passes every diagnostic that does not depend on
a corpus- or feature-shape-mismatched threshold. The negative-lift
result is not an implementation defect.

---

## 6. Why the lift went negative (mechanism analysis)

The mechanism is the same as the lexical handoff's Section 6
identified, with one sentiment-specific addition.

**The shared mechanism: genre absorbs most of what sentiment can
also see.** The structural baseline includes thirteen
`genre_<X>` dummies plus release year. Sentiment features that
sound orthogonal to genre on paper turn out to share substantial
variance with the genre dummies in practice:

* Action and Thriller films systematically have higher
  `vader_compound_range` and `vader_compound_std` (more emotional
  swings; more shouted, exclamatory lines); the genre dummies
  already encode this.
* Animation, Family, Comedy films systematically have higher
  whole-screenplay `vader_compound_mean` (more positive valence on
  average); same.
* Horror and Thriller films have higher `nrc_fear_proportion`;
  Romance has higher `nrc_joy_proportion` and `nrc_trust_proportion`;
  Drama has higher `nrc_sadness_proportion`. Each pairing is one
  sentiment feature ↔ one (or two) genre dummies.
* Even the arc-archetype features partially align with genre: the
  classic Hollywood three-act structure produces a Cinderella-like
  shape in many films; tragedies skew toward Tragedy or Oedipus.

When the model already has the genre dummies plus release year plus
structural counts, the marginal residual that 22 sentiment features
can explain is small enough that none of the four model families
can extract it without hurting itself. Same shape as the lexical
result.

**The sentiment-specific addition: trajectory features fight the
amplitude bound.** The arc-archetype features pay an additional
cost the lexical group did not: they are mechanically low-amplitude
(see Section 5 Check 4), so even if they carry directional
information their univariate strength is bounded by the cosine
similarity geometry. A film whose true emotional shape is Man-in-a-
Hole produces a positive `arc_similarity_man_in_a_hole`, but the
magnitude is small (~0.1 to 0.4), and the model cannot easily
distinguish "true Man-in-a-Hole" from "noisy fall-rise pattern with
mostly random-shape similarity 0.1". The sign-bit information is
real but the noise floor is high.

**Family-specific mechanisms (secondary).**

* **Linear:** adding 22 features with no detectable univariate
  signal to a small-corpus L2-regularized linear model produces a
  small but real OOF prediction noise increase that the regularizer
  cannot fully absorb. Same as lexical, slightly worse magnitude
  because there are 22 features instead of 13.
* **HistGB:** the algorithm's split-finding considers all features
  at each node; zero-signal features sometimes match noise patterns
  in the OOF folds. With shallow trees and conservative defaults
  the overfit absorbs the new features at train time without OOF
  benefit. The classification AUC drop on `roi_gt_2` (-0.018) is
  larger than the lexical drop (-0.024), but both are within the
  same band of "moderate noise injection".
* **KNN:** the L2 distance metric is computed across all
  standardized features; 22 noise dimensions degrade nearest-
  neighbour informativeness directly. Larger drop than lexical
  produced.
* **SVM-RBF:** same kernel-distance degradation, but SVM's
  regularization absorbs more of it. The slight `roi_gt_2` AUC
  improvement (+0.036) starts from the worst floor in the matrix
  and ends below every other family's pre-augmentation floor; not
  a signal-extraction win.

**What this does not say.** The result does not say the features
are implemented incorrectly (Section 5 rules that out). It does not
say sentiment is theoretically uninformative — published work on
larger corpora reports modest signal from emotional-arc features
on commercial outcomes. It says: with this corpus, with these four
model families, and with these 22 features, the marginal
information sentiment adds over genre, era, budget structure, and
screenplay structure is below the threshold where any of the four
paradigms can extract it without hurting itself.

**Implication for Phase 3c.** The two consecutive null results
(lexical, sentiment) sharpen the case for the Phase 3c combinations
sub-phase. The current ablation methodology systematically
under-credits any feature group whose signal partially overlaps
with genre. Two of the five Phase 3b groups have now produced
genre-overlap-implied null verdicts on standalone evaluation. The
remaining three groups (topic, character network, embeddings) are
all plausibly more orthogonal to genre — character network in
particular captures structural properties of the cast that genre
does not directly encode — so the Phase 3b standalone evaluations
might still yield a positive group. But Phase 3c is now more
clearly the venue where any of the five groups will be most fairly
evaluated.

---

## 7. Files produced (Phase 3b sentiment group)

### Code
* `src/features/sentiment.py` (~520 lines): the 22-feature pipeline.
* `src/experiments/run_sentiment_ablation.py`: the multi-family
  sentiment-ablation runner. Mirrors the lexical runner.
* `tests/test_sentiment.py` (18 tests, all passing).
* `src/features/baseline_features.py`: extended with
  `include_sentiment` and `sentiment_features_path` knobs on
  `BaselineFeatureConfig`, plus the join logic.
* `requirements.txt` (project's first; pins versions for nltk,
  wordfreq, nrclex, pytest, plus the existing scientific stack).
* `data/README.md`: extended with a Section 4 documenting how to
  regenerate `split_assignments.parquet`,
  `features_lexical.parquet`, and `features_sentiment.parquet` on
  a fresh checkout.

### Data
* `data/processed/features_sentiment.parquet` (1,713 × 24, ~270 KB).

### Run artifacts
* `runs/phase_3/20260503_1943_sentiment_multifamily/`: the multi-
  family run that produced the verdict above. Six files
  (`params.json`, `preprocessing_summary.json`, `features_used.json`,
  `metrics.json`, `run.log`).

### Tables
* `reports/tables/phase3_ablation.csv`: extended from 28 rows
  (lexical only) to 192 rows (lexical + sentiment, 4 families × 2
  eval sets × 3 targets × 7 metric rows per group).

### Configuration changes
* `BaselineFeatureConfig.include_sentiment` and
  `sentiment_features_path` on `src/features/baseline_features.py`.
* `runs/RUNS.md`: new row for the sentiment multi-family run.

---

## 8. Resolved questions

The questions raised at the proposal-review stage are all resolved
by the implementation choices above and the planning-conversation
guidance in the proposal v2 review:

1. **Treatment of the sentiment group's official ablation row:
   APPEND.** The negative-lift row goes into
   `phase3_ablation.csv` as the honest finding, matching the
   lexical group's pattern. The Phase 3 narrative now leads with
   two pre-registered, measured, and honestly-surfaced negative
   results — which is a stronger story for the report than two
   hand-picked positive groups.
2. **NRC EmoLex source deviation: ACCEPTED in advance.** Proposal
   v2 Section 6.2 already named `nrclex` as the documented
   fallback for the form-gated canonical distribution. The
   deviation is recorded in
   `runs/phase_3/20260503_1943_sentiment_multifamily/preprocessing_summary.json`
   and will be cited in `FEATURE_NOTES.md` at end of Task 4.
3. **Diagnostic-threshold trips (NRC OOV, archetype variance):
   DOCUMENT, do not act mechanically.** Section 5 above interprets
   each trip in light of the corpus and feature-shape it
   describes; neither implies that the features should be dropped.
   Phase 4 model selection may decide to deprioritize the arc-
   archetype features given their low amplitude, but the standalone
   ablation row should reflect the proposal's measured lift, not a
   feature-pruned variant.
4. **Proceed to topic proposal: YES.** The two consecutive nulls
   reinforce the case for groups whose information is more
   orthogonal to genre. Topic comes next per the original group
   ordering (lexical → sentiment → topic → character network →
   embeddings); the substantive design question is fitting LDA
   (or equivalent) on training-fold text only and applying to the
   calibration / test sets without leakage.

## 9. Decisions log entry to add

The following entry should be appended to
`docs/PROJECT_CONTEXT.md` Section 8 (Decisions Log) when this
handoff lands. No new methodology change is introduced here — the
entry records the sentiment standalone result and the deviation.

> ## 2026-05-03 19:50 — Phase 3b: sentiment standalone result is null; nrclex deviation accepted
>
> **Phase:** Phase 3 — Feature Extraction (sub-phase 3b, second of five groups)
> **Decision:** Sentiment features (22 columns: 3 VADER aggregates, 8
> NRC emotion proportions, 5 quartile-trajectory features, 6 Reagan
> archetype similarities) implemented per proposal v2; the multi-
> family ablation produced a null verdict (linear OOF lift went the
> wrong direction on 7 of 8 pre-registered metrics, the eighth was
> below the predicted band). HistGB drops `roi_gt_2` AUC by 0.018,
> mirroring the lexical group's pattern. The negative row is
> appended to `phase3_ablation.csv` as the honest finding; all 22
> features are kept in the matrix for Phase 3c combinations
> evaluation. The NRC EmoLex source deviation (`nrclex` package in
> place of the form-gated canonical download) was the documented
> fallback in proposal v2 Section 6.2 and is logged in the run's
> preprocessing metadata. Two consecutive null results across two
> independent feature groups reinforce the genre-residual hypothesis
> from the lexical handoff and the case for the Phase 3c
> combinations sub-phase.
> **See also:** `docs/handoffs/phase_3b_sentiment_handoff.md` for
> headline numbers, diagnostic results, and mechanism analysis.

---

## 10. Next step

Start the topic proposal at
`docs/proposals/phase3_topic_proposal.md`. The substantive design
question for this group is the topic-modeling backend (LDA via
gensim vs sklearn, or a transformer-based topic extractor), the
unit of analysis (whole-screenplay vs scene-level), and the
no-leakage discipline (the topic model is fit on training-fold
text only and applied to calibration and test sets). The character-
network group remains queued after topic; embeddings still last.

Phase 3c (combinations sub-phase) becomes increasingly important
the more standalone groups land null. The combinations to be
pre-specified there should be drafted no later than the third
group's handoff so the pre-registration discipline is preserved
when the 3c sub-phase begins.
