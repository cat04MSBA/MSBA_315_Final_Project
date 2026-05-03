# Phase 3 Sentiment Feature Proposal (v2)

**Group:** Sentiment (2 of 5 in the Phase 3b incremental ablation)
**Status:** Awaiting planning-conversation final sign-off
**Date:** 2026-05-03
**Version:** 2 (incorporates planning-conversation review of v1, plus
the cross-cutting metric-vocabulary change to MSE / RMSE / MAE /
CVRMSE for regression and AUC-ROC / PR-AUC / F1 / log-loss for
classification, with both train and OOF values reported)

This is the revised sentiment-feature proposal. v1 was reviewed by
the planning conversation on 2026-05-03 with two required
corrections (math error in Section 9 #3; lemma versus token-form
language in Section 3.2), two required additions (concrete
specification of the Reagan archetype templates; explicit scoping
of sentiment-on-action-text), and several polish items. v2 folds
all corrections, additions, and polish guidance into the design.
The proposal also adopts the new metric vocabulary committed
2026-05-03 across the project: regression metrics are MSE, RMSE,
MAE, and CVRMSE; classification metrics are AUC-ROC, PR-AUC, F1,
and log-loss. Both in-sample (train) and out-of-fold (OOF)
evaluation values will be reported for every metric.

---

## 1. Why sentiment second

Sentiment sits second in the Phase 3b ablation queue for three
reasons.

**It shares preprocessing with lexical and is therefore
inexpensive to add.** The same NLTK word and sentence tokenizers
and the same empty-text dialogue filter that the lexical group
established carry over directly. The sentiment computation itself
adds a VADER pass and an NRC lexicon lookup; both are cheap.

**It captures information lexical does not.** The lexical group's
near-null verdict was attributed to genre absorbing most of the
between-film variance lexical aggregates can explain. Sentiment
should be more orthogonal to genre: a romantic comedy with a "Man
in a Hole" arc and a romantic comedy with a steady-rise arc may
have similar lexical profiles but very different sentiment
trajectories.

**Box-office literature has documented signal at this level.**
Reagan et al. 2016 identified six recurring emotional arc shapes
in narrative text. The "Man in a Hole" trajectory (a fall-rise
shape, where emotional valence drops in the middle of the story
and recovers by the end) has been reported as the strongest single
sentiment-derived predictor of commercial success. The pre-
registered expectation in Section 4 treats arc-shape features as
the strongest contributors and the within-screenplay aggregates as
the floor.

---

## 2. The substantive design question: pooling strategy

Sentiment is unusual among the five Phase 3b groups because the
same underlying signal (per-utterance VADER and per-token NRC
emotion tags) can be aggregated at three different levels of
abstraction, each capturing different information. The pooling
choice is the substantive design question for this group, the same
way pooling will be the substantive question for the embedding
group.

### 2.1 Three pooling levels

**Whole-screenplay aggregation.** Compute one number per film by
aggregating all dialogue lines or all dialogue tokens. Examples:
mean compound VADER score across all dialogue lines, proportion of
dialogue tokens tagged with the NRC `joy` emotion. Loses temporal
structure entirely. Robust to noise, simple to interpret, low
dimensionality.

**Scene-windowed aggregation.** Divide the screenplay into a fixed
number of contiguous windows (typically by dialogue-line index
rather than by scene number, to avoid sensitivity to the per-film
scene count). Compute the same aggregate within each window.
Per-film features become a sequence of within-window means plus
summary statistics over the sequence. Captures temporal structure
but introduces a window-count knob.

**Arc-clustered aggregation.** Compute the per-window sentiment
trajectory at a fixed length (say 100 normalized positions),
compare against canonical arc templates from Reagan et al. 2016,
and use the similarity scores against each template as features.
Captures the literature-validated archetype shape but loses
fine-grained variation.

### 2.2 Why use all three rather than picking one

Each pooling level preserves different information, and the three
are not redundant in principle. Whole-screenplay tells the model
"this film has high overall positive valence." Scene-windowed
tells it "this film's valence rises sharply between Q3 and Q4."
Arc-clustered tells it "this film is best classified as a Man in
a Hole." A linear or tree-based model receiving all three can
pick which level of abstraction is most informative for each
target. None of the four model families used in the multi-family
ablation can derive the windowed or clustered representations
from the whole-screenplay representation alone, so omitting any
level closes off information.

### 2.3 What this proposal commits to

All three pooling levels are computed and combined into a single
sentiment-feature matrix. Specifically:

* Three VADER aggregates and eight NRC emotion proportions form
  the whole-screenplay block (11 features).
* Four within-quartile compound-score means and one within-
  quartile-volatility-concentration feature form the scene-
  windowed block (5 features). The volatility feature was added
  in v2 in response to planning-conversation polish guidance:
  within-quartile standard deviations are not linear combinations
  of the within-quartile means, and tree-based models cannot
  derive volatility-concentration from the means alone.
* Six similarities to canonical Reagan archetypes form the
  arc-clustered block (6 features).

Total: 22 features. Section 3 lists them with rationale.

---

## 3. Proposed features (22 total)

### 3.1 VADER aggregates (3 features, whole-screenplay)

VADER (Valence Aware Dictionary and sEntiment Reasoner; Hutto and
Gilbert 2014) is a rule-based sentiment scorer designed for short
informal text. It returns a `compound` score in `[-1, 1]` that
captures overall sentiment polarity, plus separate `pos`, `neg`,
and `neu` proportions of each utterance's emotional content.

**Calibration caveat.** VADER is calibrated on social-media text
where sentiment tends to be loud and lexically marked
("I love this!", "this is the WORST"). Screenplay dialogue is
often crafted and subtler: period dramas, courtroom thrillers,
and literary adaptations carry emotional weight via context
rather than via sentiment-laden vocabulary. The compound mean is
robust in expectation, but tail behaviour and per-line dynamics
may compress on these films. The diagnostic step in Section 9
checks whether the compound-score distribution is suspiciously
narrow on subgenres where this is a concern; the feature is
included because relative ordering across films should still
carry signal. Same caveat-class as Flesch-Kincaid on fragmented
dialogue.

**`vader_compound_mean`. Mean compound score across all dialogue
lines.** Captures overall emotional valence.

**`vader_compound_std`. Standard deviation of compound scores
across dialogue lines.** Captures emotional volatility.

**`vader_compound_range`. Maximum minus minimum compound score
across dialogue lines.** A coarser tail-aware measure of
emotional dynamic range than the standard deviation. Captures
whether a film ever reaches extreme valence even briefly.

These three are the floor the planning conversation specified
(`mean and std as the floor`); the range is the additional
tail-aware feature.

### 3.2 NRC emotion proportions (8 features, whole-screenplay)

The NRC Word-Emotion Association Lexicon (Mohammad and Turney
2013) annotates approximately 14,000 English word forms with the
eight basic emotions of Plutchik's wheel: `anger`, `anticipation`,
`disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`. The
multi-dimensional treatment is preferred over a binary
positive-versus-negative split because the latter can hide a
meaningful distinction. A horror film and a sad drama may both
have high "negative" scores but the horror film's negative content
is dominated by `fear` and `disgust` while the drama's is
dominated by `sadness` and (often) `trust`.

**Token-form matching, not lemmatization.** NRC EmoLex is indexed
by surface word forms, not lemmas: both `happy` and `happiness`
appear as separate entries with their own emotion tags. The
tokenizer commitment carried over from the lexical group is NLTK
`word_tokenize` with no lemmatization step, which produces token
forms that match NRC's indexing convention directly. Lemmatization
would actually reduce match quality by collapsing forms NRC
treats independently. (v1 of this proposal incorrectly described
the matching as lemma-based; v2 corrects to token-form matching.)

**Stop-word removal: NLTK English stopwords list.** Stop words
(approximately 180 high-frequency English function words such as
`the`, `is`, `at`, `which`, `you`) are removed before NRC
matching to avoid skewing the per-emotion proportions toward
mechanical occurrence counts of words that NRC tags weakly or not
at all. The NLTK English stopwords list is used in its default
form; the configuration knob in
:class:`SentimentFeatureConfig` permits a no-removal variant for
sensitivity analysis.

**Eight features, one per emotion:** `nrc_anger_proportion`,
`nrc_anticipation_proportion`, `nrc_disgust_proportion`,
`nrc_fear_proportion`, `nrc_joy_proportion`,
`nrc_sadness_proportion`, `nrc_surprise_proportion`,
`nrc_trust_proportion`. Each is computed as the proportion of
non-stopword dialogue token forms whose surface form appears in
the NRC lexicon and is annotated with the given emotion. Multiple
emotions per word are allowed (`fear` and `surprise` co-tag many
horror words); each tag contributes once.

A note on a potential bias: VADER's binary positive/negative
breakdown has been documented to over-count positive sentiment in
casual text by ten to thirty percent relative to held-out
ground-truth annotations. The eight-dimensional NRC representation
side-steps this bias by reporting the emotion proportions
independently rather than as a positive-negative ratio. The
diagnostic step in Section 9 checks whether the NRC features add
information beyond the VADER aggregates or are largely redundant.

### 3.3 Sentiment quartile-trajectory features (5 features, scene-windowed)

Each screenplay's dialogue is divided into four equal-sized
contiguous windows by dialogue-line index (Q1 through Q4). The
window boundaries are deliberately defined on the dialogue-line
sequence rather than on scene number to avoid sensitivity to the
per-film scene count and to the `data_quality_flag` issue with
collapsed scene structure: a film with two scenes containing all
its dialogue still has well-defined dialogue-line quartiles.

**Four within-quartile compound means:**
`sentiment_q1_compound_mean`, `sentiment_q2_compound_mean`,
`sentiment_q3_compound_mean`, `sentiment_q4_compound_mean`. Each
is the mean of the VADER compound scores for dialogue lines
falling in that quartile of the film. Together they describe the
per-film sentiment trajectory at the coarsest meaningful
resolution. Slope, curvature, and other trajectory derivatives
are linear combinations of these four and are not separately
included as features.

**`sentiment_volatility_concentration`. Within-quartile
volatility concentration** (added in v2 per planning-conversation
guidance). Defined as the maximum minus the minimum of the four
per-quartile standard deviations of VADER compound scores. A
film where emotional volatility concentrates in one quartile (a
late climax, for example) has a high value; a film where
volatility is uniform across quartiles has a low value. Crucially,
this feature is not a linear combination of the four per-quartile
means (it is a function of the per-quartile second moments),
which means tree-based models can extract it but linear-model-
derived features cannot. The single-feature variant is preferred
over four separate per-quartile standard deviations on parsimony
grounds; the diagnostic step will surface whether this single
summary captures the volatility-pattern signal adequately.

The choice of four windows rather than ten or twenty is
deliberate. With a median of 880 dialogue lines per film, four
windows give 220 dialogue lines per window on average, which is
enough for the per-window mean and standard deviation to be
stable.

### 3.4 Reagan arc archetype similarities (6 features, arc-clustered)

Reagan, Mitchell, Kiley, Danforth, and Dodds (2016) identified six
recurring emotional arc shapes in approximately 1,300 English-
language fiction works. Each shape is a normalized sentiment
trajectory at fixed length. The six archetypes:

* **Rags to Riches:** monotonic rise.
* **Tragedy (Riches to Rags):** monotonic fall.
* **Man in a Hole:** fall then rise. Reagan et al. report this
  is one of the most common shapes in film and is associated
  with positive commercial reception.
* **Icarus:** rise then fall.
* **Cinderella:** rise, fall, rise.
* **Oedipus:** fall, rise, fall.

**Source of the archetype templates: hand-coded smoothed
mathematical shapes** (planning-conversation directive in v2
review). The original published Reagan templates were derived via
SVD on a 1,300-novel corpus; methodological replications have
shown the SVD-derived components are sensitive to corpus
composition, which would introduce a defensibility hole if we
re-derived the templates on a corpus of our choosing. Reading
shape coefficients from Reagan et al.'s supplementary materials
is the alternative; we choose hand-coded templates instead
because they are interpretable, reproducible, not subject to
overfit-the-archetype-corpus concerns, and the trajectories are
coarse enough at length 100 (the cosine-similarity comparison
length) that hand-coded smoothed shapes capture the literature-
defined qualitative meaning of each archetype adequately.

**Mathematical form of each template** at length 100 with index
`t` running from 0 to 99:

* `Rags to Riches`: `T_rr(t) = -1 + 2 * t / 99`. Linear ramp from
  -1 to +1.
* `Tragedy`: `T_tr(t) = +1 - 2 * t / 99`. Linear ramp from +1 to
  -1 (reflection of the above).
* `Man in a Hole`: `T_mh(t) = +1 - 2 * cos((t - 49.5) / 99 * pi)`.
  Cosine wave with a single trough centered at t = 49.5;
  evaluates to roughly +1 at the ends and -1 at the middle.
* `Icarus`: `T_ic(t) = -T_mh(t)`. Reflection: cosine wave with a
  single peak at t = 49.5.
* `Cinderella`: two-trough shape:
  `T_ci(t) = sin(2 * pi * (t - 33) / 66)` smoothed over the
  trajectory with two minima near t = 25 and t = 75 and three
  maxima near t = 0, 50, and 99. Implementation: a rescaled and
  shifted three-cycle cosine.
* `Oedipus`: `T_oe(t) = -T_ci(t)`. Two-peak shape (reflection of
  Cinderella): two maxima near t = 25 and t = 75 and three
  minima near t = 0, 50, and 99.

All templates are then z-score normalized before similarity
computation so the cosine similarity is invariant to amplitude
differences between the film's trajectory and the template.

**Six similarity features:** `arc_similarity_rags_to_riches`,
`arc_similarity_tragedy`, `arc_similarity_man_in_a_hole`,
`arc_similarity_icarus`, `arc_similarity_cinderella`,
`arc_similarity_oedipus`. Each is a continuous score in
`[-1, 1]`. A film closely matching the Man-in-a-Hole template
scores near 1 on `arc_similarity_man_in_a_hole` and lower (often
negative) on the inverse-shape `arc_similarity_icarus`.

Each film's per-line VADER compound trajectory is interpolated to
length 100, z-score normalized, then compared via cosine
similarity to each of the six normalized templates above. The
diagnostic step will verify that a synthetic Man-in-a-Hole-shaped
input scores highest on `arc_similarity_man_in_a_hole`.

---

## 4. Pre-registered expected lift

Predictions made before implementation. After implementation, the
actual lift over the Phase 3a revised dialogue-only floor is
recorded in `phase3_ablation.csv` for both the train (in-sample)
and OOF (out-of-fold) evaluation sets. The pre-registered bands
below apply to the linear family's OOF numbers, which are the
historical reference for the proposal's lift predictions.

The metric vocabulary used here matches the project's updated set
(MSE, RMSE, MAE, CVRMSE for regression; AUC-ROC, PR-AUC, F1,
log-loss for classification). The lift bands previously expressed
in R-squared have been translated to RMSE on the same floor.

### Regression target `log_roi`

Predicted lift in OOF metrics (linear family):

* **RMSE: -0.030 to -0.010** (lower is better; reduction band).
* **MAE: -0.030 to -0.010**.
* **CVRMSE: -0.025 to -0.010**.

The translation: a +0.015 to +0.040 R² lift on a floor of R² ≈
0.052 with var(y) ≈ 1.79 corresponds to RMSE moving from 1.339
toward roughly 1.31 to 1.32 (a 0.02 to 0.03 reduction). v1
predicted +0.015 to +0.040 R² which the planning-conversation
review flagged as optimistic; v2 narrows the upper bound so the
band is more honestly calibrated. The corresponding RMSE lift
band is -0.030 to -0.010, with -0.030 already representing a
substantial result on n = 1,199.

Mechanism: VADER aggregates and NRC emotion proportions provide
incremental signal about emotional tone that genre and structural
counts do not fully absorb. The arc-archetype features add
temporal shape that none of the other groups capture.

### Classification target `roi_gt_1` (gross-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: 0.000 to +0.010**.
* **PR-AUC: -0.005 to +0.010**.
* **F1: 0.000 to +0.005** (target dominated by the 80% positive
  base rate; F1 saturates near 0.89 across families).
* **log-loss: -0.020 to 0.000** (reduction band).

Mechanism: as with lexical, the 80%-positive base rate makes
this target signal-thin. Sentiment may help slightly with
identifying the unprofitable minority but the available headroom
is small.

### Classification target `roi_gt_2` (net-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: +0.015 to +0.030**.
* **PR-AUC: +0.010 to +0.025**.
* **F1: 0.000 to +0.015**.
* **log-loss: -0.020 to -0.005** (reduction band).

The AUC upper bound is narrowed from v1's +0.035 to +0.030 in
response to the planning-conversation polish guidance: v1's bands
were on the optimistic side relative to what Reagan-style
literature reports on much larger corpora.

Mechanism: this target carries the most headroom and is the most
likely beneficiary of the arc-archetype features. Blockbuster
narrative shapes (Man in a Hole, Cinderella) cluster around
commercial success in published work.

### Combined expectations relative to forward bands

The forward expectations for the full Phase 3b ablation are
`log_roi` RMSE reduction of 0.05 to 0.10 (equivalent to the older
R² band of 0.10 to 0.20) and `roi_gt_2` AUC in the 0.65 to 0.72
band. Sentiment is expected to contribute roughly a quarter to a
third of the remaining lift on top of the lexical-group null. If
sentiment also lands near zero on linear OOF, that is itself a
finding to discuss before topic, given that the genre-residual-
signal hypothesis from the lexical handoff explicitly predicts
sentiment should fare better.

---

## 5. Acknowledgement of out-of-scope features (defensibility)

Four classes of sentiment-related features are considered and
explicitly deferred.

### 5.1 Sentiment on action text

The lexical group computed MTLD and Flesch-Kincaid on dialogue
text and on action text in parallel, and the diagnostic step
showed the action-versus-dialogue distinction was paying for
itself dimensionally (mtld pair r = +0.49, F-K pair r = +0.28,
both well below the 0.85 collapse threshold). The sentiment
proposal scopes everything to dialogue. Two classes of action-
text sentiment are considered:

**VADER on action text: scoped out** for domain-mismatch reasons.
VADER is calibrated on conversational text where sentiment is
expressed via direct lexical markers ("love", "hate", "amazing",
"terrible") and via punctuation/intensifier rules. Action text
is descriptive prose ("She watches the rain. The clock strikes.")
and the emotional content is conveyed through scene-level
imagery rather than through the utterance-level lexical and
syntactic cues VADER's rule-based formulation depends on. The
expected signal-to-noise ratio for VADER on action is poor.

**NRC on action text: scoped out for parsimony, defensible future
addition.** The NRC lexicon is word-level and domain-portable:
the word `darkness` has the same `fear`-`sadness` association
whether it appears in dialogue or action description. Computing
NRC emotion proportions on action text is implementable at low
cost (the same NRC lookup, applied to action tokens). Deferred to
a future iteration on parsimony grounds (eight more features on
top of an already 22-feature group, with diminishing marginal
information given the dialogue NRC features), and to be
revisited if the dialogue NRC features show strong lift in the
ablation.

This is an explicit auditable scoping decision rather than a
silent omission, matching the auditable-scope pattern of the
lexical group's syntactic-complexity deferral.

### 5.2 Cluster-assigned arc archetypes from a corpus-fit clustering

Reagan et al. 2016 fit their six archetypes via SVD on a large
fiction corpus. Re-fitting on the project's training corpus
would produce six archetypes specific to screenplay dialogue,
which could differ from the fiction-derived templates. The
proposal commits to hand-coded smoothed mathematical templates
(Section 3.4) for two reasons: (a) defensibility against
overfitting the archetype set to the training data and then
evaluating against the same training data, and (b)
interpretability of the resulting features. Corpus-fit components
would require post-hoc interpretation.

### 5.3 Per-character sentiment

Computing sentiment trajectories for individual speakers within a
film, then summarizing across speakers, is a meaningful
direction. Deferred for two reasons: it requires a definition of
"lead character" the project has not committed to, and the
`data_quality_flag` films with collapsed scene structure produce
unreliable per-character statistics. Phase 3b's character-network
group is the better place to introduce per-character measurements
because that group's empty-text filter already gates on
character-level integrity.

### 5.4 NRC Valence-Arousal-Dominance lexicon

The NRC-VAD lexicon (Mohammad 2018) provides three-dimensional
ratings (valence, arousal, dominance) for approximately 20,000
English words. Computing per-film mean valence, arousal, and
dominance would add three features and close a literature-
reference gap that v2 of this proposal otherwise leaves implicit.
Scoped out on parsimony grounds (22 features is already on the
higher side; three more add modest information at the cost of
matrix size). Acknowledging the omission so it is auditable
rather than silent. Defensible future addition if the dialogue
NRC categorical features show strong lift in the ablation.

---

## 6. Feasibility concerns

### 6.1 VADER lexicon and tokenization

VADER ships with NLTK (`vader_lexicon` resource). The
implementation downloads the lexicon at module-load time via the
existing `ensure_nltk_resources` helper introduced in the lexical
module. VADER's compound score is computed on full dialogue lines,
not on individual tokens, because VADER's rule-based formulation
depends on multi-word interactions that token-level scoring
discards.

### 6.2 NRC lexicon distribution and reproducibility

The NRC Word-Emotion Association Lexicon is freely available for
research use under the Mohammad and Turney 2013 license. The
canonical source is `http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm`,
which has historically been more stable than the SUBTLEX-US URLs
that failed during the lexical implementation. Two integration
options:

* Direct download to `data/external/nrc_emolex.tsv` with hash
  validation, matching the pattern the lexical proposal v2
  specified.
* `nrclex` Python package: a wrapper that ships the lexicon
  alongside the package code.

Recommendation: direct download. If the URL fails at
implementation time, fall back to the `nrclex` package and
document the deviation, the same way the lexical group documented
the wordfreq fallback.

### 6.3 Stop-word handling for NRC matching

NLTK English stopwords list (approximately 180 high-frequency
function words) is used in its default form. Stop-word removal
applies to NRC token matching only, not to VADER (which uses full
lines, where stop words contribute to negation rules and
intensifier handling). The configuration knob
`SentimentFeatureConfig.remove_stopwords_for_nrc` defaults to
True; setting it to False produces a sensitivity-analysis variant.

### 6.4 Pooling: scene-windowed by dialogue-line index, not by scene

The four quartile windows are defined on the dialogue-line index
rather than on the scene index. Consequences: robust to
`data_quality_flag` films (the 30 films with collapsed scene
structure still have well-defined dialogue-line quartiles); and
the dialogue-line index approximates plot-time better than
wall-clock-time but is not perfect. The diagnostic step will check
per-window line counts and trajectory shapes.

### 6.5 Empty-text filter (Phase 2 constraint, carry-over from lexical)

Phase 2's parser sometimes inserts empty-text dialogue placeholders
into `dialogue_units` for traceability. The implementation iterates
`dialogue_units` and skips any tuple whose dialogue text is empty
or whitespace-only after stripping.

### 6.6 NRC out-of-vocabulary handling

NRC covers approximately 14,000 English word forms; many dialogue
tokens are not in the lexicon. The implementation treats out-of-
vocabulary tokens as "no emotion association" rather than imputing
a fallback. The diagnostic step reports the per-film OOV rate.

### 6.7 Computational cost

Estimated total compute on the full corpus (1,713 films): under
90 seconds. Substantially faster than the lexical group's 600
seconds because VADER and NRC are both lookup-based.

### 6.8 `data_quality_flag` films

Whole-screenplay aggregates (VADER, NRC) are robust to collapsed
scene structure. Quartile-windowed and arc-clustered features are
computed on the dialogue-line sequence (not scene boundaries), so
they are also robust. Spot-check verification in the diagnostic
step confirms the flagged films' sentiment-feature distributions
match the unflagged corpus.

---

## 7. Pre-implementation sanity check

The lexical proposal Section 7 included a pre-implementation count
of how often dialogue lines lack terminal punctuation. The
analogous question for sentiment is the per-token VADER and NRC
hit rates. Both rates are reported as part of the implementation's
diagnostic step (Section 9 below), since they are cheap to compute
and the implementation's output naturally produces them. If either
rate is below 30 percent on a meaningful number of films (say more
than 100 of 1,194 train-split films), the proposal is flagged for
revision before the ablation is finalized.

---

## 8. Implementation sketch

### 8.1 Module layout

* **Module:** `src/features/sentiment.py`.
* **External resources:** `data/external/nrc_emolex.tsv`
  (downloaded with hash validation on first use).
* **Inputs:** `ParsedScreenplay` objects from
  `data/processed/screenplays_parsed.pkl`, plus the master parquet
  for the `imdb_id` index.
* **Output:** a `pd.DataFrame` indexed by `imdb_id` with the 22
  feature columns above.

### 8.2 Public API

`compute_sentiment_features(parsed: dict[str, ParsedScreenplay],
cfg: SentimentFeatureConfig | None = None) -> pd.DataFrame`.

### 8.3 Configuration knobs (`SentimentFeatureConfig`, frozen dataclass)

* `n_quartile_windows`: 4.
* `arc_template_length`: 100.
* `remove_stopwords_for_nrc`: True.
* `archetype_set`: `"reagan_six"`.

### 8.4 Determinism

All features deterministic given input text and config. Re-running
produces byte-identical output.

### 8.5 Testing

* Smoke test: compute features on first ten films, assert non-NaN
  and finite.
* Unit test: VADER mean on reference string.
* Unit test: NRC emotion proportions on fixture string against
  fixed mock NRC table.
* Unit test: arc-similarity computation on synthetic
  Man-in-a-Hole trajectory returns highest similarity for
  `arc_similarity_man_in_a_hole`.
* Integration test: compute on full corpus, assert shape
  (1,713, 22), no all-NaN columns.

### 8.6 Multi-family ablation through `save_run`

A new `src/experiments/run_sentiment_ablation.py` mirrors the
lexical pattern. The augmented matrix joins structural baseline
features, lexical features, and the new sentiment features.
Lift is computed against the **Phase 3a revised dialogue-only
floor**, not against the lexical-augmented matrix. This matches
the lexical group's comparison point: the ablation table reports
each Phase 3b group's standalone contribution against the same
baseline. The Phase 3c combinations sub-phase is where joint-lift
evaluation belongs (sentiment-on-top-of-lexical, etc.); the
standalone row here does not pre-empt it.

The trainer reports both train (in-sample) and OOF
(cross-validated) values for every metric, matching the project's
new evaluation convention.

---

## 9. Post-implementation diagnostic checks

Each check has a specific threshold that, if tripped, prompts a
review before the ablation row gets appended.

1. **Pairwise correlations between new sentiment features and
   the structural-plus-lexical baseline.** Threshold: |r| > 0.85
   prompts a pair review.
2. **Within-sentiment pairwise correlations.** Threshold: same.
   Concerns: NRC `fear`-`disgust` historically correlate
   ~0.6-0.7 on English text; consecutive quartile means correlate
   by construction; Man-in-a-Hole and Icarus archetypes are
   roughly inverse-correlated by construction (their hand-coded
   templates are reflections of each other).
3. **Sentiment quartile coherence.** The four quartile means
   should **average** (arithmetic mean) to approximately the
   whole-screenplay mean (v2 correction: v1 incorrectly said
   "sum"). If the deviation is large (more than 0.05 on the
   compound score scale), the quartile-windowing is misaligned.
4. **Arc-similarity range and target relationship.** Each
   archetype-similarity score should span at least 0.5 units of
   variation across the corpus (low-variation archetypes get
   dropped). Additionally, compute univariate Pearson correlation
   between each of the six archetype-similarity features and each
   of the three targets on the train split (added in v2 per
   planning-conversation polish guidance: with six similarity
   features fit on n = 1,199 train films, one or two could look
   predictive purely by chance; the multiple-comparisons concern
   is addressed by requiring an archetype-target correlation to
   exceed |r| > 0.10 on a holdout fold split before claiming
   signal).
5. **VADER lexicon-hit rate distribution.** Per-film proportion
   of dialogue lines producing a non-zero compound score.
   Threshold: more than 100 films below 30% triggers a
   revision flag.
6. **NRC OOV rate distribution.** Per-film proportion of dialogue
   tokens not in the NRC lexicon. Same threshold logic.
7. **`data_quality_flag` films vs unflagged.** Spot-check
   sentiment-feature distributions on the 24 train-split flagged
   films against unflagged. Threshold: feature-level z-score
   difference greater than 1.0 is flagged for review.
8. **NRC binary-positive vs multi-dimensional comparison.**
   Compute a binary positive-versus-negative NRC score (positive
   minus negative emotion proportion) and compare its correlation
   with each target against the multi-dimensional NRC features'
   correlations. If binary captures most of the multi-dimensional
   information, the eight-dimensional choice is over-engineered.
9. **VADER tail-compression on subgenres.** The calibration-
   mismatch caveat in Section 3.1 predicts that period dramas,
   courtroom thrillers, and literary adaptations may have
   compressed compound-score distributions. Compute the
   standard deviation of VADER compound-mean by primary genre and
   flag genres whose within-genre std is more than two standard
   deviations below the cross-genre median std.

---

## 10. References

* Hutto, C., and Gilbert, E. (2014). "VADER: A Parsimonious
  Rule-based Model for Sentiment Analysis of Social Media Text."
  *Proceedings of the Eighth International Conference on Weblogs
  and Social Media*.
* Mohammad, S. M., and Turney, P. D. (2013). "Crowdsourcing a
  Word-Emotion Association Lexicon." *Computational Intelligence*,
  29(3), 436-465.
* Mohammad, S. M. (2018). "Obtaining Reliable Human Ratings of
  Valence, Arousal, and Dominance for 20,000 English Words."
  *Proceedings of ACL*. Cited as the NRC-VAD source for the
  scoped-out feature in Section 5.4.
* Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., and
  Dodds, P. S. (2016). "The emotional arcs of stories are
  dominated by six basic shapes." *EPJ Data Science*, 5(1), 31.
* Vonnegut, K. (1981). "Shape of Stories." Lecture / essay
  identifying three recurring narrative arc shapes; cited as the
  philosophical predecessor to Reagan et al.'s computational
  work.

---

Proposal v2 for the **sentiment** feature group is ready.
Implementation proceeds against v2 unless the planning conversation
flags a divergence from the polish guidance.
