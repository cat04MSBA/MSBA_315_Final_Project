# Phase 3 Lexical Feature Proposal (v2)

**Group:** Lexical (1 of 5 in the Phase 3b incremental ablation)
**Status:** Awaiting planning-conversation final sign-off
**Date:** 2026-05-03
**Version:** 2 (incorporates planning-conversation review of v1)

This is the revised lexical-feature proposal. v1 was reviewed by the
planning conversation on 2026-05-03 with two required additions
(external frequency norms via SUBTLEX-US; parallel measurement on
action text), three recommended polish items (vocd-D acknowledgement,
explicit syntactic-complexity scoping, pre-implementation
period-appending sanity check), and explicit answers to the four
open questions. v2 incorporates all required additions, all
recommended polish, all open-question answers, and the post-
implementation tracking items the planning conversation surfaced.

The pre-implementation period-appending sanity check was run as
part of preparing v2: the result is reported in Section 7 below.

---

## 1. Why lexical first

Three reasons place lexical at the head of the Phase 3b ablation
queue.

**Cheapest to compute.** Every feature in this proposal is a simple
aggregate over the dialogue or action-text tokens of each screenplay.
No model fitting, no embeddings, no LDA. The full corpus (1,713
films, median 880 dialogue lines per film, around 120,000 dialogue
characters per film) processes in well under a minute on a single
CPU. Landing the cheap groups first lets the ablation measure the
lift each contributes before paying the cost of the expensive groups
(embeddings in particular).

**Independent of upstream feature decisions.** Lexical aggregates do
not depend on sentiment lexicons, topic models, or embedding choices.
The features can be computed once and never recomputed regardless of
what later groups do.

**Most likely to expose vocabulary-level effects on the regression
target.** Vocabulary diversity, lexical sophistication, and
readability have well-documented relationships with critical
reception (which correlates with rating-style outcomes through
audience word-of-mouth) and with target-audience alignment (which
correlates with revenue through marketing efficiency). Both
mechanisms feed the regression target `log_roi` more cleanly than
they feed the binary classification targets.

---

## 2. Proposed features (14 total)

The 14 features divide into five sub-groups: vocabulary diversity,
lexical sophistication, readability, length statistics, and
punctuation/pronoun signals. Sub-groups 1, 3, and 4 are computed on
dialogue and on action text in parallel where the feature is
domain-portable; sub-groups 2 and 5 are computed on dialogue only
for reasons stated below.

For each feature: definition, rationale, and the channel it is
computed on. All features are single scalars per film.

### 2.1 Vocabulary diversity (3 features: 2 dialogue, 1 action)

**`mtld_dialogue`. Measure of Textual Lexical Diversity, dialogue
text only.** Length-robust replacement for type-token ratio. Walks
the dialogue text in order and computes the average length of token
segments needed to drop the running TTR below a threshold (default
0.72, the standard value from McCarthy and Jarvis 2010); the result
does not have the mechanical length artifact that disqualifies raw
TTR for cross-screenplay comparison. The mechanism: screenplays with
richer vocabulary score higher and this should correlate with
critical reception.

**`mtld_action`. Same measure, on action text only.** The action
channel (concatenated stage-direction and scene-description text per
the Phase 2 parser) carries a different stylistic load from
dialogue. Period dramas tend to have sparse, evocative action text
("She watches the rain. The clock strikes."); action films tend to
have dense, choreographed action text with technical vocabulary
("He pivots, drops the magazine, slams a fresh one home."). Treating
dialogue and action vocabulary diversity as separate features lets
the ablation tell us whether the predictive signal lives in the
dialogue channel, the action channel, or both. If the two columns
are highly correlated post-implementation (above the 0.85 threshold
specified in Section 8), one drops out cleanly without contaminating
the other.

**`hapax_ratio_dialogue`. Hapax legomena ratio, dialogue only.**
Proportion of distinct dialogue words that appear exactly once in
the screenplay's dialogue. High values indicate broad vocabulary
specificity (lots of one-off words, characteristic of literary or
descriptive writing); low values indicate narrow vocabulary
(repetitive dialogue, characteristic of stylized or genre-formula
films). Partially correlated with MTLD but captures the tail of the
vocabulary distribution rather than its overall shape. Computed on
dialogue only because the action-channel hapax ratio is largely
determined by descriptive vocabulary that varies more by genre than
by craft, which the genre dummies in the baseline already absorb.

**A note on the choice of MTLD over vocd-D.** Both MTLD and vocd-D
(McCarthy and Jarvis 2010) are length-robust diversity measures and
both are defensible choices. McCarthy and Jarvis report MTLD is
slightly more stable on shorter texts and vocd-D is slightly more
stable on longer texts. The corpus is on the long side (median 880
dialogue lines per film), which would weakly favour vocd-D, but
MTLD is the more common choice in recent computational work, has
simpler implementation, and is comparable in stability at corpus
length. The choice is genuinely close; MTLD is selected for
consistency with recent literature and ease of reproducibility. If
post-implementation diagnostics surface a stability issue, vocd-D is
a one-module-rewrite alternative.

### 2.2 Lexical sophistication (2 features, dialogue only)

These two features close the gap that v1's `mean_word_length_chars`
left open. Word length conflates Latinate-vs-Anglo-Saxon length
effects with actual sophistication: a screenplay heavy on
"thoughtfully", "absolutely", and "remember" scores similarly under
mean word length to a screenplay heavy on "supercilious",
"perspicacious", and "gracile", which is exactly the distinction the
sophistication signal should capture. External frequency norms
provide a direct measure.

**Choice of frequency reference: SUBTLEX-US over COCA.** SUBTLEX-US
(Brysbaert and New 2009) is a frequency table built from 51 million
words of English-language film and television subtitles. COCA
(Davies 2008-) is a 1-billion-word reference corpus drawn primarily
from prose registers (academic, fiction, magazine, newspaper). For
this project's input, dialogue from screenplays, SUBTLEX-US is the
domain-matched reference. Words that are rare in dialogue
specifically should index sophistication better than words that are
rare in prose. SUBTLEX-US is freely available for research use under
the original publication's license and ships as a single CSV file of
roughly 80,000 word-frequency pairs. Kuperman, Stadthagen-Gonzalez,
and Brysbaert 2012 age-of-acquisition norms are a defensible
alternative but measure a different construct (developmental
acquisition difficulty), so SUBTLEX-US is the more directly relevant
reference for the sophistication mechanism.

**Scoping of frequency-based features to dialogue only.** SUBTLEX-US
is calibrated on subtitle dialogue, so applying its frequency norms
to action prose introduces a small domain mismatch (the action
channel uses descriptive vocabulary that may be uncommon in subtitle
dialogue but standard in screenplay action). To avoid that mismatch
in v2, both SUBTLEX-scored features are computed on dialogue only.
If dialogue-side SUBTLEX shows strong lift in the ablation, an
action-text variant can be added in a future iteration; absent that
signal there is no benefit to introducing the domain-mismatch column.

**`mean_log_frequency_subtlex`. Mean SUBTLEX-US log-frequency over
dialogue tokens.** For each dialogue token, look up its SUBTLEX-US
log-frequency value (or assign a fallback value for out-of-vocabulary
tokens, see below); take the mean across all dialogue tokens in the
film. Low values indicate the screenplay uses rare or sophisticated
vocabulary on average; high values indicate everyday vocabulary.

**`rare_word_proportion_subtlex`. Proportion of dialogue tokens
falling in the bottom quartile of SUBTLEX-US log-frequency.** Catches
the tail behaviour the mean smooths over. A screenplay with a small
number of strikingly rare words and an otherwise common vocabulary
will register a moderate `mean_log_frequency_subtlex` but a high
`rare_word_proportion_subtlex`; the two columns separate the
average-rarity signal from the rare-tail signal cleanly. Bottom
quartile is selected over bottom decile because the decile produces
sparser counts on shorter screenplays and is more sensitive to the
specific cutoff value; the quartile is more stable.

**Out-of-vocabulary handling.** Tokens not present in the SUBTLEX-US
table (proper names, neologisms, non-English borrowings) are
assigned the SUBTLEX log-frequency of the rarest 5th-percentile word
in the table. This is a defensive choice: it treats unknown tokens
as plausibly-rare rather than as missing, which prevents the OOV
rate from swamping the mean. The rate is monitored
post-implementation as part of the diagnostic step.

### 2.3 Readability (2 features: 1 dialogue, 1 action)

**`flesch_kincaid_grade_dialogue`. Flesch-Kincaid grade level on
dialogue text.** Combines mean sentence length and mean syllable
count per word into a US school-grade-level estimate. Low scores
indicate accessible dialogue (typical of family films and broad
comedies); high scores indicate dense or formal dialogue (typical of
period dramas, legal thrillers, intellectually pitched films).

The score is computed sentence-by-sentence within each dialogue
line. Single-word exclamations like "Yes." or "Look out!" are
handled by the standard formula without modification.

A caveat applies. Dialogue is fragmented relative to the continuous
prose Flesch-Kincaid was calibrated on, so the absolute values may
compress relative to published reference distributions. The relative
ordering across films should still carry signal, which is what the
modelling phase consumes. If post-implementation diagnostics show
the column has unusually narrow variance or behaves erratically
relative to the reference distribution, it gets dropped before
ablation rather than being kept on principle.

**`flesch_kincaid_grade_action`. Same measure, on action text only.**
Action text is closer in form to continuous prose than dialogue is,
which means the Flesch-Kincaid score should behave more like its
calibration target on this channel. The expected predictive
mechanism differs from the dialogue version: action-text complexity
captures the stylistic difference between sparse evocative action
(low score, characteristic of literary drama) and dense technical
action (higher score, characteristic of action and thriller films).
As with `mtld_action`, this lets the ablation surface where the
signal lives. If the dialogue and action columns are highly
correlated post-implementation, one drops out.

### 2.4 Length statistics (3 features, dialogue only)

These are dialogue-only because the action channel does not have a
natural "line" unit: stage directions are paragraphs and scene
descriptions are prose blocks, so length statistics on action would
mostly reflect formatting choices rather than stylistic signal.

**`mean_dialogue_line_tokens`. Mean tokens per dialogue line.**
Captures average utterance length. Action and comedy tend toward
shorter average utterances (fast cuts, reactive dialogue); drama and
period pieces tend toward longer (monologues, exposition). Tokens
rather than characters because tokens normalize across vocabulary
register.

**`std_dialogue_line_tokens`. Standard deviation of tokens per
dialogue line.** Captures how much utterance length varies within a
screenplay. High values indicate rhythmic variation (mix of
back-and-forth and monologue); low values indicate uniform pacing.
Mean and standard deviation together describe the shape of the
dialogue-line-length distribution well enough that the median is not
separately included.

**`short_line_proportion`. Proportion of dialogue lines under five
tokens.** Captures the rate of "snappy" exchanges. A common feature
of fast-paced comedies and action films; rarer in slow-burn drama.
Five tokens is a deliberately tight cutoff: it captures one-word and
two-word reactions that the mean-and-standard-deviation pair does
not isolate cleanly. Cutoffs at three or seven were considered; both
would be highly correlated with the five-token version and add
little incremental information.

### 2.5 Punctuation and pronoun signals (3 features, dialogue only)

These are dialogue-only because action text does not contain dialogue
punctuation in the relevant senses (questions and exclamations) and
does not address listeners with pronouns in the same way.

**`question_rate_per_1k_tokens`. Question marks per 1,000 dialogue
tokens.** Captures interrogative tone. Mystery, thriller, and
crime-procedural screenplays carry more questions per thousand tokens
than action or animation. The rate normalizes for total dialogue
length so that long screenplays do not score artificially high.

**`exclamation_rate_per_1k_tokens`. Exclamation marks per 1,000
dialogue tokens.** Captures intensity, urgency, or shouted dialogue.
Action, horror, and intense drama carry more; quiet drama and
slow-burn suspense carry fewer. Useful as a marker of tonal pitch
beyond the information already encoded in `primary_genre_bucketed`.

**`first_to_second_pronoun_ratio`. Ratio of first-person tokens to
second-person tokens.** Specifically: count of "i", "me", "my",
"mine", "myself" divided by count of "you", "your", "yours",
"yourself", "yourselves", with case folded and a small epsilon
(1.0) added to the denominator to prevent division by zero. The
archaic and stylized forms "thou", "thee", "thy", "thine", "thyself"
are added to the second-person count per the planning conversation's
direction (the corpus extends back to 1932 and includes period
dramas; including these forms costs nothing and asymmetrically
benefits the films that use them). High values indicate interiority
and self-reflection (memoir-style narration, personal drama); low
values indicate confrontation and direct address (debate, romance,
courtroom). Selected over having two separate pronoun-rate features
because the ratio captures the relevant relational dimension in a
single dimension and avoids scale artifacts from differences in
overall pronoun density across screenplays.

---

## 3. Pre-registered expected lift

These are predictions made before implementation. After
implementation, the actual lift over the Phase 3a baseline is
recorded alongside these predictions in the ablation table.
Substantial deviation in either direction is an item to surface
before proceeding to the next group.

### Regression target `log_roi`

Predicted lift in R-squared: **+0.010 to +0.025**.

(v1 predicted +0.005 to +0.020; v2 widens the upper bound modestly to
account for the SUBTLEX-sophistication features, which capture a
mechanism v1 did not.)

Mechanism: vocabulary diversity, lexical sophistication, and
readability provide incremental signal about screenplay craft and
audience-targeting clarity, both of which feed log_roi through
different channels (rating reception and revenue capture
respectively). The lift is expected to be modest because the corpus
survivor-filters to films that already cleared the development
threshold, compressing the between-film variance these features can
explain. Action-channel features (`mtld_action`,
`flesch_kincaid_grade_action`) may or may not contribute beyond
their dialogue counterparts; the ablation tells us.

### Classification target `roi_gt_1` (gross-profitable)

Predicted lift in AUC-ROC: **+0.000 to +0.010**.

Mechanism: the 80% positive base rate makes this target signal-thin
in general. Lexical features may help slightly with identifying the
subset of films whose dialogue suggests a poor commercial fit
(over-indexed period vocabulary, low conversational rhythm) but the
small minority class limits the available headroom. Unchanged from
v1.

### Classification target `roi_gt_2` (net-profitable)

Predicted lift in AUC-ROC: **+0.015 to +0.035**.

(v1 predicted +0.010 to +0.030; v2 widens modestly to account for
the SUBTLEX-sophistication and action-text features.)

Mechanism: this target carries the most headroom in the baseline
and should respond best to features that capture audience-targeting
clarity (readability, dialogue pacing, vocabulary register). The
"net-profitable" line roughly tracks blockbuster versus
non-blockbuster, which lexical features should weakly but cleanly
differentiate beyond what genre and era already capture.

### Combined expectations relative to forward bands

The forward expectations for the full Phase 3b ablation are
`log_roi` R-squared in the 0.10 to 0.20 band and `roi_gt_2` AUC in
the 0.65 to 0.72 band. The lexical group alone is expected to
contribute roughly one-fifth to one-third of those total lifts;
sentiment, topic, embedding, and character-network groups each add
their own share. If lexical alone delivers most of the predicted
band, that is itself a finding to surface (suggests the other groups
will plateau quickly).

---

## 4. Resolved open questions (from v1 review)

### 4.1 Tokenizer choice

**Decision:** Commit to NLTK `word_tokenize` for Phase 3b lexical.

NLTK is the proposed default and the planning conversation confirmed
it. The switch-point reasoning if a downstream group requires spaCy:
re-run lexical with spaCy at that point, OR accept the tokenizer
split with a brief justification. Penn-Treebank (NLTK) and spaCy
English tokenizers agree on more than 99 percent of tokens for
prose-like text; the disagreement concentrates on contractions,
which affects a handful of tokens per film. MTLD and Flesch-Kincaid
aggregates are not materially tokenizer-sensitive at the corpus
scale they operate on. Either path is defensible; the choice gets
documented in `FEATURE_NOTES.md` when the time comes.

For sentence-boundary detection inside the readability score,
NLTK `sent_tokenize` is the matching default.

### 4.2 Stop-word policy for vocabulary diversity

**Decision:** No stop-word removal.

The standard MTLD definition does not remove stop words. Pronouns
and function words are part of vocabulary diversity, and removing
them changes the metric's behaviour without clean theoretical
justification. The configuration toggle to remove stop words is kept
in the implementation as an ablation knob for later curiosity, but
the default and the run that lands in the report is no removal.

### 4.3 Scope of pronoun count for the first-to-second-person ratio

**Decision:** Include archaic and stylized forms.

The corpus genuinely spans 1932 to 2023 (the pre-1995 cutoff was
reversed during Phase 2 once a count error in the original
justification was discovered) and pre-1980 has visible film counts
of roughly five to fifteen per year, which means the corpus contains
period dramas and Shakespeare adaptations. The archaic second-person
forms "thou", "thee", "thy", "thine", "thyself" are added to the
second-person count. Cost is one extra five-element string list, the
risk of including the forms is essentially zero on modern dialogue
(it does not use them anyway), and the asymmetric upside (correctly
handling the films that do use them) makes the inclusion trivially
worth doing.

### 4.4 Readability per-film exclusion

**Decision:** All-films plan with documented caveat.

The conditional version (compute the readability score only when the
median dialogue line length exceeds some threshold, mark it missing
otherwise) introduces a threshold that becomes its own methodology
debate. The cost of including the column for every film is zero
because the model's regularization will downweight the column on
films where the score is degenerate. Per the planning conversation:
"don't multiply mechanisms".

---

## 5. Acknowledgement of out-of-scope features (defensibility)

The Phase 3 brief lists "syntactic complexity" under the lexical
group umbrella. Established syntactic-complexity measures (Mean
Length of T-Unit, Dependent Clauses per T-Unit, parse-tree depth)
require dependency parsing, which means a heavyweight tokenizer
like spaCy with the appropriate English model loaded. The current
proposal commits to NLTK `word_tokenize` because the cheap features
in this group do not need parsing and the lift from syntactic
features is not yet established for screenplay-style text.

**Decision:** syntactic-complexity features are explicitly deferred
out of this group. They will be revisited if spaCy comes online for
a downstream group (sentiment if a transformer-based classifier is
chosen; embeddings if sentence-transformer preprocessing benefits
from spaCy normalization). Flesch-Kincaid grade serves as a
degenerate proxy in the meantime: it captures a coarse syntactic
signal through mean sentence length but does not replicate the
finer-grained measures from the syntactic-complexity literature.

This is an explicit auditable scoping decision rather than a silent
omission, per the planning conversation's review of v1.

---

## 6. Feasibility concerns

### 6.1 Tokenization cost and choice

NLTK `word_tokenize` and `sent_tokenize` are the chosen tokenizers.
NLTK is roughly ten times faster than spaCy for the operations this
group needs. For the full corpus the tokenization step dominates
runtime and runs in well under a minute total on a single CPU.
Memory footprint is negligible.

### 6.2 Empty-text dialogue filter (Phase 2 constraint)

Phase 2's parser sometimes inserts empty-text dialogue placeholders
into `dialogue_units` to preserve traceability of structural
irregularities in the source XML. These must be filtered before any
per-token aggregate. The implementation iterates `dialogue_units`
and skips any tuple whose dialogue text is empty or whitespace-only
after stripping. This is the same filter the Phase 2 Tier 1.3 fix
applied to `n_unique_characters`. The filter is applied
defensively at every entry point in `src/features/lexical.py`, even
when the upstream parser is expected to have applied it already.

### 6.3 Length sensitivity of vocabulary measures

Standard type-token ratio is length-dependent and would produce
mechanically different values for short versus long screenplays.
MTLD is the length-robust replacement. The hapax ratio is also
length-sensitive in principle (longer text has proportionally fewer
one-off words), but the corpus's screenplay lengths fall in a
relatively narrow band (median 880 dialogue lines, inter-quartile
range roughly 600 to 1,200), so the bias is small in practice. The
residual correlation with total dialogue length is monitored
post-implementation; the column is dropped if the correlation is
above 0.4 (lenient threshold accepted by the planning conversation
in v1 review). If post-implementation the residual correlation lands
in the 0.3 to 0.4 band rather than well below 0.3, the threshold
gets revisited.

### 6.4 `data_quality_flag` films (n = 30)

Phase 2 flagged 30 films whose source XML encodes the entire
screenplay as fewer than 10 scenes. The handling for Phase 3b is
confirmed: keep these films in the train, calibration, and test
splits for sample size, exclude them from features that depend on
scene-level integrity (character-network features in particular),
and use them as-is for features that aggregate across the whole
screenplay. Lexical features are whole-screenplay aggregates and
are robust to scene-level degeneracy, so all 30 flagged films use
the same lexical-feature pipeline as the rest of the corpus. Their
distributions on the lexical features are sanity-checked
post-implementation as part of the diagnostic step (Section 8).

### 6.5 Redundancy with structural baseline features

Two of the proposed dialogue length features overlap conceptually
with structural metrics already in the baseline.

* `mean_dialogue_line_tokens` overlaps with `n_dialogue_lines` and
  `total_dialogue_chars`, which jointly imply average line length
  in characters.
* `short_line_proportion` is a function of the dialogue-line-length
  distribution that is not trivially recoverable from the baseline
  features.

Pairwise correlations between the new features and the baseline
structural features are monitored post-implementation; correlations
above 0.85 prompt a review of which member of the pair stays.

### 6.6 SUBTLEX-US distribution and reproducibility

The SUBTLEX-US frequency table is a single CSV file of approximately
80,000 word-frequency rows. The file is freely available for research
use under the Brysbaert and New 2009 license. The implementation
ships the file alongside the code under `data/external/subtlex_us.csv`
and reads it deterministically at module-load time. Hashing the file
on first read protects against silent corruption or substitution.
The table loads in under a second and remains in memory for the
duration of feature extraction; memory cost is roughly 5 MB.

### 6.7 Punctuation conventions in the source XML

Some MovieSum dialogue lines end without terminal punctuation. The
implementation appends a period to the end of any dialogue text
that lacks terminal punctuation before sentence tokenization, but
only for the readability score; the question and exclamation rates
count actual terminal punctuation as present in the source.

Section 7 reports the empirical rate at which the period-appending
fallback fires.

---

## 7. Pre-implementation sanity check: terminal-punctuation rate

The planning conversation requested a pre-implementation count of
how often dialogue lines lack terminal punctuation in the corpus, to
determine whether the period-appending heuristic in Section 6.7
would fire often enough to make the readability score's
sentence-segmentation input largely synthetic.

**Result:** of 1,527,169 non-empty dialogue lines across all 1,713
films, 1,414,941 (92.7 percent) end with terminal punctuation in
the source XML and 112,228 (7.3 percent) do not.

The 7.3-percent rate is in the "harmless" band (below 10 percent)
the planning conversation's framing identified. The
period-appending heuristic fires on roughly one dialogue line in
fourteen, which is a small fraction of the input to sentence
tokenization. The Flesch-Kincaid score's sentence-length input is
therefore mostly real, not synthetic. The column is included with
the documented caveat from Section 2.3 and no stronger handling.

---

## 8. Implementation sketch

### 8.1 Module layout

* **Module:** `src/features/lexical.py`.
* **External resource:** `data/external/subtlex_us.csv` (the SUBTLEX-US
  frequency table). A small bootstrap script under
  `src/utils/external_data.py` downloads the file on first use and
  validates its hash.
* **Inputs at compute time:** the per-film `ParsedScreenplay` objects
  from `data/processed/screenplays_parsed.pkl`, plus the master
  parquet for the `imdb_id` index.
* **Output:** a `pd.DataFrame` indexed by `imdb_id` with the 14
  feature columns above, joined into the running feature matrix
  alongside the structural baseline features.

### 8.2 Public API

* `compute_lexical_features(parsed: dict[str, ParsedScreenplay],
  cfg: LexicalFeatureConfig | None = None) -> pd.DataFrame`. The
  primary entry point.

### 8.3 Configuration knobs (`LexicalFeatureConfig`, frozen dataclass)

* `tokenizer`: `"nltk"` (default) or `"spacy"`.
* `mtld_threshold`: 0.72 (the McCarthy and Jarvis 2010 default).
* `short_line_cutoff_tokens`: 5.
* `remove_stop_words`: False (default; ablation knob only).
* `subtlex_oov_percentile`: 5 (the percentile assigned to
  out-of-vocabulary tokens, see Section 2.2).
* `subtlex_rare_quartile`: 4 (the quartile counted as "rare" for
  `rare_word_proportion_subtlex`).

### 8.4 Determinism

All features are deterministic given the input text and the
configuration. No randomness involved. Re-running the module
produces byte-identical output.

### 8.5 Testing

* Smoke test computes the features on the first ten films of the
  corpus and asserts non-NaN, finite values across all 14 columns.
* Unit test verifies that MTLD on a known reference string matches
  the expected value to four decimal places.
* Unit test verifies that `mean_log_frequency_subtlex` on a fixed
  fixture string returns the expected value given a fixed (mock)
  SUBTLEX table.
* Integration test runs `compute_lexical_features` on the full
  corpus and asserts (a) the output DataFrame has 1,713 rows, (b)
  the column set matches the expected 14, (c) no NaN values across
  the entire matrix.

---

## 9. Post-implementation diagnostic checks

These are the checks the planning conversation pulled together as
items to track once the features are computed. They run as a small
diagnostic step before the lexical features are committed to the
ablation table.

1. **Pairwise correlations between new lexical features and
   existing baseline structural features.** Threshold for review:
   correlation above 0.85. Pairs above the threshold are reviewed
   to decide which member of the pair is retained for the
   ablation.
2. **Hapax ratio's residual correlation with total dialogue
   length.** Threshold for dropping the feature: correlation above
   0.4. If the correlation lands in the 0.3 to 0.4 band, the
   threshold itself is revisited rather than mechanically applied.
3. **Flesch-Kincaid grade column variance against published
   reference distributions.** If the corpus distribution is
   unusually narrow or behaves erratically (especially the
   action-text variant), the column is dropped before the
   ablation.
4. **Distributions of the lexical features on the 30
   `data_quality_flag` films.** Informal sanity check: do these
   films behave like the rest of the corpus on whole-screenplay
   aggregates, as the theoretical handling assumes? Flagged for
   review only if obviously out of distribution.
5. **Residual correlation between SUBTLEX-frequency features and
   `mean_word_length_chars`-style proxies (if a length-based
   sophistication proxy ends up in the matrix; v2 does not include
   one explicitly, but the overlap is monitored regardless).**
   Correlation above 0.85 prompts dropping one.
6. **Residual correlation between `_action`-suffixed features and
   their `_dialogue` siblings.** Correlation above 0.85 means the
   dialogue-versus-action distinction is not paying for itself;
   the action variants get collapsed.
7. **SUBTLEX-US out-of-vocabulary rate.** Tracked as a diagnostic.
   If OOV exceeds 15 percent of dialogue tokens on a meaningful
   number of films, the OOV-handling strategy is revisited (the
   corpus has post-1980 American films primarily, so the OOV rate
   should be modest; high OOV would suggest a tokenization or
   normalization issue).

---

## 10. References

* Brysbaert, M., and New, B. (2009). "Moving beyond Kučera and
  Francis: A critical evaluation of current word frequency norms
  and the introduction of a new and improved word frequency
  measure for American English." *Behavior Research Methods*,
  41(4), 977-990.
* Kuperman, V., Stadthagen-Gonzalez, H., and Brysbaert, M. (2012).
  "Age-of-acquisition ratings for 30,000 English words." *Behavior
  Research Methods*, 44(4), 978-990. Cited as the alternative
  reference in Section 2.2.
* McCarthy, P. M., and Jarvis, S. (2010). "MTLD, vocd-D, and HD-D:
  A validation study of sophisticated approaches to lexical
  diversity assessment." *Behavior Research Methods*, 42(2),
  381-392.
* Davies, M. (2008-). The Corpus of Contemporary American English
  (COCA). Cited as the prose-corpus reference rejected in
  Section 2.2 in favour of SUBTLEX-US.

---

Proposal v2 for the **lexical** feature group is ready. Ready for
final planning-conversation sign-off before implementation begins.
