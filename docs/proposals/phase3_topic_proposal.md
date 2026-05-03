# Phase 3 Topic Feature Proposal (v1)

**Group:** Topic (3 of 5 in the Phase 3b incremental ablation)
**Status:** Awaiting planning-conversation review
**Date:** 2026-05-03

This is the v1 draft of the topic-feature proposal. It is structured
to match the lexical and sentiment proposals so the planning
conversation can review it under the same template. Polish guidance
returned by the planning conversation will be folded into a v2 before
implementation begins, matching the pattern used for the previous two
groups.

---

## 1. Why topic third

The topic group sits third in the Phase 3b ablation queue for three
reasons.

**It is the first group whose mechanism plausibly produces signal
orthogonal to genre.** The lexical and sentiment groups both produced
null standalone verdicts (see
`docs/handoffs/phase_3b_lexical_handoff.md` and
`docs/handoffs/phase_3b_sentiment_handoff.md`); the most likely
mechanism in both cases was that the features carried information the
genre dummies and structural counts had already absorbed. Topic
features are different in kind. Two films can sit in the same genre
but address very different subjects: a Drama can be a courtroom drama
about a wrongful conviction, a Drama can be a coming-of-age story
about a piano teacher. These produce different topic distributions
even though the genre dummy is the same. Topic distributions therefore
have a real chance of carrying within-genre signal that the structural
baseline cannot pre-empt.

**Subject matter has documented effects on commercial outcomes.**
Eliashberg et al. (2014) reported that production-stage subject-matter
classifications predict box office in pre-release work; Mariani et al.
(2020) found that latent topics extracted from subtitles correlate
with audience reception. The proposals in this group's design
deliberately stay within the LDA topic-modelling tradition both papers
operated in, so any positive signal here is interpretable against the
prior literature.

**It is computationally moderate.** LDA on this corpus runs in single-
digit minutes on a single CPU. Sentence-transformer embeddings (the
fifth group) are an order of magnitude more expensive; saving the
expensive group for last lets us learn from the previous four groups
before paying that cost.

---

## 2. The substantive design question: backend, unit, and topic count

Topic modelling has three substantive choices that the proposal must
commit to before implementation: the modelling backend, the document
unit, and the topic count `K`.

### 2.1 Backend: LDA via scikit-learn

**Decision (v1): scikit-learn `LatentDirichletAllocation` with
variational Bayes inference.**

Three backends are defensible here. **Latent Dirichlet Allocation
(Blei, Ng, and Jordan 2003)** is the textbook choice and the one the
brief explicitly names. **Non-negative Matrix Factorization** (NMF)
is a defensible alternative that produces sparser topic distributions
but lacks the probabilistic interpretation. **BERTopic** (Grootendorst
2022) wraps sentence-transformer embeddings with a UMAP-and-HDBSCAN
clustering pipeline; it produces more coherent topics on small
corpora but introduces a substantial dependency stack and an
embedding-fitting step that overlaps with the fifth feature group's
work.

The v1 commits to scikit-learn LDA for three reasons. First, it is
already a project dependency (no new install). Second, it is the most
direct implementation of the brief's "topic distributions over
screenplay text" specification. Third, the no-leakage discipline is
trivial to enforce: the `LatentDirichletAllocation` estimator's
`fit` and `transform` methods cleanly separate training-fold fitting
from out-of-fold inference. NMF is documented as the alternative
backend that would land if LDA produces low-coherence topics in the
diagnostic step. BERTopic is explicitly out of scope for this group;
it overlaps with Phase 3b group 5 (embeddings) on infrastructure
and is more naturally evaluated there.

### 2.2 Document unit: whole-screenplay dialogue

**Decision (v1): one document per film, concatenating dialogue text.**

Three unit choices were considered:

* **Whole-screenplay dialogue** — concatenate every dialogue line
  per film into a single document. Simplest to implement, produces
  one topic distribution per film directly, matches the
  whole-screenplay aggregation pattern used by the VADER aggregates
  and the NRC proportions in the sentiment group.
* **Per-scene dialogue** — fit LDA on per-scene documents, then pool
  back to film level by averaging or max-pooling the per-scene topic
  distributions. Captures within-film topical variation; loses
  coherence on short scenes; complicates the no-leakage discipline.
* **Per-line dialogue** — too short for LDA to extract meaningful
  topics.

The v1 commits to whole-screenplay dialogue. The pooling-across-units
question is the substantive design question for the embeddings group
(group 5); preserving that as a single methodology decision rather
than splitting it across two groups keeps the ablation results easier
to interpret. If the topic group's standalone-lift result is null
under whole-screenplay aggregation but the planning conversation
wants to revisit per-scene pooling at Phase 3c combinations, the LDA
implementation is configured to make per-scene fitting a one-flag
change.

Action text is excluded for the same parsimony reason cited in the
sentiment proposal: the lexical group already evaluates action-channel
features, and the marginal gain from adding action-channel topics on
top of dialogue topics is uncertain enough that scoping it out keeps
the standalone-lift comparison clean.

### 2.3 Topic count `K`: 20

**Decision (v1): K = 20, with diagnostic guidance to revisit at
implementation time if coherence is low.**

`K` is the most consequential single hyperparameter. Common heuristics
suggest somewhere between sqrt of the document count (about 35 for
1,199 train films) and 10 to 20 for interpretability-favoured small-
corpus work. Lower K produces broader, more interpretable topics that
are also more likely to overlap with genre dummies; higher K produces
finer-grained topics with better orthogonality to genre but worse
coherence at the per-topic level.

The v1 commits to K = 20. This is on the low-interpretable end of the
sensible range and is the conventional choice in published screenplay-
LDA work (Eliashberg 2014 used K = 19; Mariani 2020 used K = 25).
Diagnostic step #4 below computes per-topic coherence and reports it
alongside the ablation row; if mean coherence is below the
literature's "interpretable" band of 0.4 to 0.5, the planning
conversation can rerun at K = 30 or K = 15 in v2 polish.

`K` is exposed as a configuration knob (`TopicFeatureConfig.n_topics`)
so a v2 sensitivity sweep is a one-line change.

---

## 3. Proposed features (22 total)

The 22 features split into three sub-blocks: 20 topic proportions,
1 topic-distribution concentration measure, and 1 dominant-topic
indicator (encoded numerically rather than one-hot to avoid blowing
up the matrix).

### 3.1 Topic proportions (20 features)

**`topic_00_proportion` through `topic_19_proportion`.** For each
film, the 20 topic proportions sum to 1.0 by construction (the LDA
posterior gives a Dirichlet-distributed probability vector over
topics). Each proportion measures the share of the film's dialogue
content allocated to that topic.

The K = 20 topics will not have human-readable names at extraction
time; the diagnostic step assigns each topic a label by inspecting
the top-10 most representative words and producing a
`reports/tables/phase3_topic_labels.csv` table. The numeric labels
`topic_00 ... topic_19` are stable across runs (LDA's topic ordering
is implementation-defined but deterministic given a fixed
`random_state`); the human labels in the diagnostic table are for
the report.

The 20 features are correlated by construction (the proportions sum
to 1.0), which means the linear-family models will absorb one
dimension as redundant. The diagnostic step reports the empirical
maximum pairwise correlation; if it exceeds the 0.85 review
threshold, the proposal documents the redundancy without dropping a
column (preserving column count is preferable to picking which
arbitrary topic to drop).

### 3.2 Topic distribution concentration (1 feature)

**`topic_concentration_entropy`.** Shannon entropy of the per-film
topic distribution, rescaled to `[0, 1]` by dividing by `log(K)`.
A film with all probability mass on one topic has entropy 0; a film
with uniform distribution across the 20 topics has entropy 1.

The mechanism: a high-concentration film (entropy near 0) is
focused on one or two topics; a low-concentration film (entropy near
1) is topically diffuse. The hypothesis is that focused films are
more clearly positioned for a target audience, which feeds revenue
through marketing efficiency, while diffuse films may struggle to
find a clear audience. This is the topic-group analog of the
sentiment volatility-concentration feature.

This feature is not a linear combination of the 20 topic proportions
(it is a function of their second moment), so tree-based models can
extract it but linear-model-derived features cannot.

### 3.3 Dominant topic (1 feature)

**`topic_dominant_id`.** The index of the topic with the highest
proportion (an integer from 0 to 19). This captures the "single most
important subject" of the film at a coarser granularity than the 20-
dimensional distribution.

Encoded as an integer rather than one-hot for two reasons. First,
one-hot would add 20 sparse binary columns and inflate the feature
matrix without adding information beyond what the 20 proportions
already carry. Second, several of the four model families
(particularly HistGB) handle integer-encoded categorical features
natively. The downside is that linear models will treat the integer
as an ordinal, which is incorrect; the diagnostic step reports the
linear-family OOF impact of including vs excluding this feature so
the trade-off is visible.

If the diagnostic step shows linear-family performance degrades when
the integer is included, the proposal v2 will switch this feature to
a small set of "top-3 dominant-topic indicators" (binary flags for
the three most-common dominant topics across the corpus) rather than
the integer encoding.

---

## 4. Pre-registered expected lift

These are predictions made before implementation. After
implementation, the actual lift over the Phase 3a revised dialogue-
only floor is recorded alongside these predictions in
`reports/tables/phase3_ablation.csv`. The pre-registered bands apply
to the linear family's OOF numbers (the historical reference for
proposal-side pre-registration).

The mechanism behind the predictions: topic features are the first
group whose information has plausible orthogonality to genre. A
"family-vs-relationships" topic and a "law-and-justice" topic may
each appear across multiple genres and within multiple genres, which
gives the model a way to discriminate between films that the genre
dummy alone cannot. The literature priors (Eliashberg 2014,
Mariani 2020) suggest modest but real lift on rating-style outcomes
and stronger lift on revenue-style outcomes.

### Regression target `log_roi`

Predicted lift in OOF metrics (linear family):

* **RMSE: -0.030 to -0.005** (lower is better; reduction band).
* **MAE: -0.025 to -0.005**.
* **CVRMSE: -0.025 to -0.005**.

The lower end of the band reflects the genre-residual hypothesis:
even if topic information is orthogonal to genre, the corpus size
limits how reliably it can be extracted. The upper end is in the
range published topic-modelling-on-screenplay work has reported.

### Classification target `roi_gt_1` (gross-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: 0.000 to +0.015**.
* **PR-AUC: 0.000 to +0.015**.
* **F1: 0.000 to +0.005**.
* **log-loss: -0.020 to 0.000** (reduction band).

Mechanism: the 80% positive base rate continues to make this target
signal-thin. Topic features may help slightly on the unprofitable
minority by capturing subject matter that audiences reliably reject
(certain niche topics).

### Classification target `roi_gt_2` (net-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: +0.015 to +0.040**.
* **PR-AUC: +0.010 to +0.030**.
* **F1: 0.000 to +0.020**.
* **log-loss: -0.025 to -0.005** (reduction band).

Mechanism: this target carries the most headroom. Subject matter is
strongly correlated with the blockbuster-versus-mid-budget split (e.g.
superhero topics, family-adventure topics dominate the blockbuster
class). The pre-registered AUC band is the widest of the three groups
proposed so far, reflecting both the genre-orthogonality of the
mechanism and the literature priors.

### Combined expectations

If the topic group lands in its predicted band, this is the first
positive standalone result of Phase 3b and a good signal that the
overall Phase 3b methodology is calibrated correctly. If it lands
null, that strengthens the case for the Phase 3c combinations sub-
phase, where topic-plus-character-network and topic-plus-embedding
combinations may surface signal that none of the standalone runs can.

---

## 5. Acknowledgement of out-of-scope features (defensibility)

Three classes of topic-related features are considered and explicitly
deferred.

### 5.1 Topics on action text

The lexical group computed MTLD and Flesch-Kincaid on dialogue and
action in parallel; the lexical handoff documented that the action-
versus-dialogue distinction was paying for itself dimensionally.
Topics on action text are conceptually defensible (action prose
describes settings, props, and physical action that would surface
locale-style topics like "urban-night", "rural-domestic"). Scoped
out for parsimony in v1: 22 features per channel, two channels, 44
features total is too many for n = 1,199. Defensible future addition
if the dialogue-channel topics show strong lift in the ablation.

### 5.2 Topic-derived sentiment polarities

Some topic-modelling work decomposes sentiment by topic ("this film
is positive on the family topic but negative on the work topic").
The infrastructure for this is the cross-product of the topic group
and the sentiment group, which is more naturally evaluated in the
Phase 3c combinations sub-phase than as a standalone feature here.

### 5.3 Hierarchical / dynamic LDA

Hierarchical LDA (Blei et al. 2003) and dynamic topic models (Blei
and Lafferty 2006) are mature alternatives to flat LDA. Neither is
necessary for a 1,199-document corpus and both add substantial
implementation complexity. Scoped out without ambiguity.

---

## 6. Feasibility concerns

### 6.1 Tokenization and preprocessing

The lexical group already commits to NLTK `word_tokenize` and the
ensure-NLTK-resources helper. The topic group reuses both. Additional
preprocessing for LDA:

* **Stopword removal:** NLTK English stopword list (~180 function
  words). Standard for LDA.
* **Lowercase:** applied at tokenization time (already in the lexical
  group's tokenizer).
* **Punctuation and short-token filter:** drop tokens that are pure
  punctuation or shorter than 3 characters (drops one- and two-letter
  contractions and noise). Standard.
* **Lemmatization:** off by default. The proposal v2 may revisit if
  the diagnostic step shows topics that are obvious morphological
  variants (e.g., a "love-loved-loving" topic). v1 keeps it off
  because lemmatization adds a dependency (TextBlob, already pulled
  in by `nrclex`) and the corpus is large enough that surface-form
  topics should be coherent without it.
* **Vocabulary thresholds:** `min_df = 5` (drop words appearing in
  fewer than 5 films), `max_df = 0.5` (drop words appearing in more
  than 50% of films). Standard for LDA on small-corpus screenplay
  text.
* **N-grams:** unigrams only in v1. Bigrams add vocabulary size and
  the diagnostic step often shows diminishing returns on small
  corpora.

### 6.2 No-leakage discipline (CRITICAL)

LDA is the first feature group in this phase whose feature
computation depends on the data distribution. The discipline is
non-negotiable per `PROJECT_CONTEXT.md` Section 6:

* The `CountVectorizer` and `LatentDirichletAllocation` estimators
  are fit on **training-fold tokens only** (the 1,199 train-split
  films).
* The fitted vocabulary, document-term matrix, and LDA model are
  saved to `data/processed/topic_model_artifacts/` for downstream
  inference.
* The 1,713-film topic-distribution matrix is computed by applying
  `transform` on every film's tokens, using the train-fitted
  vocabulary and topic-word distributions.
* The cal and test films contribute zero information to the fitted
  vocabulary or topic-word distributions.

This is the same discipline the brief reserves for LDA explicitly
(Section 1, "LDA topic models, sentence-embedder fine-tuning [...]
are fit on training data only"). The implementation will document
the leak-prevention mechanism prominently in
`src/features/topic.py` and the proposal v2 will include a
`tests/test_topic.py` test asserting that `transform` produces the
same output regardless of cal/test inclusion at fit time.

### 6.3 Computational cost

LDA on 1,199 films × ~20,000-vocabulary documents × K = 20 topics
runs in 30 to 90 seconds on a single CPU using scikit-learn's
variational Bayes solver with `learning_method="batch"` and 10
iterations. The full feature-extraction pipeline (tokenization +
vectorization + LDA fit + transform on all 1,713 films + per-film
feature derivation) is well under 5 minutes. Substantially cheaper
than the lexical group's 600 seconds.

### 6.4 `data_quality_flag` films

Phase 2 flagged 30 films whose source XML encodes the entire
screenplay as fewer than 10 scenes. The dialogue tokens for these
films are still well-defined, just concentrated in fewer scene
boundaries. Topic features are whole-screenplay aggregates that do
not depend on scene structure, so the flagged films use the same
topic pipeline as the rest of the corpus. The diagnostic step
spot-checks the flagged films' topic distributions to confirm they
behave like unflagged films.

### 6.5 Reproducibility

Scikit-learn's LDA accepts a `random_state` argument; the project
standard seed 42 is used. With identical input tokens and identical
preprocessing, the fitted model is byte-identical across reruns.

### 6.6 Empty-text dialogue filter

Phase 2 Tier 1.3 fix and the lexical / sentiment groups already apply
the empty-text filter at every dialogue iteration. The topic
extraction inherits this filter via the same `_dialogue_text` helper
introduced in `src.features.lexical`.

### 6.7 Corpus-size sensitivity for LDA

LDA is known to produce noisy or degenerate topics on small corpora.
The 1,199-document training set is at the lower end of the range
where LDA is recommended; the diagnostic step measures topic
coherence empirically and the proposal v2 will revisit `K` if the
diagnostics surface a coherence issue.

---

## 7. Implementation sketch

### 7.1 Module layout

* **Module:** `src/features/topic.py`.
* **Inputs at compute time:** `ParsedScreenplay` objects from
  `data/processed/screenplays_parsed.pkl`, plus the train-split IMDb
  ID list from `data/processed/split_assignments.parquet`.
* **External dependencies:** `scikit-learn` (already installed).
* **Output:** a `pd.DataFrame` indexed by `imdb_id` with the 22
  feature columns above.

### 7.2 Public API

```python
def fit_topic_model(
    parsed_corpus: dict[str, ParsedScreenplay],
    train_ids: Sequence[str],
    cfg: TopicFeatureConfig | None = None,
) -> FittedTopicModel: ...

def compute_topic_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    fitted: FittedTopicModel,
    cfg: TopicFeatureConfig | None = None,
) -> pd.DataFrame: ...
```

`fit_topic_model` consumes train-split parsed screenplays and
produces a `FittedTopicModel` dataclass holding the
`CountVectorizer`, the `LatentDirichletAllocation`, and the topic-
labels table. `compute_topic_features` takes a fitted model and a
parsed-corpus dictionary (which may contain cal and test films) and
returns the per-film feature matrix.

### 7.3 Configuration knobs (`TopicFeatureConfig`, frozen dataclass)

* `n_topics`: 20.
* `min_df`: 5.
* `max_df`: 0.5.
* `n_lda_iterations`: 10.
* `learning_method`: `"batch"`.
* `random_state`: 42.
* `remove_stopwords`: True.
* `min_token_length`: 3.

### 7.4 Determinism

All operations deterministic given input tokens, the configuration,
and the random seed. Re-running produces byte-identical output.

### 7.5 Testing

* Smoke test: fit on the first 100 train films, transform on the
  first 10 films, assert non-NaN finite output of shape (10, 22).
* Unit test: assert the 20 topic proportions sum to 1.0 within
  numerical tolerance.
* Unit test: assert `topic_concentration_entropy` is in `[0, 1]`.
* Unit test: assert `topic_dominant_id` matches `argmax` of the 20
  proportions.
* No-leakage test: fit a model on `train_ids[:100]`, transform on
  `train_ids[:10]`. Then fit a second model on `train_ids[:100] +
  test_ids[:10]`, transform on `train_ids[:10]`. The two transform
  outputs must match within numerical tolerance for the train
  films, asserting that adding test films at fit time does not
  change train-film inference.
* Integration test: fit on the full 1,199 train films, transform on
  the full 1,713 corpus, assert (1,713, 22) shape and no all-NaN
  columns.

### 7.6 Multi-family ablation through `save_run`

A new `src/experiments/run_topic_ablation.py` mirrors the lexical
and sentiment runners. The augmented matrix joins structural
baseline features and the new topic features. Lift is computed
against the **Phase 3a revised dialogue-only floor**, not against
any prior group's augmented matrix. This matches the standalone-
lift methodology established in the lexical handoff. Joint
evaluation (topic + sentiment, topic + lexical, etc.) is the
domain of the Phase 3c combinations sub-phase.

---

## 8. Post-implementation diagnostic checks

Each check has a specific threshold that, if tripped, prompts a
review before the ablation row gets appended.

1. **Topic coherence (UMass coherence on top-10 topic words).**
   Threshold: mean coherence below 0.4 prompts a `K` revisit.
   Coherence is computed on the train-fold document-term matrix
   only.
2. **Topic-label diagnostic table.** For each of the 20 topics,
   list the top-10 most representative words. Saved to
   `reports/tables/phase3_topic_labels.csv` with a manually
   editable `human_label` column.
3. **Pairwise correlations between topic proportions and structural
   baseline features.** Threshold: |r| > 0.85 prompts a pair
   review.
4. **Pairwise correlations between topic proportions and genre
   dummies.** Informational. The genre-residual hypothesis predicts
   strong correlations here. The diagnostic surfaces which topics
   most overlap with which genres so the report can describe the
   overlap quantitatively.
5. **`data_quality_flag` films vs unflagged.** Spot-check the topic
   distributions of the 24 train-split flagged films. Threshold:
   feature-level z-score difference greater than 1.0 is flagged
   for review.
6. **Topic-distribution dominance.** Per film, fraction of films
   where one topic captures more than 50% of the probability mass.
   Diagnostic only.
7. **Univariate target correlations.** Computed on the train split
   only. If any topic feature exceeds |r| = 0.10 with any target,
   noted in the handoff (matching the lexical and sentiment
   diagnostic pattern).
8. **Leakage smoke test (also a unit test):** confirm that adding
   cal or test films to the fit step does not change the train-
   film transform outputs.

---

## 9. References

* Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). "Latent
  Dirichlet Allocation." *Journal of Machine Learning Research*, 3,
  993-1022.
* Blei, D. M., and Lafferty, J. D. (2006). "Dynamic topic models."
  *International Conference on Machine Learning*. Cited as the
  scoped-out alternative.
* Eliashberg, J., Hui, S. K., and Zhang, Z. J. (2014). "Assessing
  the Future Box Office Performance of Movies Before Their
  Production." *Management Science*, 60(2), 379-396. Used K = 19
  topics on a screenplay corpus; closest direct methodological
  precedent.
* Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure." *arXiv preprint*. Cited as the
  scoped-out alternative backend.
* Mariani, M. M., Buhalis, D., Czakon, W., and Vitouladiti, O.
  (2020). [Subtitle-based topic modeling on a 3,000-film corpus.]
  *Nature Humanities & Social Sciences Communications.*
* Newman, D., Lau, J. H., Grieser, K., and Baldwin, T. (2010).
  "Automatic evaluation of topic coherence." *NAACL*. Source for
  the UMass coherence diagnostic.

---

Proposal v1 for the **topic** feature group is ready. Please bring
to the planning conversation for review before implementation.
