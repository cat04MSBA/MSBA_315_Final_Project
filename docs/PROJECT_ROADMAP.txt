# MSBA315 Project — Roadmap

> This document outlines the nine phases of the project at a stable level:
> what each phase is, what success looks like, what it produces, and how
> phases connect.
>
> The roadmap describes phases at the level that should not change as the
> project progresses. Specific methodology choices within each phase
> (which models, which hyperparameters, which features) are decided in
> the per-phase brief drafted just before execution.
>
> Read this together with `PROJECT_CONTEXT.md` (project framing, data,
> decisions) and `CLAUDE_CODE_GUIDELINES.md` (engineering standards).

---

## Project Shape at a Glance

The project builds a four-layer script triage system in three blocks of work:

```
Block A — Data foundation
  Phase 1: Verify data feasibility
  Phase 2: Build the data pipeline
  Phase 3: Extract features

Block B — The four layers
  Phase 4: Layer 1 — Core prediction model
  Phase 5: Layer 2 — Calibrated uncertainty
  Phase 6: Layer 3 — Asymmetric-cost decision
  Phase 7: Layer 4 — SHAP explanations

Block C — Integration and delivery
  Phase 8: End-to-end integration and evaluation
  Phase 9: Report and presentation
```

Phases run sequentially. Each phase reads inputs from artifacts saved by
previous phases and produces artifacts that downstream phases consume.
Mandatory checkpoints exist at the end of Phases 1, 4, 5, and 7 — see
`PROJECT_CONTEXT.md` Section 11.

---

## Block A — Data Foundation

### Phase 1 — Data Feasibility Verification

**Goal:** Confirm we have enough usable data to support the project before
investing in the pipeline. Answer the question: "How many films will end
up with all four data points (screenplay, IMDb rating, budget, revenue)?"

**Why this phase exists:** Every downstream decision depends on corpus
size and data quality. Discovering at Phase 4 that the corpus is too small
would invalidate weeks of work. Phase 1 answers this question with low
effort before the work scales up.

**What it delivers:**
- Confirmed working dataset size after MovieSum × TMDB join
- Year and genre distribution of the joined set
- Documented decision on whether to proceed with MovieSum-only, expand
  with IMSDb supplementary scripts, or revise scope
- A small Python utility module for loading each raw dataset
- Updated `PROJECT_CONTEXT.md` Section 5 (Data Summary)

**Definition of done:** A documented number for the joined-corpus size,
plus a recommendation to the planning conversation on which corpus
configuration to commit to.

**Mandatory checkpoint at end:** Yes. Decision: proceed with planned
scale, expand, or reduce scope.

---

### Phase 2 — Data Pipeline

**Goal:** Build the production pipeline that turns raw downloaded data
into a clean, joined working dataset ready for feature extraction.

**Why this phase exists:** Phase 1 validates feasibility on samples;
Phase 2 commits to processing the full data once. Subsequent phases
read the artifact this phase produces and never re-run the pipeline.

**What it delivers:**
- Loaders for each raw source (MovieSum, IMDb-TMDB ratings dataset, optionally IMSDb fallback)
- A joining module that produces the master film table
- A screenplay parser that extracts structured content from MovieSum's
  XML format (scenes, characters, dialogue, stage directions)
- The joined working dataset saved as a Parquet file in `data/processed/`
- A `docs/DATA_NOTES.md` describing the corpus characteristics
  (distributions, missingness, edge cases, sanity-check observations)

**Definition of done:** A single Parquet file containing one row per film,
with columns for IMDb ID, title, year, genre, budget, revenue, IMDb rating,
and the parsed screenplay structure. Reproducible from raw data with one
script invocation.

**Mandatory checkpoint at end:** No (data understanding is captured in
DATA_NOTES, no strategic decision needed).

---

### Phase 3 — Feature Extraction

**Goal:** Convert each screenplay into a fixed-length numerical feature
vector that downstream models can consume.

**Why this phase exists:** This is where domain knowledge enters the
pipeline. The features chosen here determine what the model can learn.
The phase is treated as its own stage (rather than folded into modeling)
because feature engineering is a substantial methodological contribution
in its own right and deserves explicit reasoning in the report.

**What it delivers:**
- A feature extraction module per feature group:
  - Lexical features (vocabulary diversity, word/sentence length, totals)
  - Sentiment features (VADER scores, sentiment arc across scenes,
    sentiment variance)
  - Topic features (LDA topic distributions, fit on training data only)
  - Embedding features (sentence-transformer embeddings, averaged
    appropriately)
  - Structural features (scene count, character count, dialogue-to-action
    ratio, character interaction graph metrics)
- A feature matrix saved as Parquet, with rows indexed by IMDb ID and
  columns being all extracted features
- Diagnostic plots for each feature group, with written interpretations
- A train/calibration/test split file (saved as IMDb ID lists), so all
  later phases use the same split

**Definition of done:** Feature matrix saved, split file saved, every
feature group has a diagnostic plot and interpretation in the phase
summary.

**Critical methodology note:** The train/calibration/test split happens
in this phase, before any feature fitting that involves the data
distribution (LDA, scaling, etc.). All such fitting is on training data
only and applied to the other splits.

**Mandatory checkpoint at end:** No.

---

## Block B — The Four Layers

### Phase 4 — Layer 1: Core Prediction Model

**Goal:** Train a model that predicts film outcomes (IMDb rating and
box office tier) from the dialogue features.

**Why this phase exists:** This is the foundation that all other layers
build on. Calibration without a working model is meaningless;
asymmetric-cost decisions without predictions are nothing.

**What it delivers:**
- A simple baseline (Linear / Ridge Regression) to establish that
  dialogue features carry signal at all
- A benchmark of candidate models trained on identical splits:
  tree-based (XGBoost, LightGBM, Random Forest), linear (Ridge, Lasso),
  and optionally a fine-tuned transformer (DistilBERT) if the corpus
  size justifies it
- 5-fold cross-validation results for each candidate, with bootstrapped
  confidence intervals on the metrics
- Hyperparameter optimization on the most promising candidates
- Diagnostic plots (predicted vs. actual, residual analysis,
  per-genre and per-era breakdowns)
- Saved trained model objects for the strongest candidates

**Definition of done:** A benchmark table comparing candidate models on
both outcome variables (rating regression and box office classification),
with statistical significance assessed on the differences. Strongest
candidates serialized.

**Mandatory checkpoint at end:** Yes. Decisions: which model is primary,
which outcome variable is primary for the cost-decision layer, and
whether to invest in ensemble work before moving to Phase 5.

---

### Phase 5 — Layer 2: Calibrated Uncertainty

**Goal:** Wrap the chosen model's output with calibrated confidence
intervals that empirically deliver the coverage they claim.

**Why this phase exists:** Layer 3 (decisions) cannot work correctly
without calibrated uncertainty. A model that is overconfident will
recommend Greenlight when it should abstain; one that is underconfident
will refer everything to humans, defeating the purpose. Calibration is
also the single largest piece of novelty in the project relative to
existing film-prediction literature.

**What it delivers:**
- A held-out calibration set carved from the training pool
- Conformal prediction wrapping the chosen model (`mapie` library or
  equivalent), producing prediction intervals at chosen confidence levels
- Empirical validation that intervals deliver promised coverage
  (reliability diagrams, coverage tables across confidence levels)
- Analysis of interval widths: where is the model confident, where is
  it uncertain, does the pattern make sense
- Comparison of calibration quality before and after temperature scaling
  (or equivalent technique appropriate to the model family)

**Definition of done:** A calibrated wrapper around the primary model
that, given a script, produces a prediction with a confidence interval.
Empirical coverage verified within tolerance of nominal coverage.

**Mandatory checkpoint at end:** Yes. Decision: is calibration good
enough to support the decision layer, or does the underlying model
need rework?

---

### Phase 6 — Layer 3: Asymmetric-Cost Decision

**Goal:** Convert calibrated predictions into actionable decisions
(Greenlight / Pass / Refer to human reader) using a cost matrix that
reflects studio risk asymmetries.

**Why this phase exists:** This is where the system stops being a
predictor and becomes a decision-support tool. It is also where the
modest predictive accuracy of dialogue-only features gets compensated
for: a system that abstains on uncertain cases can be useful even when
the underlying model is not strikingly accurate.

**What it delivers:**
- A documented cost matrix with sourced default values (cost of a flop,
  cost of a missed hit, cost of human reader time) and a justification
  for each
- A decision function that takes a calibrated prediction and the cost
  matrix and returns one of three actions
- System-level evaluation metrics: coverage at threshold, cost savings
  versus naive baselines (always-pass, always-greenlight, read-everything),
  decision quality among non-abstained decisions
- Sensitivity analysis: how does system behavior change as the cost
  matrix is varied? At what cost ratios does the system become useless,
  or change recommendation patterns?
- A small set of example outputs on representative test films

**Definition of done:** A complete decision pipeline that takes a script
through all three layers and outputs a justified action with a confidence
range and a cost-based rationale. Sensitivity analysis demonstrates
robustness.

**Mandatory checkpoint at end:** No.

---

### Phase 7 — Layer 4: SHAP Explanations

**Goal:** Produce interpretable explanations for each decision, ideally
at scene level so that explanations are actionable to writers.

**Why this phase exists:** A "Pass" recommendation without explanation
is useless to a writer. The actionable-feedback layer is what makes the
system valuable as a writing tool, not just a gatekeeper. It is also
the third novelty hook (along with calibration and asymmetric cost).

**What it delivers:**
- TreeSHAP attributions on the primary model for the test set
- Global feature importance (which features matter most across the corpus)
- Per-film attributions (why did this specific film get this prediction)
- Scene-level attributions: for each film, which scenes pushed the
  prediction up, which pushed it down. Achieved either by per-scene
  feature recomputation or scene-removal counterfactuals, depending on
  feature structure
- A formatted explanation report template that combines decision,
  confidence, and top scene-level reasons into a single output
- Validation that scene-level attributions are stable (consistent
  across model variants, not artifacts of noise)

**Definition of done:** Given any test-set film, the system can output
a decision plus a written explanation identifying the top scenes
driving the prediction in each direction. Explanations are stable
across reruns.

**Mandatory checkpoint at end:** Yes. Decision: are scene-level
explanations meaningful and stable enough to keep, or do we fall back
to feature-level only?

---

## Block C — Integration and Delivery

### Phase 8 — End-to-End Integration and Evaluation

**Goal:** Combine all four layers into a single pipeline, evaluate the
full system on the held-out test set, and produce the example outputs
that will appear in the report and presentation.

**Why this phase exists:** Until now, each layer has been built and
tested in relative isolation. Phase 8 validates that the whole pipeline
works coherently end-to-end and produces honest final numbers. The
held-out test set is touched only here.

**What it delivers:**
- An end-to-end pipeline function: input is a parsed screenplay, output
  is the full triage report (decision + confidence interval + scene
  explanations)
- Final test-set evaluation across all relevant metrics: predictive
  performance, calibration coverage, decision-level cost savings,
  attribution quality
- Error analysis: where does the system fail, and is the failure pattern
  interpretable? (By genre, era, budget tier, screenplay length, etc.)
- Optional: out-of-distribution validation on the Cornell Movie-Dialogs
  corpus, decided based on time and Phase 4 results
- A small set of curated example outputs on well-known films, prepared
  for the presentation
- A simple results dashboard (notebook or HTML report) that lets
  reviewers inspect system behavior

**Definition of done:** Final test-set numbers are computed and frozen.
End-to-end pipeline reproduces deterministically. Example outputs are
saved and ready for inclusion in slides and the report.

**Mandatory checkpoint at end:** No (results are what they are at this
point; discussion of meaning happens in Phase 9).

---

### Phase 9 — Report and Presentation

**Goal:** Produce the final deliverables: the merged Jupyter notebook,
the report PDF, and the presentation slides.

**Why this phase exists:** All prior phases have been building the
substance. Phase 9 is where it gets packaged for the audience that
matters: the professor, the class, and prospective readers.

**What it delivers:**
- Final report (≤10 pages) following the structure required by the
  course: Abstract, Introduction, Literature Review, Methodology,
  Results, Conclusion. Drawn from per-phase summaries plus integrative
  writing tying findings to research questions.
- Final Jupyter notebook: the merge of per-phase Python scripts and
  per-phase summary documents, organized by topic for narrative
  coherence rather than by chronological order of phases.
- Presentation slides (10-15 slides). Lead with example outputs of
  the system on real films, not with confusion matrices. Methodology
  and results in service of explaining what the system does.
- Peer evaluation form completed.

**Definition of done:** All three deliverables submitted on time,
each representing the project at the level the rubric expects.

**Mandatory checkpoint at end:** N/A (project complete).

---

## How Phases Connect

The handoff between phases is always a saved artifact on disk plus a
written summary. There is no implicit shared state.

| From | To | Handoff artifact |
|---|---|---|
| Phase 1 | Phase 2 | Decision document on which corpus configuration to use |
| Phase 2 | Phase 3 | Joined working dataset (Parquet) |
| Phase 3 | Phase 4 | Feature matrix + train/cal/test split (Parquet + lists) |
| Phase 4 | Phase 5 | Trained primary model (joblib/pickle) |
| Phase 5 | Phase 6 | Calibrated model wrapper |
| Phase 6 | Phase 7 | Decision-pipeline function plus cost-matrix configuration |
| Phase 7 | Phase 8 | SHAP explainer + scene-attribution function |
| Phase 8 | Phase 9 | Final test results + example outputs + error analysis |

---

## Mandatory Checkpoints

Four phases end with mandatory escalation to the planning conversation
before the next phase starts. These are documented in
`PROJECT_CONTEXT.md` Section 11. Briefly:

- **End of Phase 1:** corpus size and quality. Decision on scope.
- **End of Phase 4:** model and outcome benchmarks. Decision on which
  to use as primary, and on ensemble work.
- **End of Phase 5:** calibration validation. Decision on whether
  calibration is sound or model needs rework.
- **End of Phase 7:** SHAP results. Decision on whether scene-level
  explanations hold up.

At each checkpoint, Claude Code writes the phase summary, prepares
specific questions for the planning conversation, and pauses execution
until the user returns with answers.

---

## Appendix: Mapping to the Course Rubric

The roadmap is organized around what we are building, not around the
grading criteria. This appendix shows how the project's deliverables map
to the rubric so that nothing required is missed.

### Report rubric (out of 100)

- **Abstract (5pts):** Drafted in Phase 9 from the project context and
  final results.
- **Introduction (5pts):** Drafted in Phase 9. Material from
  `PROJECT_CONTEXT.md` Section 1 (project framing) feeds it directly.
- **Literature review (20pts):** Drafted in Phase 9 with sources
  accumulated across the project. Key references identified and tracked
  in `PROJECT_CONTEXT.md` Section 10 from the start.
- **Methodology (20pts):** Built up across Phases 2-7 in per-phase
  summaries; merged and edited in Phase 9. Each phase summary's "Why we
  did it that way" section is the raw material.
- **Optimization (20pts):** Phase 4 (model selection, hyperparameter
  tuning), Phase 5 (calibration), and Phase 6 (cost-threshold tuning)
  each contribute. Optimization effort is documented per phase in the
  summaries.
- **Results (20pts):** Phase 8 produces the headline test-set numbers
  and error analysis. Per-phase results from Phases 4-7 also contribute.
- **Conclusion (10pts):** Phase 9 synthesizes findings and limitations.
  Survivorship bias and other limitations are acknowledged explicitly
  per `PROJECT_CONTEXT.md` Section 6.

### Presentation rubric (out of 100, equally weighted)

- **Slide content visible and self-explainable:** Phase 9 deliverable.
- **Confidence and clarity in delivery:** Practice in Phase 9.
- **Background, problem, and objectives clear:** Drawn from
  `PROJECT_CONTEXT.md` Section 1.
- **Discussion of related work:** Drawn from the Phase 9 literature review.
- **Methods and contributions clear and justified:** Drawn from per-phase
  methodology rationales.
- **Thorough experimentation:** Documented across Phases 4-7.
- **Detailed results analysis and recommendations:** Phase 8 + Phase 9.
- **Time management:** Practice run in Phase 9.
- **Q&A confidence:** Comes from having executed the project rather than
  delegated it.

### Code rubric (out of 100)

- **Readability (15pts):** Enforced by `CLAUDE_CODE_GUIDELINES.md` Section 3.
- **Structure (15pts):** Enforced by the project structure in
  `PROJECT_CONTEXT.md` Section 7 and the per-phase summary template.
- **Effective and modular code (15pts):** Enforced by the "no copy-pasted
  blocks" and "single-purpose functions" rules in
  `CLAUDE_CODE_GUIDELINES.md`.
- **Code efficiency (15pts):** Saved intermediate artifacts prevent
  redundant computation; vectorization and library use are standard
  practice.
- **Optimization Efforts (25pts):** Phase 4 (hyperparameters), Phase 5
  (calibration), Phase 6 (decision thresholds), Phase 7 (attribution
  granularity) each contribute documented optimization work.
- **Reproducibility (15pts):** Enforced by the seed and path conventions
  in `CLAUDE_CODE_GUIDELINES.md` Section 3.
