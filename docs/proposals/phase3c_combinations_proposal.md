# Phase 3c Combinations Sub-Phase Proposal

**Phase:** Phase 3c (combinations sub-phase, closes Phase 3 before Phase 4)
**Status:** Pre-registered set; combinations locked before measurement
**Date:** 2026-05-03

This proposal pre-registers the small set of feature-group
combinations to evaluate against the Phase 3a revised dialogue-only
floor, **before any of them is measured**. The pre-registration
discipline that governed Phase 3b's standalone-group ablation rows
extends here: the combinations set is locked at proposal time and is
not expanded after seeing results. Locking the set is the
multiple-comparisons firewall the methodology relies on.

The decision to add Phase 3c was logged in `PROJECT_CONTEXT.md`
Section 8 (entry dated 2026-05-03 18:30) after the lexical group's
near-null verdict raised the concern that incremental ablation
systematically under-credits any group whose signal partially
overlaps with genre. Phase 3b proceeded as planned; with all five
groups now evaluated standalone (two nulls, three partial
positives), Phase 3c tests whether feature-group combinations
produce additive lift, redundant lift, or interaction lift relative
to the standalone numbers.

---

## 1. What Phase 3c is and is not

**Is:** the same multi-family ablation harness used in Phase 3b
(linear, HistGB, KNN, SVM-RBF; train and OOF eval sets; bootstrap
CIs) applied to feature-group combinations rather than to single
groups. Output: `reports/tables/phase3c_combinations.csv` (one row
per combination × family × eval_set × target × metric, schema
matching `phase3_ablation.csv`).

**Is not:** model selection, hyperparameter tuning, or primary-model
selection. Those are Phase 4 work. The constant-comparator harness
stays unchanged. The output is a feature-decision artifact: which
feature combinations carry signal that survives multi-family
evaluation, feeding into Phase 4's model benchmark.

**Phase 3 close criterion.** When Phase 3c lands, Phase 3 is
complete: split assignments saved, five Phase 3b standalone
ablation rows recorded, four Phase 3c combination rows recorded,
final `FEATURE_NOTES.md` written. Phase 4 begins on the curated
feature set that earned its place either standalone or in
combination.

---

## 2. The pre-specified combinations (locked)

Four combinations. Each tests a specific hypothesis derived from
Phase 3b standalone results.

### 2.1 `all_five`: every Phase 3b group joined onto the structural baseline

**Components:** lexical (14) + sentiment (24) + topic (22) +
character_network (13) + embedding (32) on top of the structural
baseline (~26 columns including genre dummies and log_runtime).
Total feature count approximately 131.

**Hypothesis:** maximum information matrix. If feature signals are
mostly orthogonal, this combination should produce the largest lift.
If signals are mostly redundant, lifts should be similar to the best
single group.

**Why include:** the upper-bound test. Sets the ceiling against
which the more parsimonious combinations are compared.

### 2.2 `partial_positives`: drop the nulls

**Components:** topic + character_network + embedding (the three
groups whose Phase 3b standalone rows showed partial positive lift)
on top of the structural baseline. Total feature count approximately
93 (excludes lexical and sentiment).

**Hypothesis:** removing the two null groups removes noise that
hurt regularized linear models in the lexical and sentiment
ablations. If this combination matches or beats `all_five`, the
nulls were genuinely adding noise; if `all_five` exceeds it, the
nulls were carrying small interaction signal that the standalone
ablation missed.

**Why include:** directly tests the lexical handoff's
genre-residual-signal hypothesis. Distinguishes "nulls are noise"
from "nulls are redundant but harmless."

### 2.3 `topic_plus_cn`: complementary classification targets

**Components:** topic (22) + character_network (13) on top of the
structural baseline. Total feature count approximately 61.

**Hypothesis:** the two classification-target-complementary groups
jointly cover both `roi_gt_1` (topic dominant: standalone lift
+0.032) and `roi_gt_2` (character_network dominant: standalone lift
+0.016). The pair should produce simultaneous AUC lifts on both
targets that neither group achieves alone.

**Why include:** the cleanest pair-effect test. If the joint matrix
matches the union of standalone lifts, signals are additive. If it
exceeds them, there's interaction. If it falls short, the standalone
lifts were partially redundant with structural baseline features
that one group's inclusion already absorbed.

### 2.4 `semantic_trio`: shared versus additive semantic signal

**Components:** sentiment (24) + topic (22) + embedding (32) on top
of the structural baseline. Total feature count approximately 104.

**Hypothesis:** the three semantic groups (sentiment, topic, and
embedding) capture related dimensions of dialogue meaning at
different abstraction levels (per-utterance valence, document-level
topic distribution, distributed vector representation). They may
share signal (combined lift roughly equal to the best individual
lift) or stack additively (combined lift roughly equal to the sum).

**Why include:** tests the planning-conversation original
hypothesis from the Phase 3c decisions-log entry ("these may share
information with each other, so testing them jointly tells us
whether the semantic signal is over-counted by adding all three").
Distinguishing shared from additive matters for Phase 4: shared
signal means picking one semantic group is sufficient; additive
means all three should stay.

### 2.5 What is not in the locked set

Two combinations were considered and excluded for parsimony:

* **`lexical_plus_cn`** (the planning conversation's "structural-
  leaning" hypothesis): tests whether character_network's
  genre-orthogonal signal compensates for lexical's null. Excluded
  because the `all_five` combination implicitly tests this as a
  sub-component, and four combinations is at the centre of the
  3-5 range the original Phase 3c spec suggested.
* **`embedding_alone`** (the friend's "sanity check" suggestion):
  excluded because it is not a combination at all; it is the
  Phase 3b standalone embedding row already in
  `phase3_ablation.csv`.

Pre-registration discipline forbids expanding the set after seeing
results. If a Phase 3c finding suggests a fifth combination would be
informative, it gets surfaced in the handoff as "future test if
Phase 4 results are ambiguous," not added to this run.

---

## 3. Pre-registered expected lift

Predictions made before measurement. Bands are for the **linear
family OOF** numbers (the historical reference), against the Phase
3a revised dialogue-only floor (linear OOF: log_roi RMSE 1.339;
roi_gt_1 AUC 0.558; roi_gt_2 AUC 0.602).

Bands are tighter than the Phase 3b standalone-group bands because
the standalone-group results provide a strong prior on what each
component contributes.

### 3.1 `all_five`

| Target | Metric | Predicted lift band |
|---|---|---|
| log_roi | RMSE | -0.025 to -0.005 (lower is better) |
| roi_gt_1 | AUC | +0.020 to +0.040 |
| roi_gt_2 | AUC | +0.010 to +0.030 |

Mechanism: embedding's standalone -0.007 RMSE lift carries forward
plus or minus interaction effects from the other groups; topic's
+0.032 standalone roi_gt_1 lift roughly carries forward
(possibly diluted by null-group noise); CN's +0.016 standalone
roi_gt_2 lift likewise. The wider bands than partial_positives
account for null-group noise.

### 3.2 `partial_positives`

| Target | Metric | Predicted lift band |
|---|---|---|
| log_roi | RMSE | -0.025 to -0.010 |
| roi_gt_1 | AUC | +0.025 to +0.045 |
| roi_gt_2 | AUC | +0.015 to +0.035 |

Mechanism: same components as `all_five` but cleaner. If the nulls
were noise-additive, this combination matches or exceeds `all_five`.
If the nulls carried small interaction signal, `all_five` exceeds
this combination.

### 3.3 `topic_plus_cn`

| Target | Metric | Predicted lift band |
|---|---|---|
| log_roi | RMSE | -0.005 to +0.010 (lower or near floor) |
| roi_gt_1 | AUC | +0.025 to +0.040 |
| roi_gt_2 | AUC | +0.015 to +0.030 |

Mechanism: embedding excluded, so log_roi RMSE lift is much smaller
or zero. Topic's roi_gt_1 lift and CN's roi_gt_2 lift carry forward
roughly additively.

### 3.4 `semantic_trio`

| Target | Metric | Predicted lift band |
|---|---|---|
| log_roi | RMSE | -0.015 to 0.000 |
| roi_gt_1 | AUC | +0.020 to +0.035 |
| roi_gt_2 | AUC | 0.000 to +0.020 |

Mechanism: embedding's RMSE lift carries forward; topic's roi_gt_1
lift carries; sentiment's standalone null suggests it adds little
or nothing on classification targets. The combined lift on
roi_gt_2 should be smaller than `topic_plus_cn` because CN is
absent.

### 3.5 Relative-rank prediction

If signals are mostly orthogonal: `all_five` > `partial_positives`
roughly, both > `topic_plus_cn` roughly, all three > `semantic_trio`
on classification targets. If signals are heavily shared:
`partial_positives` matches `all_five` (nulls were noise),
`topic_plus_cn` matches `partial_positives` minus embedding's
RMSE contribution, `semantic_trio` underperforms on roi_gt_2.

---

## 4. Comparison point and methodology

**Floor:** Phase 3a revised dialogue-only baseline, linear family
OOF (matching the Phase 3b standalone-group comparison point).
Reading from `reports/tables/phase3a_baseline.csv`,
`feature_set == "dialogue_only_logged"`, `eval_set == "oof"`.

**Lift sign convention:**

* For lower-is-better metrics (MSE, RMSE, MAE, CVRMSE, log-loss):
  negative lift means improvement.
* For higher-is-better metrics (AUC-ROC, PR-AUC, F1): positive lift
  means improvement.

**Metrics reported:** the new project metric set (MSE, RMSE, MAE,
CVRMSE for regression; AUC-ROC, PR-AUC, F1, log-loss for
classification). Both train (in-sample) and OOF (out-of-fold) eval
sets.

**Model families:** all four (linear, HistGB, KNN, SVM-RBF). Same
harness as Phase 3b. SVM produces decision-function scores rather
than calibrated probabilities, so log-loss is `NaN` for SVM rows.

**In-band check:** the `in_predicted_band` column is populated for
the linear-family OOF numbers only, matching the Phase 3b
convention. Other families' lifts are reported but their
predicted-band column stays blank.

---

## 5. Implementation

* **Module:** `src/experiments/run_combinations_ablation.py`. New
  runner, structurally identical to the Phase 3b group runners.
* **Feature matrix construction:** uses `BaselineFeatureConfig`
  with combinations of `include_*` flags. The friend's
  Phase 3b implementation already added flags for all five
  groups (`include_lexical`, `include_sentiment`, `include_topic`,
  `include_character_network`, `include_embedding`), so the four
  combinations above are expressed as different flag combinations on
  the same config dataclass. No changes to `baseline_features.py`
  needed.
* **save_run integration:** one `save_run` block per combination,
  producing four directories under
  `runs/phase_3/<timestamp>_combinations_<combination_name>/`.
* **Output table:** `reports/tables/phase3c_combinations.csv`,
  with the same schema as `phase3_ablation.csv`. The
  `feature_group` column carries the combination name (one of
  `all_five`, `partial_positives`, `topic_plus_cn`,
  `semantic_trio`) instead of a single-group name.
* **No diagnostic step beyond the standard.** The Phase 3b group
  diagnostics (cross-correlations, OOV rates, archetype variance)
  were specific to feature-construction questions. Phase 3c uses
  pre-computed feature parquets and does not introduce new feature
  construction; the diagnostic surface area shrinks to the
  ablation-output level (in-band yes/no per family per metric per
  combination).

---

## 6. Estimated runtime

Each combination runs the same 4-family × 2-eval-set × 3-target
harness. Approximate timing per combination:

| Combination | Feature count | Estimated time |
|---|---:|---:|
| `topic_plus_cn` | ~61 | 1-2 min |
| `semantic_trio` | ~104 | 2-3 min |
| `partial_positives` | ~93 | 2-3 min |
| `all_five` | ~131 | 3-5 min |

Total: approximately 10-15 minutes. SVM-RBF is the slowest in each
combination (time scales roughly with feature count squared on the
RBF kernel).

---

## 7. What gets surfaced in the handoff

The Phase 3c handoff (`docs/handoffs/phase_3c_combinations_handoff.md`)
will report:

* The four combinations, each with the linear-family OOF lift on
  the headline metrics, train-vs-OOF gap, and the in-band-or-not
  check against the pre-registered band.
* A multi-family table per combination (4 families × headline
  metrics).
* A relative-rank narrative against the prediction in Section 3.5.
* The verdict on each combination's contribution: which combinations
  (if any) earned a place in the Phase 4 input matrix beyond what
  the partial-positive standalone groups already justify.
* Open questions for the planning conversation, if any combination's
  lift falls dramatically outside the pre-registered band.

---

## 8. Phase 3 closure path after Phase 3c

Per the Phase 3 brief, when Phase 3c lands:

1. Update `FEATURE_NOTES.md` with the final feature decisions from
   Phases 3a, 3b, and 3c.
2. Save the consolidated `data/processed/features.parquet` as the
   union of features that earned their place either standalone or
   in combination.
3. Write the Phase 3 final summary at
   `docs/summaries/phase_3_summary.md` using the
   `CLAUDE_CODE_GUIDELINES.md` Section 7 template.
4. Update `PROJECT_CONTEXT.md` Section 5 (data summary) and Section
   9 (phase status) to reflect Phase 3 complete.
5. Notify the user that Phase 3 is complete and Phase 4 should
   begin.

These steps are out of scope for this proposal. They follow Phase 3c
in sequence.
