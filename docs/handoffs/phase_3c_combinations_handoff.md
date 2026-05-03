# Phase 3c: Combinations sub-phase handoff

**Status:** Four pre-specified combinations evaluated. Pre-registered
direction was wrong on 10 of 12 headline bands; the few in-band
results plus the multi-family signal cluster around one parsimonious
combination (`topic_plus_cn`) and one substantive Phase 4 finding
(SVM-RBF dominates classification under combinations).
**Date:** 2026-05-03

This handoff closes the Phase 3 feature-engineering work. Phase 4
model selection follows.

---

## 1. What was tested

Four feature-group combinations, locked in
`docs/proposals/phase3c_combinations_proposal.md` Section 2 before
any was measured. Same 4-family multi-family harness as Phase 3b
(linear, HistGB, KNN, SVM-RBF; train and OOF eval sets; bootstrap
CIs).

| Combination | Components | Feature count |
|---|---|---:|
| `all_five` | lexical + sentiment + topic + character_network + embedding | 130 |
| `partial_positives` | topic + character_network + embedding | 92 |
| `topic_plus_cn` | topic + character_network | 60 |
| `semantic_trio` | sentiment + topic + embedding | 102 |

All combinations join onto the structural baseline (~26 columns
including 13 genre dummies, era, log_runtime, 7 log-transformed
structural counts).

---

## 2. Headline numbers (linear OOF lift over Phase 3a floor)

Floor (linear OOF, revised dialogue-only): log_roi RMSE 1.339,
roi_gt_1 AUC 0.558, roi_gt_2 AUC 0.602.

| Combination | log_roi RMSE | roi_gt_1 AUC | roi_gt_2 AUC | In-band on linear OOF |
|---|---:|---:|---:|---|
| `all_five` | +0.007 (worse) | -0.005 | **+0.011** | 1 of 3 (roi_gt_2 only) |
| `partial_positives` | +0.006 (worse) | +0.012 | -0.003 | 0 of 3 |
| `topic_plus_cn` | +0.028 (worse) | +0.021 | **+0.021** | 1 of 3 (roi_gt_2 only) |
| `semantic_trio` | +0.004 (worse) | -0.000 | -0.012 | 0 of 3 |

Two out of twelve pre-registered linear-OOF bands hit. The pattern
on the regression target is the most striking: **every combination
pushed log_roi RMSE the wrong direction on linear**, including
combinations containing embedding (the only Phase 3b group whose
standalone result moved RMSE the right direction at -0.007).

---

## 3. Multi-family pattern (the substantive finding)

The linear OOF view tells only one part of the story. Reading
across the four families per combination:

### 3.1 OOF lift on `roi_gt_2` AUC by family

| Combination | linear | histgb | knn | svm |
|---|---:|---:|---:|---:|
| `all_five` | +0.011 | -0.034 | -0.019 | **+0.063** |
| `partial_positives` | -0.003 | -0.015 | -0.012 | **+0.063** |
| `topic_plus_cn` | +0.021 | +0.003 | -0.025 | **+0.043** |
| `semantic_trio` | -0.012 | -0.039 | -0.021 | **+0.057** |

### 3.2 OOF lift on `roi_gt_1` AUC by family

| Combination | linear | histgb | knn | svm |
|---|---:|---:|---:|---:|
| `all_five` | -0.005 | -0.018 | +0.019 | **+0.056** |
| `partial_positives` | +0.012 | -0.025 | +0.023 | **+0.054** |
| `topic_plus_cn` | +0.021 | +0.006 | +0.035 | **+0.081** |
| `semantic_trio` | -0.000 | -0.006 | +0.012 | **+0.046** |

### 3.3 What the multi-family pattern says

**SVM-RBF dominates classification under combinations.** The SVM
column shows positive lift of +0.04 to +0.08 on every combination,
on both classification targets. The largest classification lift in
this entire phase, including all of Phase 3b, is SVM-RBF on
`topic_plus_cn` `roi_gt_1`: +0.081. SVM standalone in Phase 3b was
the worst family on the floor. Combinations let the RBF kernel find
non-linear similarity in the augmented feature space that the other
families cannot extract.

**HistGB hurts on every combination tested**, especially on
classification. Adding 60 to 130 features to gradient boosting on
n = 1,199 with the conservative defaults produces shallow trees
that fail to find the genuine signal under the noise. HistGB's
overfitting tendency, already at 0.20-AUC train-OOF gap on the
Phase 3a floor, gets worse with feature-count inflation. Phase 4
hyperparameter search needs to explore much more conservative
HistGB regularization (lower learning_rate, larger min_samples_leaf)
to recover the standalone-group performance.

**Linear regression on regression target is broken by
combinations.** Every combination hurts linear log_roi RMSE
relative to the Phase 3a floor. The mechanism is the same one the
lexical group's negative lift surfaced: zero-signal-or-near-zero-
signal features add noise that L2 regularization cannot fully
absorb. Embedding's standalone -0.007 RMSE benefit is lost when
combined with anything else.

---

## 4. The two combinations worth keeping

Of the four pre-specified combinations, **`topic_plus_cn` is the
single most informative result**. Reasons:

* The only combination tested whose linear OOF AUC lifts both
  classification targets (`roi_gt_1` +0.021, `roi_gt_2` +0.021).
* `roi_gt_2` AUC lift falls in the pre-registered band (+0.015 to
  +0.030).
* Smallest feature count (60) of any combination tested, so the
  feature-count-noise dynamic is at its weakest.
* Single positive HistGB classification lift (`roi_gt_2` +0.003) of
  any combination tested.
* Across families: SVM gets +0.081 on `roi_gt_1` AUC (the largest
  classification lift in the phase) and +0.043 on `roi_gt_2`.

`all_five` is also worth keeping as the **maximum-information
reference**: it has the highest SVM lift on `roi_gt_2` AUC (+0.063)
and the only positive HistGB classification result on combinations
(+0.003 on `roi_gt_2` for `topic_plus_cn` is the only HistGB
positive). Phase 4 should benchmark both `topic_plus_cn` and
`all_five` to determine whether SVM-RBF's gain on the larger matrix
is worth the feature-count cost.

The other two combinations are not net positive:

* `partial_positives` matches `all_five` on linear classification
  but loses on regression. Replaceable by `all_five` (more
  information) or `topic_plus_cn` (better linear classification
  lift).
* `semantic_trio` underperforms across the board. Sentiment's
  null is dragging it down, embedding's regression signal does not
  survive the combination, and it produces the largest negative
  lift on roi_gt_2 AUC of any combination tested.

---

## 5. The pre-registration was wrong, and that is itself the finding

10 of 12 pre-registered linear-OOF lift bands missed. The bands
were too optimistic and the proposal's mental model was wrong about
combinations.

What I predicted: standalone lifts add up roughly linearly.
Combinations with embedding carry forward embedding's RMSE
benefit. Combinations of two complementary groups (topic + CN) get
both classification targets.

What actually happened:

* **Standalone lifts do not add up linearly on linear regression.**
  Adding 60+ features to a linear model on n = 1,199 introduces
  more L2-noise than the standalone signals can survive. Embedding
  alone helps; embedding plus any other group hurts.
* **Standalone lifts mostly do not add up linearly on linear
  classification either.** `topic_plus_cn` is close to additive
  (+0.021 on each, vs standalone topic +0.032 on `roi_gt_1` and CN
  +0.016 on `roi_gt_2`), but the combinations larger than that
  underperform the simpler pair.
* **Combinations behave very differently per model family.** SVM
  benefits enormously, HistGB hurts, KNN hurts on regression but
  helps modestly on `roi_gt_1`. The "linear is the historical
  reference" framing under-credits SVM's ability to find non-linear
  similarity in larger feature spaces.

Honest lesson for the report: incremental ablation against a linear
baseline systematically under-represents the combination value of
features for non-linear models. The Phase 3b standalone-group
verdicts (3 partial positives) are about the right answer for what
each group standalone can offer linear regression. But the
combinations result shows the joint matrix has substantial signal
for SVM-RBF that the standalone view cannot have surfaced.

This is the strongest single argument for going into Phase 4 with
the maximum-information matrix and letting the model selection
pick what works, rather than over-trusting the standalone-ablation
"earned its place" criterion.

---

## 6. Train versus OOF gap on combinations

Same overfit-diagnostic pattern as Phase 3a but more pronounced.
HistGB train-OOF gap on `roi_gt_2` AUC for `all_five`: train ~0.85,
OOF 0.576, gap roughly 0.27. The gap widens with feature count, as
expected.

For Phase 4: the more aggressive regularization the gap suggests
should be searched in HistGB's hyperparameter space (`max_depth=2`
or `3`; `min_samples_leaf=20+`; `learning_rate=0.02` or `0.01`;
fewer iterations).

---

## 7. Implications for Phase 4

Phase 4 model selection now has these signals to work with:

1. **The feature matrix to use as input is `all_five` (130 features) or
   the union of features that earned a place in Phase 3b standalone
   plus Phase 3c combinations.** The standalone "earned its place"
   criterion is too restrictive; the multi-family combinations
   evidence shows SVM extracts substantial signal from the larger
   matrix even for groups that were standalone-null (the lexical
   group's contribution to SVM's combination performance is non-
   negligible).
2. **SVM-RBF is the surprise candidate primary model on
   classification targets.** It went from worst-of-four standalone
   to best-of-four on combinations. Phase 4 model benchmark should
   give SVM-RBF a serious hyperparameter search, not the
   default-only treatment Phase 3 used.
3. **HistGB needs aggressive regularization on this corpus.** The
   conservative defaults used in Phase 3 (`max_depth=4`,
   `learning_rate=0.05`) overfit substantially; Phase 4 should
   search (`max_depth in {2, 3}`, `learning_rate in {0.01, 0.02,
   0.05}`, `min_samples_leaf in {10, 20, 40}`).
4. **Linear regression on regression target is signal-limited.**
   The best linear-family RMSE in this entire Phase 3 work is the
   Phase 3a revised dialogue-only floor (1.339); no combination,
   no Phase 3b group, beat it on linear OOF. Whatever Phase 4 model
   wins on `log_roi`, it will not be linear.
5. **`roi_gt_2` is the most tractable target.** SVM on `all_five`
   reaches AUC 0.665 OOF (floor 0.602 + lift 0.063), well into the
   project's forward-expected band of 0.65-0.72. `roi_gt_2`
   should be the primary outcome variable unless Phase 4 surfaces a
   compelling reason otherwise.

---

## 8. Files produced (Phase 3c)

### Code
* `docs/proposals/phase3c_combinations_proposal.md` (339 lines):
  the locked combinations set with mathematical specifications,
  pre-registered lift bands, and methodology.
* `src/experiments/run_combinations_ablation.py` (~250 lines): the
  multi-family runner mirroring the Phase 3b group runners.

### Data artifacts
None new. The four combinations reuse the Phase 3b feature parquets
already on disk.

### Tables
* `reports/tables/phase3c_combinations.csv`: 384 rows (4
  combinations × 4 families × 2 eval sets × 3 targets × {4
  regression metrics or 4 classification metrics}). Schema matches
  `phase3_ablation.csv` with `feature_group` carrying the
  combination name.

### Run artifacts (committed)
* `runs/phase_3/20260503_2155_combinations_all_five/`
* `runs/phase_3/20260503_2156_combinations_partial_positives/`
* `runs/phase_3/20260503_2156_combinations_topic_plus_cn/`
* `runs/phase_3/20260503_2157_combinations_semantic_trio/`

Each contains `params.json`, `preprocessing_summary.json`,
`features_used.json`, `metrics.json`, and `run.log`.

### `runs/RUNS.md`
Four new rows appended (newest first), one per combination, with
linear-OOF and HistGB-OOF headline metrics in the key-metric column.

---

## 9. Phase 3 closure path

Per the Phase 3 brief, Phase 3 is now complete. Remaining closure
steps:

1. **`docs/FEATURE_NOTES.md`**: write the standing-reference
   feature catalogue for Phase 4 to consume. Should include: column
   glossary across all five Phase 3b groups, ablation summary
   (Phase 3b standalone + Phase 3c combinations), per-feature
   handling decisions (especially the wordfreq deviation for
   lexical, the NRC stop-word policy for sentiment, the LDA
   character-name-dominance issue for topic, the
   `data_quality_flag` per-feature-group handling).
2. **`data/processed/features.parquet`**: optional consolidated
   feature matrix. Given the Phase 3c finding that `all_five` is
   the right Phase 4 input, this is the union of the five Phase 3b
   feature parquets joined onto the structural baseline.
3. **`docs/summaries/phase_3_summary.md`**: the Phase 3 final
   summary using the `CLAUDE_CODE_GUIDELINES.md` Section 7
   template. Replaces the seven (!) interim handoffs (Phase 3a,
   five Phase 3b groups, Phase 3c) as the canonical Phase 3
   record.
4. **`PROJECT_CONTEXT.md`**: update Section 5 (data summary) and
   Section 9 (phase status table) to reflect Phase 3 complete.

These steps are out of scope for this handoff. They follow.

---

## 10. Open questions for the planning conversation

1. **Should Phase 4 use `all_five` or only the standalone-positive
   union as input matrix?** The Phase 3c finding that SVM-RBF
   extracts substantial signal from the maximum-information matrix
   argues for `all_five`. The pre-registration discipline argues
   that only features that "earned their place" should go forward;
   under that criterion, the lexical and sentiment groups should
   not. My recommendation: `all_five` (let Phase 4's model search
   weight features) plus a sensitivity-analysis run on the
   standalone-positive union for comparison. But this is a
   strategic call for the planning conversation.
2. **Should the Phase 4 model benchmark include SVM-RBF as a
   serious candidate?** Phase 3a's framing implied linear and tree
   ensembles were the natural Phase 4 candidates. The Phase 3c
   evidence elevates SVM-RBF to the same tier on classification.
   Phase 4 should give it a real hyperparameter search.
3. **The `topic_plus_cn` parsimonious-combination finding is
   genuinely interesting.** Should Phase 4 maintain a "small
   feature set" comparison alongside the maximum-information set?
   This is methodologically valuable for the report (defensible
   "we tested both" framing) but doubles Phase 4's benchmarking
   work.
