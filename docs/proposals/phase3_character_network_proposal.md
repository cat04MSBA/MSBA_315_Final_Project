# Phase 3 Character Network Feature Proposal (v1)

**Group:** Character network (4 of 5 in the Phase 3b incremental ablation)
**Status:** Awaiting planning-conversation review
**Date:** 2026-05-03

This is the v1 draft of the character-network proposal. It is
structured to match the lexical, sentiment, and topic proposals so
the planning conversation can review it under the same template.
Polish guidance returned by the planning conversation will be folded
into a v2 before implementation begins, matching the pattern used
for the previous groups.

---

## 1. Why character network fourth

The character-network group sits fourth in the Phase 3b ablation
queue. The ordering reflects three considerations.

**It is the strongest remaining candidate for genre orthogonality.**
The two previous null verdicts (lexical, sentiment) both attributed
their flatness to information overlap with genre dummies and
structural counts. Character-network features are different in kind:
they describe the *structure* of how characters interact (cast size,
ensemble vs lead-driven, density of interactions), not the *content*
of what they say. A 30-character ensemble blockbuster and a
3-character chamber drama have very different graph structure, and
that structure is captured only at the coarsest level by the genre
dummies. The hypothesis going in is that this group will produce the
first positive standalone lift of Phase 3b.

**The Phase 2 parser already laid the groundwork.** The Tier 1.x
character-name normalization fixes (canonical character names,
copyright-header rejection, mistagged-content filtering) directly
support character-graph construction. The fixes were partially
motivated by anticipating a character-network feature group; cashing
that work in here is the natural payoff.

**It is computationally moderate.** Graph construction and metric
extraction on 1,713 films, each with median 56 characters and 130
scenes, runs in single-digit minutes on a single CPU using NetworkX
out of the box. Cheaper than embeddings, comparable to topic.

---

## 2. The substantive design question: graph construction

Character-network feature design hinges on one substantive choice:
how do you build the graph in the first place. That choice
determines what every downstream metric measures.

### 2.1 Two building-block decisions

**Decision A — Edge definition.** Two characters share an edge if
they appear together in at least one scene. The edge is weighted by
the number of shared scenes.

The alternative is dialogue-adjacency: an edge between A and B if A
and B speak consecutively (B's line follows A's line within the same
scene). Adjacency is more granular but considerably more sensitive
to the parser's flow-tracking; scene co-occurrence is more robust
on the corpus and matches the unit the screenplay's scene boundaries
naturally provide. The v1 commits to **scene co-occurrence with
edge weights from shared-scene counts**.

**Decision B — Significance filter.** A character is included in
the graph only if they deliver at least 5 non-empty dialogue lines.
This drops the long tail of incidental characters (extras,
one-line walk-ons) whose presence in a scene is more often a
parser artifact or a noise signal than a structurally meaningful
node.

Phase 2's Tier 1.3 fix already requires that a character must have
delivered at least one non-empty dialogue line to count toward
`n_unique_characters`. The 5-line threshold here is the same idea
applied at higher granularity, scoped specifically to graph
construction. Films with fewer than 5 significant characters after
the filter trigger a fallback path described in Section 6.

The threshold is a config knob; the diagnostic step measures
sensitivity at thresholds 3, 5, and 10 to surface whether the
choice is robust.

### 2.2 What the graph looks like in practice

A typical 1,200-film median: 56 distinct character tags pre-Tier-1,
filtered down to roughly 12-25 significant characters with the
5-line threshold. Roughly 60-150 edges, depending on how many
scenes the characters share. Density (edges / possible edges) is
typically in the 0.20 to 0.45 range. Films with collapsed scene
structure (the `data_quality_flag` group; see Section 6) produce
degenerate graphs where everyone shares every scene; these are
handled via the fallback path.

---

## 3. Proposed features (12 total)

The 12 features split into four sub-blocks: cast structure, density
and connectivity, lead-character dominance, and graph topology.

### 3.1 Cast structure (3 features)

**`network_n_significant_characters`.** Number of characters who
delivered at least 5 non-empty dialogue lines. This is conceptually
adjacent to the structural-baseline `n_unique_characters` feature
but with a quality filter. The diagnostic step verifies the
correlation with `n_unique_characters` is strong but not 1.0; if
it is too high (|r| > 0.95), the feature is dropped as redundant.
Log-transformed (`log1p`) at the model-input boundary by the
trainer's existing structural-feature treatment.

**`network_lead_role_count`.** Number of characters whose dialogue-
line count places them in the top decile of their film's
significant cast. A "small lead-role count" signals an ensemble
work; a "large lead-role count relative to cast size" signals an
ensemble; a "small lead-role count" signals lead-driven structure.
Captures information that cast size alone does not.

**`network_dialogue_gini`.** Gini coefficient of dialogue-line
counts across the film's significant characters. A value near 0
means lines are distributed equally across the cast (true
ensemble); a value near 1 means most lines come from one or two
characters (lead-driven). This is the cleanest single-number
summary of "how concentrated is the dialogue load".

### 3.2 Density and connectivity (3 features)

**`network_density`.** Number of edges divided by the number of
possible edges in a simple undirected graph (`E / (N * (N - 1) / 2)`).
A low-density graph means characters mostly interact with a
specific subset; a high-density graph means most characters share
scenes with most others. The mechanism for predictive lift: dense
graphs may signal ensemble-driven plotting; sparse graphs may signal
parallel storylines that the audience can latch onto more cleanly.

**`network_n_components`.** Number of connected components in the
significant-character graph. A film with multiple disconnected
character clusters (`n_components > 1`) has parallel storylines;
films with one component have a unified plot. Captures structural
information that both density and gini miss.

**`network_mean_clustering_coefficient`.** Mean local clustering
coefficient across the significant-character nodes. High clustering
means characters tend to share scenes in tight clusters (groups of
three or four mostly-co-occurring characters); low clustering
means characters interact via "bridge" characters who connect
otherwise-separate cliques. Captures the difference between "many
distinct social groups" and "one large interconnected cast".

### 3.3 Lead-character dominance (3 features)

**`network_top1_dialogue_share`.** Fraction of total dialogue lines
delivered by the top-1 character (the character with the most
lines). High values (above 0.30) signal a single-protagonist
structure; low values (below 0.10) signal a true ensemble. This is
the simplest single-number measure of "lead dominance" and is
expected to correlate with the gini coefficient but capture
specifically the top-of-distribution behaviour.

**`network_top3_dialogue_share`.** Fraction delivered by the top-3
characters combined. Captures whether a film has 1-3 leads with
the bulk of the lines (top-3 share above 0.65) versus distributed
across more characters.

**`network_top1_eigenvector_centrality`.** Eigenvector centrality
of the top-1 character (by dialogue lines) in the significant-
character graph. Captures structural centrality, not just dialogue
volume; a character who shares scenes with many other central
characters scores higher than one who shares scenes only with
peripheral characters.

### 3.4 Graph topology (3 features)

**`network_modularity`.** Newman-Girvan modularity computed via
networkx's `greedy_modularity_communities` algorithm. A high-
modularity graph splits cleanly into community subgraphs (parallel
storylines, distinct social groups); a low-modularity graph is
more uniformly mixed. The mechanism: ensemble films often have
high modularity because they sustain parallel storylines that the
characters in each storyline cluster around.

**`network_max_betweenness_centrality`.** Maximum betweenness
centrality across all significant-character nodes. A high value
indicates a single "bridge" character who connects otherwise-
separate communities (characteristic of mentor / messenger / fixer
roles in ensemble films). Captures structural information that
modularity does not.

**`network_diameter`.** Diameter of the largest connected
component (longest shortest-path between any two nodes). A high
diameter means characters are linked through long chains of
intermediate characters (sprawling ensembles); a low diameter
means most characters are within 1-2 scenes of each other (tight
casts). Computed only on the largest component when the graph has
multiple components, to keep the metric well-defined.

---

## 4. Pre-registered expected lift

These predictions are made before implementation. After
implementation, the actual lift over the Phase 3a revised dialogue-
only floor is recorded alongside these predictions in
`reports/tables/phase3_ablation.csv`. The pre-registered bands
apply to the linear family's OOF numbers.

The mechanism for predictive lift: character-network features
capture the structural shape of the cast, which the genre dummies
encode only crudely. Action and Adventure films skew toward larger,
ensemble-leaning casts; Drama and Romance skew toward smaller,
lead-driven casts. But within Drama there is meaningful variation
between chamber-drama (3-5 cast) and ensemble-drama (12-20 cast),
and similar within other genres. This cast-structure-within-genre
variation is what the features try to capture.

Of the five Phase 3b groups, character network has the highest
prior probability of producing a positive standalone lift. The
predicted lift bands therefore lean wider than the previous two
groups.

### Regression target `log_roi`

Predicted lift in OOF metrics (linear family):

* **RMSE: -0.040 to -0.010** (lower is better; reduction band).
* **MAE: -0.030 to -0.010**.
* **CVRMSE: -0.030 to -0.010**.

Mechanism: cast structure correlates with production scale (large
casts cost more to assemble) and with audience-targeting clarity
(single-protagonist films have a single point of identification).
Both feed log_roi.

### Classification target `roi_gt_1` (gross-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: 0.000 to +0.015**.
* **PR-AUC: 0.000 to +0.015**.
* **F1: 0.000 to +0.005**.
* **log-loss: -0.020 to 0.000** (reduction band).

Mechanism: as before, the 80% positive base rate keeps available
headroom small. Cast-structure features may help on the
unprofitable minority by surfacing films with budgets disproportion-
ate to their cast size.

### Classification target `roi_gt_2` (net-profitable)

Predicted lift in OOF metrics (linear family):

* **AUC-ROC: +0.020 to +0.045**.
* **PR-AUC: +0.015 to +0.035**.
* **F1: 0.000 to +0.020**.
* **log-loss: -0.025 to -0.005** (reduction band).

Mechanism: the blockbuster-versus-mid-budget split aligns strongly
with cast structure (blockbusters tend to have large multi-lead
ensembles or single-protagonist franchises with sprawling
supporting casts). The pre-registered AUC band is the widest of
the four groups proposed so far.

### Combined expectations

If character network lands in its predicted band, this is the first
strong positive standalone result of Phase 3b and validates the
genre-orthogonality interpretation of the previous two null
verdicts. If it also lands null, the two-and-a-half-out-of-five
null verdict pattern (lexical null, sentiment null, character
network null, topic / embedding still pending) seriously
reinforces the case that any single-group standalone lift on this
corpus and at this baseline is hard to extract, and the Phase 3c
combinations sub-phase becomes the primary venue for any positive
result.

---

## 5. Acknowledgement of out-of-scope features (defensibility)

Four classes of character-network-related features are considered
and explicitly deferred.

### 5.1 Per-character sentiment trajectories

The sentiment proposal Section 5.3 deferred per-character sentiment
trajectories explicitly to the character-network group on the
grounds that this group's empty-text filter already gates on
character-level integrity. The v1 of this proposal does not include
per-character sentiment because it would couple two feature groups
at the standalone-lift step (you couldn't say whether the lift
came from cast structure or from per-character emotional dynamics).
Defensible Phase 3c combinations addition. Phase 3b standalone keeps
the boundaries clean.

### 5.2 Directed dialogue graphs

A directed graph where an edge from A to B reflects A speaking
immediately before B (a dialogue-turn pattern) captures who-talks-
to-whom information that the symmetric scene-cooccurrence graph
discards. Directed graphs add notational complexity (asymmetric
centrality, in-degree vs out-degree) and would inflate the feature
count. Scoped out for parsimony in v1; defensible future addition
if the symmetric graph features show strong lift.

### 5.3 Triadic motif counts

Counting specific small subgraph patterns (triangles, stars,
paths-of-length-3) is a standard graph-analysis tool. The motif
zoo is large; selecting a small set without overfitting requires
a separate methodology pass. Scoped out for v1.

### 5.4 Temporal graph evolution

Building per-act or per-quartile graphs and measuring how the
network evolves through the film is conceptually compelling. The
infrastructure for it is the cross-product of the character-network
group and the sentiment group's quartile-pooling pattern, which is
more naturally evaluated in Phase 3c combinations. Scoped out for
the standalone row.

---

## 6. Feasibility concerns

### 6.1 NetworkX dependency

NetworkX is added to `requirements.txt` (a small pure-Python
package, no compilation, MIT license). Standard for graph-feature
work.

### 6.2 Tier 1.x character-name normalization

The Phase 2 parser already strips parenthetical-suffix variants
(`TONY (CONT'D)` → `TONY`), rejects copyright-header pseudo-
characters, and applies a non-empty-dialogue requirement to
character counting. The character-network features inherit all
three fixes via the parsed-screenplay pickle's structured
representation. No additional name-normalization is performed.

### 6.3 `data_quality_flag` films (CRITICAL)

This is the group most affected by the `data_quality_flag` issue.
The 30 films with degenerate scene structure (`Elvis`,
`12 Angry Men`, `Manhattan Murder Mystery`, etc.) have all their
dialogue concentrated in 1-9 scenes, which means scene-cooccurrence
edges become trivially dense or trivially sparse, depending on
how the source XML grouped the dialogue.

The v1 commits to **NaN-fallback for graph features on flagged
films**. When a film's `data_quality_flag` is True (or when the
significant-character graph has fewer than 5 nodes after
filtering), the 12 features are emitted as NaN. The trainer's
imputer (`SimpleImputer(strategy="median")`) handles them at
training time; HistGB handles NaN natively.

Three alternatives were considered and rejected:

* **Fall back to dialogue-adjacency edges.** Adds an asymmetric
  feature-construction path that complicates interpretation. NaN
  fallback is cleaner.
* **Drop flagged films from the train split.** Loses 24 of 1,199
  training rows for one feature group; the rest of the pipeline
  uses these films.
* **Treat the entire screenplay as a single scene.** Produces
  degenerate complete graphs with density 1.0 for every flagged
  film, distorting the feature distribution.

The diagnostic step empirically validates that the unflagged films'
feature distributions are sensible after the NaN-fallback decision.

### 6.4 Significance threshold sensitivity

The 5-line threshold is the proposal's commitment, but the
diagnostic step measures the cast-size and density features at
thresholds 3, 5, and 10 to surface whether the choice is robust.
If sensitivity is high (more than 30% of films change category at
threshold 3 vs threshold 5), the proposal v2 will revisit. Default
stays at 5.

### 6.5 Computational cost

NetworkX graph construction and metric extraction on a single film
runs in milliseconds; the full corpus runs in under 60 seconds on
a single CPU. Modularity via `greedy_modularity_communities` is the
slowest single step (single-digit milliseconds per film).
Eigenvector centrality requires the graph to be connected; for
multi-component graphs the implementation computes it on the
largest component and returns NaN for nodes outside it.

### 6.6 Empty-graph and small-graph edge cases

A film with fewer than 2 significant characters cannot define an
edge. A film with no dialogue at all (highly unusual but
possible after the empty-text filter) produces no nodes. The
features handle these:

* `network_n_significant_characters`: integer count, well-defined.
* All other features: NaN if `network_n_significant_characters < 2`.

### 6.7 Determinism

NetworkX algorithms used here are deterministic given the input
graph and node-ordering. The implementation explicitly sorts nodes
by character name before graph construction so node-ordering is
not an implicit source of non-determinism.

---

## 7. Implementation sketch

### 7.1 Module layout

* **Module:** `src/features/character_network.py`.
* **External dependency:** `networkx` (added to `requirements.txt`).
* **Inputs at compute time:** `ParsedScreenplay` objects from
  `data/processed/screenplays_parsed.pkl`. The
  `data_quality_flag` per film is read from the master Parquet at
  `data/processed/films_joined.parquet`.
* **Output:** a `pd.DataFrame` indexed by `imdb_id` with the 12
  feature columns above plus 1 diagnostic column
  (`_n_dropped_minor_characters`).

### 7.2 Public API

```python
def build_character_graph(
    parsed: ParsedScreenplay,
    cfg: CharacterNetworkConfig,
) -> nx.Graph: ...

def compute_character_network_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    data_quality_flags: pd.Series,
    cfg: CharacterNetworkConfig | None = None,
) -> pd.DataFrame: ...
```

### 7.3 Configuration knobs (`CharacterNetworkConfig`, frozen dataclass)

* `min_dialogue_lines_per_character`: 5.
* `top_lead_decile_fraction`: 0.10.
* `treat_flagged_films_as_nan`: True.

### 7.4 Determinism

All features deterministic given the input graph and the
configuration. Sorted node-ordering at construction time.

### 7.5 Testing

* Smoke test: compute on the first 10 films, assert (10, 13) shape
  and no infinities.
* Unit test: 3-character chain graph (A-B-C) has density = 2/3,
  diameter = 2, modularity = 0 for `greedy_modularity_communities`.
* Unit test: a 5-character complete graph has density 1.0,
  diameter 1, modularity 0.
* Unit test: a graph with two disjoint 3-character cliques has
  `n_components` = 2 and modularity > 0.5.
* Unit test: a flagged film returns NaN for all 12 model features.
* Integration test: full corpus, assert (1,713, 13) shape, every
  unflagged film has at least the 12 model features non-NaN.

### 7.6 Multi-family ablation through `save_run`

A new `src/experiments/run_character_network_ablation.py` mirrors
the lexical, sentiment, and topic runners. Standalone-lift
methodology: the augmented matrix joins structural baseline
features and the new character-network features. Lift is computed
against the **Phase 3a revised dialogue-only floor**.

---

## 8. Post-implementation diagnostic checks

1. **Significant-character count distribution.** Per-film
   distribution of `network_n_significant_characters` across the
   corpus. Sanity check: median should be in the 12-25 range; if
   wildly different from `n_unique_characters` (the parser's
   pre-filter count), investigate.
2. **Pairwise correlations between character-network features and
   structural baseline features.** Threshold: |r| > 0.85 prompts a
   pair review. Particular concern: `network_n_significant_characters`
   ↔ `n_unique_characters` and ↔ `n_dialogue_lines` are both
   expected to correlate strongly. Above 0.95 prompts a feature
   drop.
3. **Pairwise correlations within character-network features.**
   Threshold: |r| > 0.85. Particular concern:
   `network_dialogue_gini` ↔ `network_top1_dialogue_share` and
   `network_density` ↔ `network_mean_clustering_coefficient` are
   expected to correlate; above 0.85 prompts a drop decision.
4. **Significance-threshold sensitivity.** Re-compute the cast-
   size and density features at thresholds 3, 5, and 10. Report
   the fraction of films that change category at each pair of
   thresholds.
5. **`data_quality_flag` films vs unflagged.** All flagged films'
   features should be NaN by construction. Verify via assertion.
6. **Empty-graph rate.** Number of films with fewer than 2
   significant characters. Diagnostic only; expected to be small.
7. **Univariate target correlations.** Computed on the train split.
   If any character-network feature exceeds |r| = 0.10 with any
   target, noted in the handoff (matches the lexical / sentiment
   pattern).
8. **Genre overlap diagnostic.** Per-genre means and standard
   deviations of the 12 features. Surfaces whether character-
   network features correlate with genre as the genre-residual
   hypothesis would predict.

---

## 9. References

* Newman, M. E. J., and Girvan, M. (2004). "Finding and evaluating
  community structure in networks." *Physical Review E*, 69(2).
  Source for the modularity feature.
* Clauset, A., Newman, M. E. J., and Moore, C. (2004). "Finding
  community structure in very large networks." *Physical Review E*,
  70(6). Source for the greedy modularity algorithm used in
  networkx.
* Wasserman, S., and Faust, K. (1994). *Social Network Analysis:
  Methods and Applications.* Cambridge University Press. Standard
  reference for graph centrality measures.
* Hagberg, A. A., Schult, D. A., and Swart, P. J. (2008).
  "Exploring network structure, dynamics, and function using
  NetworkX." *SciPy*. Library citation.

---

Proposal v1 for the **character network** feature group is ready.
Please bring to the planning conversation for review before
implementation.
