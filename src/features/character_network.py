"""Phase 3b character-network feature group.

Computes 12 graph-structural features per film, plus 1 diagnostic
column. Features are derived from a per-film character-cooccurrence
graph: nodes are characters who deliver at least 5 non-empty
dialogue lines; edges connect characters who appear in the same
scene; edge weights are shared-scene counts.

The feature design follows the planning-conversation-approved v1
proposal at ``docs/proposals/phase3_character_network_proposal.md``.
See that document for per-feature rationale and pre-registered lift
bands.

Phase 2 Tier 1.x parser fixes (character-name normalization,
copyright-header rejection, non-empty-dialogue requirement for
unique characters) are inherited via the parsed-screenplay pickle
and feed graph construction directly. No additional name-
normalization is performed here.

`data_quality_flag` films receive NaN for all 12 model features
because their collapsed scene structure makes scene-cooccurrence
edges meaningless. The trainer's `SimpleImputer(median)` handles
NaNs at fit time on the linear / KNN / SVM families; HistGB
handles NaN natively. The diagnostic column
`_n_dropped_minor_characters` records how many characters were
filtered out by the 5-line threshold per film.

Dependencies
------------

* ``networkx`` (added to requirements.txt for this group).

References
----------

* Newman, M. E. J. (2004). "Modularity and community structure in
  networks." Source for the modularity feature.
* Wasserman & Faust (1994). Standard reference for graph
  centrality measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from src.data.parse_screenplay import ParsedScreenplay
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterNetworkConfig:
    """Knobs for the character-network feature pipeline.

    Defaults reproduce proposal v1 Section 7.3.
    """

    # Minimum number of non-empty dialogue lines required for a
    # character to count as "significant" and enter the graph.
    min_dialogue_lines_per_character: int = 5

    # Lead-role threshold: a character is a "lead" if their dialogue-
    # line count places them in the top decile of their film's
    # significant cast.
    top_lead_decile_fraction: float = 0.10

    # Whether to emit NaN for all 12 model features on data-quality-
    # flagged films (films with collapsed scene structure).
    treat_flagged_films_as_nan: bool = True

    # Floor on graph size below which all metrics are NaN. A graph
    # with fewer than 2 nodes cannot define an edge.
    min_significant_characters: int = 2


CHARACTER_NETWORK_FEATURE_COLUMNS: tuple[str, ...] = (
    # Cast structure (3)
    "network_n_significant_characters",
    "network_lead_role_count",
    "network_dialogue_gini",
    # Density and connectivity (3)
    "network_density",
    "network_n_components",
    "network_mean_clustering_coefficient",
    # Lead-character dominance (3)
    "network_top1_dialogue_share",
    "network_top3_dialogue_share",
    "network_top1_eigenvector_centrality",
    # Graph topology (3)
    "network_modularity",
    "network_max_betweenness_centrality",
    "network_diameter",
    # Diagnostic (1)
    "_n_dropped_minor_characters",
)

DIAGNOSTIC_ONLY_COLUMNS: frozenset[str] = frozenset({
    "_n_dropped_minor_characters",
})

MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in CHARACTER_NETWORK_FEATURE_COLUMNS if c not in DIAGNOSTIC_ONLY_COLUMNS
)


# ---------------------------------------------------------------------------
# Per-film helpers
# ---------------------------------------------------------------------------


def _significant_characters_with_lines(
    parsed: ParsedScreenplay, min_lines: int,
) -> tuple[dict[str, int], int]:
    """Return ({character_name: line_count}, n_dropped_minor_characters).

    Empty-text dialogue is filtered (Phase 2 Tier 1.3). Returned
    dict maps significant character names to their non-empty
    dialogue-line count. The integer is the number of characters
    seen in the parsed screenplay but filtered out by the
    ``min_lines`` threshold (a diagnostic).
    """
    line_counts: dict[str, int] = {}
    for scene in parsed.scenes:
        for character, text in scene.dialogue_units:
            if not character or not text or not text.strip():
                continue
            line_counts[character] = line_counts.get(character, 0) + 1
    significant = {c: n for c, n in line_counts.items() if n >= min_lines}
    n_dropped = len(line_counts) - len(significant)
    return significant, n_dropped


def _scene_cooccurrence_edges(
    parsed: ParsedScreenplay, significant: set[str],
) -> dict[tuple[str, str], int]:
    """Build edge-weight dictionary keyed by sorted (a, b) tuples.

    Two significant characters share an edge with weight equal to
    the number of scenes in which both deliver at least one
    non-empty dialogue line.
    """
    edges: dict[tuple[str, str], int] = {}
    for scene in parsed.scenes:
        scene_characters: set[str] = set()
        for character, text in scene.dialogue_units:
            if not character or not text or not text.strip():
                continue
            if character in significant:
                scene_characters.add(character)
        # All pairs of co-present significant characters in this scene.
        chars = sorted(scene_characters)
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                key = (chars[i], chars[j])
                edges[key] = edges.get(key, 0) + 1
    return edges


def _build_graph(
    line_counts: dict[str, int], edges: dict[tuple[str, str], int],
) -> nx.Graph:
    """Construct a weighted undirected graph with one node per significant character."""
    g = nx.Graph()
    # Nodes added in sorted order for determinism downstream.
    for character in sorted(line_counts):
        g.add_node(character, line_count=line_counts[character])
    for (a, b), w in edges.items():
        g.add_edge(a, b, weight=w)
    return g


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _gini(values: Sequence[int]) -> float:
    """Gini coefficient of a non-negative integer sequence.

    Returns 0 if all values are equal (perfect equality), 1 if all
    mass is on one element. Defined for non-empty inputs only.
    """
    if not values:
        return float("nan")
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n == 1:
        return 0.0
    if arr.sum() == 0:
        return 0.0
    # Standard formula: G = (2 * sum(i * x_i) - (n + 1) * sum(x)) / (n * sum(x))
    indices = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(indices * arr) - (n + 1) * arr.sum())
        / (n * arr.sum())
    )


def _mean_clustering_coefficient(g: nx.Graph) -> float:
    """Mean local clustering coefficient over all nodes.

    NetworkX's ``average_clustering`` handles disconnected graphs
    correctly. Returns NaN for graphs with fewer than 3 nodes
    (clustering requires triangle-completion potential).
    """
    if g.number_of_nodes() < 3:
        return float("nan")
    return float(nx.average_clustering(g))


def _modularity(g: nx.Graph) -> float:
    """Newman-Girvan modularity via greedy community detection.

    Uses ``networkx.community.greedy_modularity_communities``. Returns
    NaN for graphs with no edges (modularity undefined).
    """
    if g.number_of_edges() == 0:
        return float("nan")
    try:
        communities = list(nx.community.greedy_modularity_communities(g))
        return float(nx.community.modularity(g, communities))
    except Exception:
        # Fallback: undefined modularity (e.g., singleton components).
        return float("nan")


def _max_betweenness(g: nx.Graph) -> float:
    """Max betweenness centrality across all nodes.

    Returns 0 for graphs with fewer than 3 nodes (no intermediate
    paths possible). NetworkX returns 0 for isolated nodes; the max
    is over the connected component nodes' values.
    """
    if g.number_of_nodes() < 3:
        return 0.0
    try:
        bc = nx.betweenness_centrality(g)
        return float(max(bc.values())) if bc else 0.0
    except Exception:
        return float("nan")


def _diameter_largest_component(g: nx.Graph) -> float:
    """Diameter of the largest connected component.

    Returns NaN for empty graphs and 0 for single-node largest
    components (degenerate but well-defined).
    """
    if g.number_of_nodes() == 0:
        return float("nan")
    components = list(nx.connected_components(g))
    if not components:
        return float("nan")
    largest = max(components, key=len)
    sub = g.subgraph(largest)
    if sub.number_of_nodes() < 2:
        return 0.0
    try:
        return float(nx.diameter(sub))
    except Exception:
        return float("nan")


def _top1_eigenvector_centrality(g: nx.Graph, line_counts: dict[str, int]) -> float:
    """Eigenvector centrality of the top-1 character (by line count).

    The top-1 character is identified by the highest dialogue-line
    count across all significant characters. NetworkX requires the
    graph to be non-empty and at least one connected component; for
    multi-component graphs the implementation operates on the
    component containing the top-1 character. Returns NaN if the
    eigenvector solver fails to converge (rare for small graphs).
    """
    if g.number_of_nodes() == 0 or not line_counts:
        return float("nan")
    top1 = max(line_counts.items(), key=lambda kv: kv[1])[0]
    if top1 not in g:
        return float("nan")
    # Operate on the component that contains the top-1 character.
    component_of_top1 = nx.node_connected_component(g, top1)
    sub = g.subgraph(component_of_top1)
    if sub.number_of_nodes() < 2:
        return 0.0
    try:
        ec = nx.eigenvector_centrality(sub, max_iter=1000, tol=1e-6)
        return float(ec[top1])
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        # Try numpy backend as a fallback for difficult cases.
        try:
            ec = nx.eigenvector_centrality_numpy(sub)
            return float(ec[top1])
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Per-film orchestration
# ---------------------------------------------------------------------------


def _nan_features() -> dict[str, float]:
    """Return a feature dict where every model column is NaN."""
    return {c: float("nan") for c in CHARACTER_NETWORK_FEATURE_COLUMNS}


def _compute_one_film(
    parsed: ParsedScreenplay,
    is_flagged: bool,
    cfg: CharacterNetworkConfig,
) -> dict[str, float]:
    """Compute the 12 features (plus 1 diagnostic) for one screenplay."""
    if cfg.treat_flagged_films_as_nan and is_flagged:
        out = _nan_features()
        # The diagnostic column is still meaningful even on flagged
        # films; record the dropped-character count.
        line_counts, n_dropped = _significant_characters_with_lines(
            parsed, cfg.min_dialogue_lines_per_character,
        )
        out["_n_dropped_minor_characters"] = float(n_dropped)
        return out

    line_counts, n_dropped = _significant_characters_with_lines(
        parsed, cfg.min_dialogue_lines_per_character,
    )
    n_significant = len(line_counts)

    if n_significant < cfg.min_significant_characters:
        # Empty / single-node graph: most features undefined.
        out = _nan_features()
        out["network_n_significant_characters"] = float(n_significant)
        out["_n_dropped_minor_characters"] = float(n_dropped)
        return out

    edges = _scene_cooccurrence_edges(parsed, set(line_counts.keys()))
    g = _build_graph(line_counts, edges)

    # ---- Cast structure ----
    sorted_line_counts = sorted(line_counts.values(), reverse=True)
    total_lines = sum(sorted_line_counts)
    top_decile_n = max(1, int(np.ceil(cfg.top_lead_decile_fraction * n_significant)))
    lead_threshold = sorted_line_counts[top_decile_n - 1]
    lead_role_count = sum(1 for n in sorted_line_counts if n >= lead_threshold)

    # ---- Density ----
    n_possible_edges = n_significant * (n_significant - 1) / 2
    density = g.number_of_edges() / n_possible_edges if n_possible_edges > 0 else float("nan")

    # ---- Connectivity ----
    n_components = nx.number_connected_components(g)

    # ---- Lead dominance ----
    top1_share = sorted_line_counts[0] / total_lines if total_lines > 0 else float("nan")
    top3_share = (
        sum(sorted_line_counts[:3]) / total_lines if total_lines > 0 else float("nan")
    )

    # ---- Topology ----
    top1_eig = _top1_eigenvector_centrality(g, line_counts)
    modularity = _modularity(g)
    max_betweenness = _max_betweenness(g)
    diameter = _diameter_largest_component(g)
    mean_clustering = _mean_clustering_coefficient(g)

    return {
        "network_n_significant_characters": float(n_significant),
        "network_lead_role_count": float(lead_role_count),
        "network_dialogue_gini": _gini(sorted_line_counts),
        "network_density": float(density),
        "network_n_components": float(n_components),
        "network_mean_clustering_coefficient": mean_clustering,
        "network_top1_dialogue_share": float(top1_share),
        "network_top3_dialogue_share": float(top3_share),
        "network_top1_eigenvector_centrality": top1_eig,
        "network_modularity": modularity,
        "network_max_betweenness_centrality": max_betweenness,
        "network_diameter": float(diameter),
        "_n_dropped_minor_characters": float(n_dropped),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_character_network_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    data_quality_flags: pd.Series,
    cfg: CharacterNetworkConfig | None = None,
) -> pd.DataFrame:
    """Compute all 12 character-network features (plus 1 diagnostic) per film.

    Parameters
    ----------
    parsed_corpus
        Mapping of ``imdb_id`` to :class:`ParsedScreenplay` objects.
    data_quality_flags
        Series indexed by ``imdb_id`` with bool values; True means
        the film has degenerate scene structure and graph features
        will be set to NaN.
    cfg
        Feature configuration; ``None`` uses defaults.

    Returns
    -------
    pd.DataFrame
        One row per film, indexed by ``imdb_id``, with columns
        matching :data:`CHARACTER_NETWORK_FEATURE_COLUMNS` (12 model
        features + 1 diagnostic column).
    """
    cfg = cfg or CharacterNetworkConfig()

    logger.info(
        "Computing character-network features for %d films (min_lines=%d, "
        "treat_flagged_as_nan=%s)",
        len(parsed_corpus),
        cfg.min_dialogue_lines_per_character,
        cfg.treat_flagged_films_as_nan,
    )

    flag_lookup: dict[str, bool] = {}
    for imdb_id, flag in data_quality_flags.items():
        flag_lookup[str(imdb_id)] = bool(flag)

    records: dict[str, dict[str, float]] = {}
    for i, (imdb_id, parsed) in enumerate(parsed_corpus.items(), start=1):
        is_flagged = flag_lookup.get(str(imdb_id), False)
        records[imdb_id] = _compute_one_film(parsed, is_flagged, cfg)
        if i % 200 == 0:
            logger.info(
                "Character-network features computed: %d / %d",
                i, len(parsed_corpus),
            )

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "imdb_id"
    df = df[list(CHARACTER_NETWORK_FEATURE_COLUMNS)].astype(float)
    logger.info(
        "Character-network feature matrix complete: %d films x %d columns",
        df.shape[0], df.shape[1],
    )
    return df
