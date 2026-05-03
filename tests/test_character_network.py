"""Tests for ``src.features.character_network``.

Synthetic graph-structure tests on hand-built ParsedScreenplay
fixtures, plus an integration test on real corpus data.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.parse_screenplay import ParsedScreenplay, Scene
from src.features.character_network import (
    CHARACTER_NETWORK_FEATURE_COLUMNS,
    DIAGNOSTIC_ONLY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    CharacterNetworkConfig,
    _gini,
    compute_character_network_features,
)
from src.utils import paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenplay_from_scenes(
    imdb_id: str, scenes: list[list[tuple[str, str]]],
) -> ParsedScreenplay:
    """Build a ParsedScreenplay where each scene's dialogue is a list of (char, text)."""
    scene_objs = tuple(
        Scene(
            scene_number=i,
            stage_direction="",
            scene_description="",
            dialogue_units=tuple(units),
        )
        for i, units in enumerate(scenes, start=1)
    )
    return ParsedScreenplay(
        imdb_id=imdb_id,
        scenes=scene_objs,
        n_scenes=len(scene_objs),
        n_unique_characters=len({c for sc in scene_objs for c, _ in sc.dialogue_units}),
        n_dialogue_lines=sum(len(sc.dialogue_units) for sc in scene_objs),
        total_dialogue_chars=sum(len(t) for sc in scene_objs for _, t in sc.dialogue_units),
        total_stage_direction_chars=0,
        total_scene_description_chars=0,
        total_action_chars=0,
        dialogue_to_action_ratio=1.0,
        dialogue_to_total_text_ratio=1.0,
        parse_warnings=tuple(),
    )


def _three_char_chain_screenplay(imdb_id: str = "tt_chain") -> ParsedScreenplay:
    """A→B→C chain: A and B share scene 1, B and C share scene 2.

    Each character delivers >= 5 lines so they all pass the
    significance threshold.
    """
    line = "Hello there."
    s1 = [("A", line)] * 6 + [("B", line)] * 6
    s2 = [("B", line)] * 6 + [("C", line)] * 6
    return _make_screenplay_from_scenes(imdb_id, [s1, s2])


def _five_char_complete_screenplay(imdb_id: str = "tt_complete") -> ParsedScreenplay:
    """Five characters all sharing one big scene; each delivers >= 5 lines."""
    line = "Hi."
    units = []
    for c in ("A", "B", "C", "D", "E"):
        units.extend([(c, line)] * 6)
    return _make_screenplay_from_scenes(imdb_id, [units])


def _two_clique_screenplay(imdb_id: str = "tt_two_cliques") -> ParsedScreenplay:
    """Two disjoint 3-character cliques in two separate scenes."""
    line = "Yes."
    s1 = []
    for c in ("A", "B", "C"):
        s1.extend([(c, line)] * 6)
    s2 = []
    for c in ("D", "E", "F"):
        s2.extend([(c, line)] * 6)
    return _make_screenplay_from_scenes(imdb_id, [s1, s2])


def _flagged_screenplay(imdb_id: str = "tt_flagged") -> ParsedScreenplay:
    """A screenplay with one scene; semantically a flagged film."""
    return _five_char_complete_screenplay(imdb_id)


# ---------------------------------------------------------------------------
# Schema / column-count tests
# ---------------------------------------------------------------------------


def test_feature_columns_have_expected_count():
    """12 model features + 1 diagnostic column = 13 total."""
    assert len(CHARACTER_NETWORK_FEATURE_COLUMNS) == 13
    assert len(MODEL_FEATURE_COLUMNS) == 12
    assert len(DIAGNOSTIC_ONLY_COLUMNS) == 1


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------


def test_gini_uniform_returns_zero():
    """All values equal: gini = 0 (perfect equality)."""
    assert _gini([10, 10, 10, 10]) == pytest.approx(0.0)


def test_gini_concentration_returns_high():
    """All mass on one element: gini approaches 1."""
    g = _gini([0, 0, 0, 100])
    assert g > 0.7


def test_gini_empty_returns_nan():
    assert np.isnan(_gini([]))


# ---------------------------------------------------------------------------
# Synthetic graph structure tests
# ---------------------------------------------------------------------------


def test_chain_graph_has_density_two_thirds():
    """3-char chain: 2 edges, 3 possible edges, density = 2/3."""
    parsed = _three_char_chain_screenplay()
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_n_significant_characters"] == 3
    assert row["network_density"] == pytest.approx(2 / 3)
    assert row["network_n_components"] == 1
    assert row["network_diameter"] == 2  # A → B → C


def test_complete_graph_has_density_one():
    """5-char complete graph: density = 1.0, diameter = 1."""
    parsed = _five_char_complete_screenplay()
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_n_significant_characters"] == 5
    assert row["network_density"] == pytest.approx(1.0)
    assert row["network_n_components"] == 1
    assert row["network_diameter"] == 1


def test_two_cliques_has_two_components():
    """Two disjoint 3-cliques: n_components = 2, modularity > 0.4."""
    parsed = _two_clique_screenplay()
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_n_significant_characters"] == 6
    assert row["network_n_components"] == 2
    # Two disjoint triangles: modularity should be high.
    assert row["network_modularity"] > 0.4


def test_flagged_film_returns_nan_for_model_features():
    """When data_quality_flag is True, all 12 model features are NaN."""
    parsed = _flagged_screenplay()
    flags = pd.Series({parsed.imdb_id: True})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    for c in MODEL_FEATURE_COLUMNS:
        assert np.isnan(row[c]), f"flagged film should be NaN on {c}; got {row[c]}"


def test_minor_character_filter_drops_below_threshold():
    """Characters with fewer than min_dialogue_lines_per_character are dropped."""
    line = "Hi."
    # A delivers 5 lines (significant), B delivers 1 line (dropped),
    # C delivers 5 lines (significant). Result: graph has 2 nodes
    # (A, C); the 5-line B node is dropped.
    units = [("A", line)] * 5 + [("B", line)] + [("C", line)] * 5
    parsed = _make_screenplay_from_scenes("tt_filter", [units])
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_n_significant_characters"] == 2
    assert row["_n_dropped_minor_characters"] == 1


def test_too_few_significant_characters_returns_nan():
    """Films with fewer than 2 significant characters return NaN on most metrics."""
    line = "Hello."
    units = [("A", line)] * 6  # only 1 significant character
    parsed = _make_screenplay_from_scenes("tt_solo", [units])
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_n_significant_characters"] == 1
    assert np.isnan(row["network_density"])
    assert np.isnan(row["network_modularity"])


def test_top1_dialogue_share_matches_max_count():
    """top1 share equals top character's lines / total lines."""
    line = "x"
    # A: 10, B: 5, C: 5. total = 20. top1_share = 0.5; top3_share = 1.0.
    units = [("A", line)] * 10 + [("B", line)] * 5 + [("C", line)] * 5
    parsed = _make_screenplay_from_scenes("tt_share", [units])
    flags = pd.Series({parsed.imdb_id: False})
    df = compute_character_network_features(
        {parsed.imdb_id: parsed}, flags,
    )
    row = df.loc[parsed.imdb_id]
    assert row["network_top1_dialogue_share"] == pytest.approx(0.5)
    assert row["network_top3_dialogue_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration test on real corpus
# ---------------------------------------------------------------------------


def test_smoke_full_corpus_first_ten_films():
    """Smoke test: compute features on first 10 corpus films, check shape and finiteness."""
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    if not pkl_path.is_file():
        pytest.skip(f"Corpus pickle missing at {pkl_path}; run Phase 2 first")
    with pkl_path.open("rb") as f:
        full_corpus = pickle.load(f)
    ids = list(full_corpus.keys())[:10]
    sub = {i: full_corpus[i] for i in ids}

    parquet_path = paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    df = pd.read_parquet(parquet_path).set_index("imdb_id")
    flags = df.loc[ids, "data_quality_flag"]

    out = compute_character_network_features(sub, flags)

    assert out.shape == (10, len(CHARACTER_NETWORK_FEATURE_COLUMNS))
    assert out.index.name == "imdb_id"
    assert list(out.index) == ids
    assert not np.isinf(out.values).any()
    # Unflagged films should have at least the cast-size column
    # populated.
    for imdb_id in ids:
        if not flags.loc[imdb_id]:
            assert not np.isnan(
                out.loc[imdb_id, "network_n_significant_characters"]
            )
