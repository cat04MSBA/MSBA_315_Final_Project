"""Tests for ``src.features.sentiment``.

Covers per-feature unit tests on synthetic fixtures, plus an integration
test that runs ``compute_sentiment_features`` on the first ten films of
the actual corpus and checks shape and finiteness invariants.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.parse_screenplay import ParsedScreenplay, Scene
from src.features.sentiment import (
    DIAGNOSTIC_ONLY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NRC_EMOTIONS,
    REAGAN_ARCHETYPES,
    SENTIMENT_FEATURE_COLUMNS,
    SentimentFeatureConfig,
    build_reagan_templates,
    compute_arc_similarities,
    compute_nrc_proportions,
    compute_quartile_features,
    compute_sentiment_features,
    compute_vader_aggregates,
    ensure_nltk_resources,
)
from src.utils import paths


# ---------------------------------------------------------------------------
# Module-level setup: download NLTK resources before any tests run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _setup_nltk():
    ensure_nltk_resources()


@pytest.fixture(scope="module")
def vader_analyzer():
    """Cached VADER analyzer; the lexicon load is non-trivial."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@pytest.fixture(scope="module")
def english_stopwords():
    """Cached NLTK English stopword set."""
    from nltk.corpus import stopwords as _stopwords
    return frozenset(_stopwords.words("english"))


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_screenplay(scenes: list[tuple[str, str, list[tuple[str, str]]]]) -> ParsedScreenplay:
    """Build a ParsedScreenplay from (stage_dir, scene_desc, [(speaker, text)]) tuples."""
    scene_objs = tuple(
        Scene(
            scene_number=i,
            stage_direction=stage,
            scene_description=desc,
            dialogue_units=tuple(units),
        )
        for i, (stage, desc, units) in enumerate(scenes, start=1)
    )
    return ParsedScreenplay(
        imdb_id="tt_test",
        scenes=scene_objs,
        n_scenes=len(scene_objs),
        n_unique_characters=len({s for sc in scene_objs for s, _ in sc.dialogue_units}),
        n_dialogue_lines=sum(len(sc.dialogue_units) for sc in scene_objs),
        total_dialogue_chars=sum(len(t) for sc in scene_objs for _, t in sc.dialogue_units),
        total_stage_direction_chars=sum(len(sc.stage_direction) for sc in scene_objs),
        total_scene_description_chars=sum(len(sc.scene_description) for sc in scene_objs),
        total_action_chars=sum(
            len(sc.stage_direction) + len(sc.scene_description) for sc in scene_objs
        ),
        dialogue_to_action_ratio=0.5,
        dialogue_to_total_text_ratio=0.5,
        parse_warnings=tuple(),
    )


# ---------------------------------------------------------------------------
# VADER aggregates
# ---------------------------------------------------------------------------


def test_vader_aggregates_empty_input(vader_analyzer):
    """Empty dialogue produces NaN across all four return values."""
    mean, std, rng, zero_rate = compute_vader_aggregates([], vader_analyzer)
    assert all(np.isnan(v) for v in (mean, std, rng, zero_rate))


def test_vader_aggregates_positive_sentiment_higher_than_negative(vader_analyzer):
    """A clearly positive set of lines scores higher than a clearly negative one."""
    positive = ["I love this so much!", "This is amazing.", "What a great day."]
    negative = ["I hate this.", "This is awful.", "Terrible, just terrible."]
    pos_mean, _, _, _ = compute_vader_aggregates(positive, vader_analyzer)
    neg_mean, _, _, _ = compute_vader_aggregates(negative, vader_analyzer)
    assert pos_mean > neg_mean


def test_vader_aggregates_zero_compound_rate_counts_correctly(vader_analyzer):
    """Lines that produce VADER compound=0 are counted toward zero_compound_rate."""
    # "The cat sat on the mat." is sentiment-neutral; VADER returns 0.
    lines = ["The cat sat on the mat.", "I love this!", "The wall is white."]
    _, _, _, zero_rate = compute_vader_aggregates(lines, vader_analyzer)
    # At least one line is unambiguously neutral; the rate should be > 0.
    assert zero_rate > 0.0


# ---------------------------------------------------------------------------
# NRC emotion proportions
# ---------------------------------------------------------------------------


def test_nrc_proportions_emit_correct_keys(english_stopwords):
    """The returned dict has exactly the eight emotion keys."""
    proportions, _ = compute_nrc_proportions(
        ["happy", "joy", "fear"],
        remove_stopwords=False,
        stopwords=english_stopwords,
    )
    assert set(proportions.keys()) == set(NRC_EMOTIONS)


def test_nrc_proportions_emotional_words_score_higher(english_stopwords):
    """Emotion-laden tokens produce non-zero per-emotion proportions."""
    tokens = ["fear", "fearful", "darkness", "terror", "horror"]
    proportions, _ = compute_nrc_proportions(
        tokens, remove_stopwords=False, stopwords=english_stopwords,
    )
    # "fear", "fearful", "darkness", "terror", "horror" all carry the
    # ``fear`` emotion in NRC EmoLex.
    assert proportions["fear"] > 0.0


def test_nrc_proportions_neutral_tokens_score_zero(english_stopwords):
    """Tokens with no emotion tags produce zero across every emotion."""
    # Made-up nonsense tokens should not match any NRC entry.
    tokens = ["xyznotaword", "qpwoeirutyalsobad"]
    proportions, oov = compute_nrc_proportions(
        tokens, remove_stopwords=False, stopwords=english_stopwords,
    )
    assert all(v == 0.0 for v in proportions.values())
    assert oov == 1.0


def test_nrc_stopword_removal_changes_proportions(english_stopwords):
    """Removing stopwords reshapes the proportions (denominator shrinks)."""
    # Mix emotion words with English stopwords. Without removal the
    # denominator is larger so the per-emotion proportions are smaller.
    tokens = ["the", "happy", "and", "joy", "is"]
    p_with, _ = compute_nrc_proportions(
        tokens, remove_stopwords=True, stopwords=english_stopwords,
    )
    p_without, _ = compute_nrc_proportions(
        tokens, remove_stopwords=False, stopwords=english_stopwords,
    )
    # Denominator shrinks from 5 to 2 with stopwords removed; the
    # per-emotion proportion for ``joy`` therefore rises.
    assert p_with["joy"] > p_without["joy"]


# ---------------------------------------------------------------------------
# Quartile-windowed features
# ---------------------------------------------------------------------------


def test_quartile_features_short_input_returns_nan():
    """Sequences shorter than n_windows return all-NaN."""
    out = compute_quartile_features([0.1, 0.2], n_windows=4)
    assert all(np.isnan(v) for v in out)


def test_quartile_features_uniform_input_zero_volatility():
    """A constant trajectory has zero per-quartile std and zero concentration."""
    seq = [0.5] * 100
    q1, q2, q3, q4, vol = compute_quartile_features(seq, n_windows=4)
    assert q1 == pytest.approx(0.5)
    assert q2 == pytest.approx(0.5)
    assert q3 == pytest.approx(0.5)
    assert q4 == pytest.approx(0.5)
    assert vol == pytest.approx(0.0)


def test_quartile_features_concentrated_volatility():
    """A trajectory with one volatile quartile has high volatility concentration."""
    # Q1, Q2, Q3 calm; Q4 erratic.
    quiet = [0.1] * 75
    erratic = [-0.9, 0.9] * 12 + [0.0]  # 25 alternating values
    _, _, _, _, vol = compute_quartile_features(quiet + erratic, n_windows=4)
    assert vol > 0.5


def test_quartile_features_rejects_non_four_windows():
    """The schema is hard-coded to four windows."""
    with pytest.raises(ValueError):
        compute_quartile_features([0.0] * 50, n_windows=5)


# ---------------------------------------------------------------------------
# Reagan archetype templates and arc similarities
# ---------------------------------------------------------------------------


def test_reagan_templates_include_six_keys():
    """All six archetype names appear in the template mapping."""
    templates = build_reagan_templates(length=100)
    assert set(templates.keys()) == set(REAGAN_ARCHETYPES)


def test_reagan_templates_are_zscore_normalized():
    """Each template has approximately zero mean and unit standard deviation."""
    templates = build_reagan_templates(length=100)
    for name, vec in templates.items():
        assert vec.mean() == pytest.approx(0.0, abs=1e-10)
        # Std may not be exactly 1 due to discrete sampling but should be
        # within a small tolerance.
        assert vec.std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_reagan_templates_pairs_are_reflections():
    """Tragedy/Rags-to-Riches, Icarus/Man-in-a-Hole, Oedipus/Cinderella are reflections."""
    templates = build_reagan_templates(length=100)
    np.testing.assert_allclose(
        templates["tragedy"], -templates["rags_to_riches"], atol=1e-10,
    )
    np.testing.assert_allclose(
        templates["icarus"], -templates["man_in_a_hole"], atol=1e-10,
    )
    np.testing.assert_allclose(
        templates["oedipus"], -templates["cinderella"], atol=1e-10,
    )


def test_arc_similarity_man_in_a_hole_matches_synthetic_fall_rise():
    """A synthetic Man-in-a-Hole trajectory scores highest on the matching template."""
    templates = build_reagan_templates(length=100)
    # Build a fall-rise trajectory at length 200 (the per-line input
    # length the function would see for a real film) so the
    # interpolation step has work to do.
    t = np.linspace(0.0, 1.0, 200)
    # cos(2*pi*t) over [0, 1] gives +1 at endpoints and -1 at midpoint —
    # exactly the Man-in-a-Hole shape.
    trajectory = np.cos(2.0 * np.pi * t)
    sims = compute_arc_similarities(trajectory.tolist(), templates, target_length=100)
    # Highest similarity should be Man-in-a-Hole.
    best = max(sims, key=lambda k: sims[k])
    assert best == "man_in_a_hole"
    # Symmetric: Icarus is the inverse, so Icarus score should be near
    # -Man-in-a-Hole score.
    assert sims["icarus"] == pytest.approx(-sims["man_in_a_hole"], abs=1e-2)


def test_arc_similarity_constant_trajectory_returns_nan():
    """A flat trajectory cannot be z-normalized and produces NaN similarities."""
    flat = [0.5] * 200
    templates = build_reagan_templates(length=100)
    sims = compute_arc_similarities(flat, templates, target_length=100)
    assert all(np.isnan(v) for v in sims.values())


# ---------------------------------------------------------------------------
# End-to-end: feature columns and shape
# ---------------------------------------------------------------------------


def test_feature_columns_have_expected_count():
    """SENTIMENT_FEATURE_COLUMNS has 22 model features + 2 diagnostics."""
    assert len(SENTIMENT_FEATURE_COLUMNS) == 24
    assert len(MODEL_FEATURE_COLUMNS) == 22
    assert len(DIAGNOSTIC_ONLY_COLUMNS) == 2


def test_smoke_compute_sentiment_features_on_first_ten_films():
    """Smoke test on real corpus data (first 10 films).

    Asserts: output shape is (10, 24), no all-NaN columns, all values
    finite or NaN (no inf), index matches IMDb IDs.
    """
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    if not pkl_path.is_file():
        pytest.skip(f"Corpus pickle missing at {pkl_path}; run Phase 2 first")
    with pkl_path.open("rb") as f:
        full_corpus = pickle.load(f)
    first_ten_ids = list(full_corpus.keys())[:10]
    sub = {i: full_corpus[i] for i in first_ten_ids}

    df = compute_sentiment_features(sub)

    assert df.shape == (10, len(SENTIMENT_FEATURE_COLUMNS))
    assert df.index.name == "imdb_id"
    assert list(df.index) == first_ten_ids
    n_all_nan = df.isna().all(axis=0).sum()
    assert n_all_nan == 0
    assert not np.isinf(df.values).any()
