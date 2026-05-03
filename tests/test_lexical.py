"""Tests for ``src.features.lexical``.

Covers per-feature unit tests on synthetic fixtures, plus an integration
test that runs ``compute_lexical_features`` on the first ten films of
the actual corpus and checks shape and finiteness invariants.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.parse_screenplay import ParsedScreenplay, Scene
from src.features.lexical import (
    DIAGNOSTIC_ONLY_COLUMNS,
    LEXICAL_FEATURE_COLUMNS,
    LexicalFeatureConfig,
    MODEL_FEATURE_COLUMNS,
    compute_flesch_kincaid_grade,
    compute_hapax_ratio,
    compute_lexical_features,
    compute_mean_log_frequency,
    compute_mtld,
    compute_pronoun_ratio,
    compute_punctuation_rates,
    compute_rare_word_proportion,
    ensure_nltk_resources,
)
from src.utils import paths


# ---------------------------------------------------------------------------
# Module-level setup: ensure NLTK punkt is available before any tests run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _setup_nltk():
    ensure_nltk_resources()


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
# MTLD reference values
# ---------------------------------------------------------------------------


def test_mtld_returns_nan_for_short_input():
    """MTLD is undefined for sequences shorter than approximately 50 tokens."""
    short = ["the", "quick", "brown", "fox"]
    assert np.isnan(compute_mtld(short))


def test_mtld_returns_finite_for_realistic_input():
    """A long sequence with realistic redundancy produces a finite MTLD score."""
    # 30 distinct words repeated to make 240 tokens. Each word appears
    # ~8 times, giving the running TTR enough redundancy to complete
    # multiple factors but not so much that diversity collapses.
    vocab = [f"word{i}" for i in range(30)]
    tokens = (vocab * 8)
    score = compute_mtld(tokens, threshold=0.72)
    assert np.isfinite(score)
    # With this redundancy the score should be substantially above 0
    # but well below the token count.
    assert 5 < score < 200


def test_mtld_lower_for_repetitive_text():
    """A more repetitive sequence has lower MTLD than a more varied one."""
    # Both sequences must have enough redundancy for MTLD to be defined.
    # Repetitive: 4-word vocab, very low diversity.
    # Varied: 60-word vocab, moderate diversity.
    repetitive = ["the", "and", "a", "of"] * 50  # 200 tokens, 4 types
    vocab = [f"word{i}" for i in range(60)]
    varied = (vocab * 4)[:200]  # 200 tokens, 60 types
    assert compute_mtld(repetitive) < compute_mtld(varied)


# ---------------------------------------------------------------------------
# Hapax ratio
# ---------------------------------------------------------------------------


def test_hapax_ratio_all_unique():
    """Every token unique once: hapax ratio is 1.0."""
    tokens = ["a", "b", "c", "d"]
    assert compute_hapax_ratio(tokens) == pytest.approx(1.0)


def test_hapax_ratio_all_duplicated():
    """No token appears exactly once: hapax ratio is 0."""
    tokens = ["a", "a", "b", "b", "c", "c"]
    assert compute_hapax_ratio(tokens) == pytest.approx(0.0)


def test_hapax_ratio_empty_returns_nan():
    assert np.isnan(compute_hapax_ratio([]))


# ---------------------------------------------------------------------------
# Mean log frequency and rare-word proportion
# ---------------------------------------------------------------------------


def test_mean_log_frequency_common_vocab_scores_high():
    """Common words score with high Zipf values (around 5-7)."""
    tokens = ["the", "a", "and", "of", "to", "in"]
    mean_zipf, oov = compute_mean_log_frequency(tokens, oov_fallback=1.0)
    assert mean_zipf > 5.0
    assert oov == 0.0


def test_mean_log_frequency_rare_vocab_scores_low():
    """Rare/sophisticated words pull the mean down."""
    common = ["the", "a", "and", "of", "to"]
    rare = ["supercilious", "perspicacious", "gracile", "sesquipedalian", "perfunctory"]
    mean_common, _ = compute_mean_log_frequency(common, oov_fallback=1.0)
    mean_rare, _ = compute_mean_log_frequency(rare, oov_fallback=1.0)
    assert mean_rare < mean_common


def test_mean_log_frequency_oov_handling():
    """Out-of-vocabulary tokens hit the fallback path."""
    tokens = ["xyzabcnotaword", "qpwoeirutyalsobad"]
    _, oov_rate = compute_mean_log_frequency(tokens, oov_fallback=1.0)
    assert oov_rate == 1.0


def test_rare_word_proportion_thresholding():
    """Tokens at or below the cutoff count as rare."""
    # Set cutoff very high (Zipf 8) so virtually all tokens count as rare.
    tokens = ["the", "a", "and"]
    high = compute_rare_word_proportion(tokens, quartile_cutoff=8.0, oov_fallback=1.0)
    # Set cutoff very low so virtually nothing counts.
    low = compute_rare_word_proportion(tokens, quartile_cutoff=0.5, oov_fallback=1.0)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Flesch-Kincaid
# ---------------------------------------------------------------------------


def test_flesch_kincaid_simple_text():
    """Short simple sentences score at low grade levels."""
    text = "The cat sat on the mat. The dog ran. Cats and dogs play."
    grade = compute_flesch_kincaid_grade(text)
    assert np.isfinite(grade)
    # Grade should be modest (single-digit) for this kind of text.
    assert grade < 8


def test_flesch_kincaid_complex_text_higher():
    """Longer-sentence, polysyllabic text scores higher than simple text."""
    simple = "The cat sat on the mat. The dog ran fast."
    complex_text = (
        "The supercilious aristocrat, surveying the proceedings with magisterial "
        "indifference, contemplated the perspicacious observations of the assembly."
    )
    assert compute_flesch_kincaid_grade(complex_text) > compute_flesch_kincaid_grade(simple)


# ---------------------------------------------------------------------------
# Punctuation rates and pronoun ratio
# ---------------------------------------------------------------------------


def test_punctuation_rates_count_correctly():
    """Question and exclamation marks are counted per 1000 word tokens."""
    lines = ["Hello?", "How are you?", "Great!"]
    q_rate, e_rate, n_tokens = compute_punctuation_rates(lines)
    # 5 word tokens, 2 questions, 1 exclamation
    assert n_tokens == 5
    assert q_rate == pytest.approx(2 * 1000.0 / 5)
    assert e_rate == pytest.approx(1 * 1000.0 / 5)


def test_pronoun_ratio_first_dominant():
    """Lots of first-person pronouns produces a high ratio."""
    tokens = ["i", "i", "me", "my", "you"]
    # 4 first / (1 second + 1 epsilon) = 2.0
    assert compute_pronoun_ratio(tokens) == pytest.approx(4 / 2)


def test_pronoun_ratio_archaic_forms_count_as_second():
    """Archaic 'thou'/'thee' count toward the second-person denominator."""
    modern = ["you", "you"]
    archaic = ["thou", "thee"]
    # Same effective denominator; both should produce the same ratio.
    assert compute_pronoun_ratio(["i"] + modern) == compute_pronoun_ratio(["i"] + archaic)


# ---------------------------------------------------------------------------
# End-to-end smoke: feature columns and shape
# ---------------------------------------------------------------------------


def test_feature_columns_have_expected_count():
    """LEXICAL_FEATURE_COLUMNS has 14 model features + 1 diagnostic."""
    assert len(LEXICAL_FEATURE_COLUMNS) == 14
    assert len(MODEL_FEATURE_COLUMNS) == 13
    assert len(DIAGNOSTIC_ONLY_COLUMNS) == 1


def test_smoke_compute_lexical_features_on_first_ten_films():
    """Smoke test on real corpus data (first 10 films).

    Asserts: output shape is (10, 14), no all-NaN columns, all values
    finite or NaN (no inf), index matches IMDb IDs.
    """
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    if not pkl_path.is_file():
        pytest.skip(f"Corpus pickle missing at {pkl_path}; run Phase 2 first")
    with pkl_path.open("rb") as f:
        full_corpus = pickle.load(f)
    first_ten_ids = list(full_corpus.keys())[:10]
    sub = {i: full_corpus[i] for i in first_ten_ids}

    df = compute_lexical_features(sub)

    assert df.shape == (10, len(LEXICAL_FEATURE_COLUMNS))
    assert df.index.name == "imdb_id"
    assert list(df.index) == first_ten_ids
    # No column should be all-NaN; some films may have NaN for short
    # sequences (MTLD requires >=50 tokens) but not all 10.
    n_all_nan = df.isna().all(axis=0).sum()
    assert n_all_nan == 0
    # No infinities (well-defined non-degenerate computations).
    assert not np.isinf(df.values).any()
