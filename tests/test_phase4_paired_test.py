"""Tests for the Phase 4 paired Bayesian comparison wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.phase4.paired_test import (
    HIGHER_IS_BETTER,
    ROPE_BY_METRIC,
    WINNER_THRESHOLD,
    all_pairwise_comparisons,
    compare_pair,
)


def test_rope_registry_matches_metric_set():
    """All metrics named in HIGHER_IS_BETTER must have a ROPE."""
    assert set(ROPE_BY_METRIC) == set(HIGHER_IS_BETTER)


def test_clear_winner_when_a_dominates():
    """Model A consistently better by 0.05 AUC -> winner is 'a'."""
    rng = np.random.default_rng(0)
    base = 0.55 + 0.02 * rng.standard_normal(15)
    a = base + 0.05
    b = base
    cmp = compare_pair("model_a", "model_b", "auc_roc", a, b)
    assert cmp.winner == "a"
    assert cmp.p_a_better >= WINNER_THRESHOLD
    assert cmp.runs == 3
    assert cmp.n_folds == 15


def test_clear_winner_when_b_dominates():
    rng = np.random.default_rng(1)
    base = 0.6 + 0.01 * rng.standard_normal(15)
    a = base
    b = base + 0.04
    cmp = compare_pair("model_a", "model_b", "auc_roc", a, b)
    assert cmp.winner == "b"
    assert cmp.p_b_better >= WINNER_THRESHOLD


def test_rope_when_models_equivalent():
    """Identical-up-to-tiny-noise scores -> ROPE wins."""
    rng = np.random.default_rng(2)
    base = 0.6 + 0.02 * rng.standard_normal(15)
    a = base + 0.0005 * rng.standard_normal(15)
    b = base + 0.0005 * rng.standard_normal(15)
    cmp = compare_pair("model_a", "model_b", "auc_roc", a, b)
    assert cmp.winner == "rope"
    assert cmp.p_rope > 0.5


def test_minimized_metric_flipped_correctly():
    """For RMSE, lower is better. Model A with lower RMSE -> winner 'a'."""
    rng = np.random.default_rng(3)
    base = 1.3 + 0.05 * rng.standard_normal(15)
    a = base - 0.1  # lower RMSE = better
    b = base
    cmp = compare_pair("model_a", "model_b", "rmse", a, b)
    assert cmp.winner == "a"
    assert cmp.p_a_better >= WINNER_THRESHOLD


def test_nan_folds_dropped_jointly():
    """A NaN in either series drops that fold from both."""
    a = np.array([0.6, 0.62, np.nan, 0.61, 0.63] * 3)
    b = np.array([0.55, 0.57, 0.56, np.nan, 0.58] * 3)
    cmp = compare_pair("a", "b", "auc_roc", a, b)
    # 9 folds survive (3 NaN per repeat -> 6 dropped from 15)
    assert cmp.n_folds == 9
    assert cmp.winner in ("a", "b", "rope")


def test_too_few_folds_returns_rope_with_nan():
    """Fewer than 4 surviving folds -> rope with NaN posteriors."""
    a = np.array([0.6, 0.62, 0.61, np.nan, np.nan] * 3)[:5]
    b = np.array([0.55, np.nan, 0.56, np.nan, np.nan] * 3)[:5]
    cmp = compare_pair("a", "b", "auc_roc", a, b)
    assert cmp.winner == "rope"
    assert np.isnan(cmp.p_a_better)


def test_unknown_metric_raises():
    a = np.linspace(0.5, 0.6, 15)
    b = np.linspace(0.5, 0.6, 15)
    with pytest.raises(KeyError):
        compare_pair("a", "b", "kappa", a, b)


def test_shape_mismatch_raises():
    a = np.linspace(0.5, 0.6, 15)
    b = np.linspace(0.5, 0.6, 10)
    with pytest.raises(ValueError):
        compare_pair("a", "b", "auc_roc", a, b)


def test_all_pairwise_returns_n_choose_2_per_metric():
    """4 families x 1 metric = 6 pairwise comparisons."""
    rng = np.random.default_rng(42)
    families = {
        f"f{i}": {"auc_roc": 0.55 + 0.05 * i + 0.02 * rng.standard_normal(15)}
        for i in range(4)
    }
    cmps = all_pairwise_comparisons(families, ["auc_roc"])
    assert len(cmps) == 6  # C(4, 2)
    pairs = {(c.family_a, c.family_b) for c in cmps}
    assert ("f0", "f1") in pairs
    assert ("f0", "f3") in pairs
    assert ("f2", "f3") in pairs
