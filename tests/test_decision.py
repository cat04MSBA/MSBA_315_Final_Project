"""Tests for the Phase 6 cost-decision modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.cost_matrix import (
    DEFAULT_COST_MATRIX,
    CostMatrix,
    asymmetry_variants,
    refer_cost_variants,
)
from src.decision.evaluation import evaluate
from src.decision.rule import decide_batch, decide_one


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------


def test_default_cost_matrix_values_match_brief():
    """The default values come from PROJECT_CONTEXT.md Section 1."""
    cm = DEFAULT_COST_MATRIX
    assert cm.cost_greenlight_flop == 50_000_000
    assert cm.cost_greenlight_hit == 0
    assert cm.cost_pass_flop == 0
    assert cm.cost_pass_hit == 100_000_000
    assert cm.cost_refer_flop == 5_000
    assert cm.cost_refer_hit == 5_000


def test_expected_cost_at_p_zero():
    """At p=0 (definitely flop): Pass costs 0, Greenlight costs the full flop cost."""
    cm = DEFAULT_COST_MATRIX
    costs = cm.expected_cost(0.0)
    assert costs["Pass"] == 0
    assert costs["Greenlight"] == 50_000_000
    assert costs["Refer"] == 5_000


def test_expected_cost_at_p_one():
    """At p=1 (definitely hit): Greenlight costs 0, Pass costs the full miss cost."""
    cm = DEFAULT_COST_MATRIX
    costs = cm.expected_cost(1.0)
    assert costs["Greenlight"] == 0
    assert costs["Pass"] == 100_000_000
    assert costs["Refer"] == 5_000


def test_realized_cost_outcomes():
    cm = DEFAULT_COST_MATRIX
    assert cm.realized_cost("Greenlight", 0) == 50_000_000
    assert cm.realized_cost("Greenlight", 1) == 0
    assert cm.realized_cost("Pass", 0) == 0
    assert cm.realized_cost("Pass", 1) == 100_000_000
    assert cm.realized_cost("Refer", 0) == 5_000
    assert cm.realized_cost("Refer", 1) == 5_000


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def test_default_rule_at_p_zero_picks_pass_or_refer():
    """At p=0, pass costs 0 and refer costs 5K; pass strictly beats refer; pass should win."""
    d = decide_one("tt0000001", 0.0, DEFAULT_COST_MATRIX)
    assert d.recommended_action == "Pass"


def test_default_rule_at_p_one_picks_greenlight_or_refer():
    """At p=1, greenlight costs 0 and refer costs 5K; greenlight should win."""
    d = decide_one("tt0000002", 1.0, DEFAULT_COST_MATRIX)
    assert d.recommended_action == "Greenlight"


def test_default_rule_at_intermediate_p_picks_refer():
    """At p=0.5, both Greenlight ($25M) and Pass ($50M) hugely exceed Refer ($5K)."""
    d = decide_one("tt0000003", 0.5, DEFAULT_COST_MATRIX)
    assert d.recommended_action == "Refer"


def test_high_refer_cost_forces_commit():
    """If refer cost is $25M and p=0.7, Greenlight wins ($15M < $25M < $70M)."""
    cm = CostMatrix(name="huge_refer",
                    cost_refer_flop=25_000_000,
                    cost_refer_hit=25_000_000)
    d = decide_one("tt0000004", 0.7, cm)
    assert d.recommended_action == "Greenlight"


def test_decide_batch_length_validation():
    with pytest.raises(ValueError):
        decide_batch(["a", "b"], [0.5], DEFAULT_COST_MATRIX)


def test_rationale_contains_recommended_action():
    d = decide_one("tt0000005", 0.5, DEFAULT_COST_MATRIX)
    assert d.recommended_action.lower() in d.rationale.lower()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_total_cost_matches_realized():
    """3 films, 3 actions, 3 outcomes — total cost should be the sum."""
    cm = DEFAULT_COST_MATRIX
    actions = ["Greenlight", "Pass", "Refer"]
    labels = [0, 1, 0]  # flop greenlit + hit passed + flop referred
    expected = 50_000_000 + 100_000_000 + 5_000
    r = evaluate("test", actions, labels, cm)
    assert r.total_cost == expected
    assert r.action_counts == {"Greenlight": 1, "Pass": 1, "Refer": 1}
    assert sum(r.action_proportions.values()) == 1.0


def test_evaluate_length_validation():
    with pytest.raises(ValueError):
        evaluate("test", ["Greenlight"], [0, 1], DEFAULT_COST_MATRIX)


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------


def test_asymmetry_variants_have_default_in_set():
    names = {v.name for v in asymmetry_variants()}
    assert DEFAULT_COST_MATRIX.name in names


def test_refer_cost_variants_span_orders_of_magnitude():
    costs = sorted(v.cost_refer_flop for v in refer_cost_variants())
    assert costs[0] == 0
    assert costs[-1] >= 1_000_000  # at least one variant in the millions
