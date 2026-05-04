"""Per-film expected-cost decision rule.

Given a calibrated probability ``p = P(hit | features)`` and a cost
matrix, the rule picks the action minimizing expected cost. Tie-
breaks to Refer (conservative).

The function returns a dict with the per-film output schema from
``phase6_preregistration.md`` Section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.decision.cost_matrix import ACTIONS, Action, CostMatrix


@dataclass(frozen=True)
class DecisionResult:
    """Single-film decision output."""
    imdb_id: str
    calibrated_probability: float
    expected_cost_greenlight: float
    expected_cost_pass: float
    expected_cost_refer: float
    recommended_action: Action
    rationale: str
    cost_matrix_name: str


def _format_dollars(amount: float) -> str:
    """Human-readable dollar formatting; e.g. $50.0M, $5.0K."""
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if abs(amount) >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


def _build_rationale(
    p: float, costs: dict[Action, float], chosen: Action,
    cost_matrix: CostMatrix,
) -> str:
    g = costs["Greenlight"]
    pa = costs["Pass"]
    re = costs["Refer"]
    if chosen == "Greenlight":
        return (
            f"Recommended Greenlight: model probability {p:.3f} of "
            f">=2x ROI yields expected loss {_format_dollars(g)} "
            f"from Greenlight vs {_format_dollars(pa)} from Pass and "
            f"{_format_dollars(re)} from Refer."
        )
    if chosen == "Pass":
        return (
            f"Recommended Pass: model probability {p:.3f} of >=2x ROI "
            f"yields expected loss {_format_dollars(pa)} from Pass vs "
            f"{_format_dollars(g)} from Greenlight and "
            f"{_format_dollars(re)} from Refer."
        )
    return (
        f"Recommended Refer to human reader: at probability {p:.3f}, "
        f"expected losses from Greenlight ({_format_dollars(g)}) and "
        f"Pass ({_format_dollars(pa)}) both exceed the human-reader "
        f"cost ({_format_dollars(re)}). Manual review is preferred."
    )


def decide_one(
    imdb_id: str,
    calibrated_probability: float,
    cost_matrix: CostMatrix,
) -> DecisionResult:
    """Apply the expected-cost decision rule to one film."""
    costs = cost_matrix.expected_cost(calibrated_probability)
    # Tie-break to Refer (conservative): if any other action ties, prefer Refer.
    min_cost = min(costs.values())
    tied = [a for a, c in costs.items() if c == min_cost]
    if "Refer" in tied:
        action = "Refer"
    elif len(tied) == 1:
        action = tied[0]
    else:
        # Greenlight vs Pass tie (no Refer): pick the higher-expected-value
        # one; with cost ratios as in the default this rarely happens.
        action = tied[0]

    rationale = _build_rationale(calibrated_probability, costs, action, cost_matrix)
    return DecisionResult(
        imdb_id=imdb_id,
        calibrated_probability=float(calibrated_probability),
        expected_cost_greenlight=float(costs["Greenlight"]),
        expected_cost_pass=float(costs["Pass"]),
        expected_cost_refer=float(costs["Refer"]),
        recommended_action=action,
        rationale=rationale,
        cost_matrix_name=cost_matrix.name,
    )


def decide_batch(
    imdb_ids: list[str],
    calibrated_probabilities: list[float],
    cost_matrix: CostMatrix,
) -> list[DecisionResult]:
    """Apply the rule to a batch of films."""
    if len(imdb_ids) != len(calibrated_probabilities):
        raise ValueError(
            f"Length mismatch: {len(imdb_ids)} ids vs "
            f"{len(calibrated_probabilities)} probabilities",
        )
    return [
        decide_one(iid, p, cost_matrix)
        for iid, p in zip(imdb_ids, calibrated_probabilities)
    ]
