"""Total-cost evaluation on the calibration set.

Given a sequence of (action, true_label) pairs and a cost matrix,
compute total realized cost and per-action breakdown. Helpers for
the system's decisions and for naive baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from src.decision.cost_matrix import ACTIONS, Action, CostMatrix


@dataclass(frozen=True)
class EvaluationResult:
    """Total cost + breakdown for one (strategy, cost_matrix) cell."""
    strategy_name: str
    cost_matrix_name: str
    total_cost: float
    n_decisions: int
    action_counts: dict[Action, int]
    action_proportions: dict[Action, float]
    per_action_cost: dict[Action, float]
    per_outcome_cost: dict[str, float]  # 'flop' / 'hit'


def evaluate(
    strategy_name: str,
    actions: list[Action],
    true_labels: list[int],
    cost_matrix: CostMatrix,
) -> EvaluationResult:
    """Aggregate realized costs for one strategy under one cost matrix."""
    if len(actions) != len(true_labels):
        raise ValueError(
            f"Length mismatch: {len(actions)} actions vs "
            f"{len(true_labels)} labels",
        )
    n = len(actions)
    total_cost = 0.0
    per_action_cost = {a: 0.0 for a in ACTIONS}
    per_outcome_cost = {"flop": 0.0, "hit": 0.0}
    counts: Counter[Action] = Counter()

    for action, label in zip(actions, true_labels):
        c = cost_matrix.realized_cost(action, label)
        total_cost += c
        per_action_cost[action] += c
        per_outcome_cost["hit" if label == 1 else "flop"] += c
        counts[action] += 1

    action_counts: dict[Action, int] = {a: int(counts.get(a, 0)) for a in ACTIONS}
    action_proportions: dict[Action, float] = {
        a: action_counts[a] / n if n > 0 else 0.0 for a in ACTIONS
    }
    return EvaluationResult(
        strategy_name=strategy_name,
        cost_matrix_name=cost_matrix.name,
        total_cost=float(total_cost),
        n_decisions=int(n),
        action_counts=action_counts,
        action_proportions=action_proportions,
        per_action_cost={a: float(per_action_cost[a]) for a in ACTIONS},
        per_outcome_cost={k: float(v) for k, v in per_outcome_cost.items()},
    )
