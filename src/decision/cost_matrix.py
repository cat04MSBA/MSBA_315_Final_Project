"""Cost matrix dataclass + the pre-registered sweep variants.

The cost matrix encodes the project's asymmetric error costs from
``PROJECT_CONTEXT.md`` Section 1:

* Greenlight a flop (false positive): -$50M (lost production budget).
* Pass on a hit (false negative): -$100M (foregone revenue).
* Refer (any outcome): -$5K (human reader cost).

The numbers are negative-cost (positive = worse). The decision rule
in :mod:`rule` minimizes expected cost.

Sweep variants per ``phase6_preregistration.md`` Section 3.2:
asymmetry, refer cost, base magnitudes, per-genre cost matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal


Action = Literal["Greenlight", "Pass", "Refer"]
ACTIONS: tuple[Action, Action, Action] = ("Greenlight", "Pass", "Refer")


@dataclass(frozen=True)
class CostMatrix:
    """Per-action cost as a function of true binary outcome.

    Attributes are total cost in USD per (action, true_outcome) cell.
    All values are non-negative (cost; higher = worse). The decision
    rule picks the action with lowest expected cost.

    The pre-registered default values come from
    ``PROJECT_CONTEXT.md`` Section 1.
    """

    name: str
    cost_greenlight_flop: float = 50_000_000  # greenlight + flop -> lose budget
    cost_greenlight_hit: float = 0  # greenlight + hit -> correct, no error cost
    cost_pass_flop: float = 0  # pass + flop -> correct, no error cost
    cost_pass_hit: float = 100_000_000  # pass + hit -> opportunity cost
    cost_refer_flop: float = 5_000  # refer cost; outcome-independent
    cost_refer_hit: float = 5_000

    def expected_cost(self, p_hit: float) -> dict[Action, float]:
        """Return expected cost per action given P(hit | features)."""
        p = float(p_hit)
        return {
            "Greenlight": (1 - p) * self.cost_greenlight_flop + p * self.cost_greenlight_hit,
            "Pass": (1 - p) * self.cost_pass_flop + p * self.cost_pass_hit,
            "Refer": (1 - p) * self.cost_refer_flop + p * self.cost_refer_hit,
        }

    def realized_cost(self, action: Action, true_label: int) -> float:
        """Return realized cost for one decision given the true outcome."""
        if action == "Greenlight":
            return float(
                self.cost_greenlight_flop if true_label == 0
                else self.cost_greenlight_hit
            )
        if action == "Pass":
            return float(
                self.cost_pass_flop if true_label == 0
                else self.cost_pass_hit
            )
        if action == "Refer":
            return float(
                self.cost_refer_flop if true_label == 0
                else self.cost_refer_hit
            )
        raise ValueError(f"Unknown action: {action!r}")


# ---------------------------------------------------------------------------
# Pre-registered defaults + sweep variants
# ---------------------------------------------------------------------------


DEFAULT_COST_MATRIX = CostMatrix(name="default")


def asymmetry_variants() -> list[CostMatrix]:
    """Variants holding miss=$100M and refer=$5K, varying flop cost."""
    return [
        CostMatrix(name="asymmetry_1to1_flop100M",
                   cost_greenlight_flop=100_000_000),
        DEFAULT_COST_MATRIX,  # 1:2
        CostMatrix(name="asymmetry_1to4_flop25M",
                   cost_greenlight_flop=25_000_000),
        CostMatrix(name="asymmetry_2to1_flop200M",
                   cost_greenlight_flop=200_000_000),
    ]


def refer_cost_variants() -> list[CostMatrix]:
    """Variants varying only the refer cost; default flop:$50M, miss:$100M."""
    return [
        CostMatrix(name="refer_0K", cost_refer_flop=0, cost_refer_hit=0),
        DEFAULT_COST_MATRIX,  # $5K
        CostMatrix(name="refer_25K", cost_refer_flop=25_000, cost_refer_hit=25_000),
        CostMatrix(name="refer_100K", cost_refer_flop=100_000, cost_refer_hit=100_000),
        CostMatrix(name="refer_1M", cost_refer_flop=1_000_000, cost_refer_hit=1_000_000),
        # Refer cost so high it's never preferred (forces commit).
        CostMatrix(name="refer_25M", cost_refer_flop=25_000_000, cost_refer_hit=25_000_000),
    ]


def base_magnitude_variants() -> list[CostMatrix]:
    """Scale all costs uniformly. Should produce identical actions to default
    (decision rule is scale-invariant); sanity check."""
    scales = [(0.1, "scale_0.1x"), (1.0, "scale_1x_default"), (10.0, "scale_10x")]
    return [
        CostMatrix(
            name=name,
            cost_greenlight_flop=50_000_000 * s,
            cost_pass_hit=100_000_000 * s,
            cost_refer_flop=5_000 * s,
            cost_refer_hit=5_000 * s,
        )
        for s, name in scales
    ]


def all_sensitivity_variants() -> list[CostMatrix]:
    """Concatenated unique pre-registered variants across the three sweeps."""
    seen: dict[str, CostMatrix] = {}
    for variants in (asymmetry_variants(), refer_cost_variants(), base_magnitude_variants()):
        for cm in variants:
            seen.setdefault(cm.name, cm)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Per-genre matrices (built lazily from training data)
# ---------------------------------------------------------------------------


def per_genre_matrices(
    median_budget_per_genre: dict[str, float] | None = None,
    median_revenue_per_genre: dict[str, float] | None = None,
    base_cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
    min_budget_floor: float = 5_000_000,
    min_revenue_floor: float = 5_000_000,
) -> dict[str, CostMatrix]:
    """Per-genre cost matrices derived from per-genre median financials.

    For each genre with at least 30 films in the corpus, set:
    * cost_greenlight_flop = median_budget_per_genre[genre]
    * cost_pass_hit = median_revenue_per_genre[genre] -
      median_budget_per_genre[genre]   (the "hit profit" foregone)
    * cost_refer = same as base

    If financials are unavailable, fall back to the base matrix.
    """
    if not median_budget_per_genre or not median_revenue_per_genre:
        return {"default": base_cost_matrix}

    out: dict[str, CostMatrix] = {}
    for genre, budget in median_budget_per_genre.items():
        revenue = median_revenue_per_genre.get(genre, 0)
        budget_clipped = max(float(budget), min_budget_floor)
        miss_cost = max(float(revenue) - float(budget), min_revenue_floor)
        out[genre] = CostMatrix(
            name=f"per_genre_{genre}",
            cost_greenlight_flop=budget_clipped,
            cost_pass_hit=miss_cost,
            cost_refer_flop=base_cost_matrix.cost_refer_flop,
            cost_refer_hit=base_cost_matrix.cost_refer_hit,
        )
    return out
