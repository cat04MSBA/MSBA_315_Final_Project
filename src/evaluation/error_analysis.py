"""Phase 8 error-analysis cuts on the test set.

Implements the four pre-registered cuts from
``docs/proposals/phase8_preregistration.md`` Section 4.6:

* By primary genre (bucketed)
* By release decade
* By production budget tier
* By screenplay-length tier (scenes)

Plus the most-correct / most-wrong galleries on the headline target.
The cuts are pre-specified and are not expanded after seeing
results.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from src.decision.cost_matrix import (
    CostMatrix,
    DEFAULT_COST_MATRIX,
)
from src.decision.evaluation import evaluate
from src.decision.rule import decide_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def assign_decade_bucket(year: float | int) -> str:
    """Same buckets as the Phase 3 stratified split."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "unknown"
    if y < 1980:
        return "pre_1980"
    if y < 1990:
        return "1980s"
    if y < 2000:
        return "1990s"
    if y < 2010:
        return "2000s"
    if y < 2020:
        return "2010s"
    return "2020s"


def assign_budget_tier(budget: float) -> str:
    """Standard industry tiers."""
    if pd.isna(budget) or budget <= 0:
        return "unknown"
    b = float(budget)
    if b < 10_000_000:
        return "under_10M"
    if b < 50_000_000:
        return "10M_50M"
    if b < 150_000_000:
        return "50M_150M"
    return "over_150M"


def assign_length_tier(n_scenes: float | int) -> str:
    """Quartile-style tiers on screenplay scene count."""
    try:
        n = float(n_scenes)
    except (TypeError, ValueError):
        return "unknown"
    if n < 60:
        return "under_60"
    if n < 131:
        return "60_130"
    if n < 201:
        return "131_200"
    return "over_200"


# ---------------------------------------------------------------------------
# Per-cut metric computation
# ---------------------------------------------------------------------------


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    try:
        return float(log_loss(y_true, p, labels=[0, 1]))
    except ValueError:
        return float("nan")


def cut_metrics(
    df: pd.DataFrame,
    cut_col: str,
    target_col: str,
    prob_col: str,
    action_col: str,
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
) -> pd.DataFrame:
    """For each value of ``cut_col``, compute AUC / F1 / refer rate / total cost.

    Expects ``df`` to carry: target_col (binary 0/1), prob_col
    (calibrated probability), action_col (Greenlight/Pass/Refer
    string), and the cut_col itself.
    """
    rows = []
    for value, sub in df.groupby(cut_col, dropna=False):
        n = len(sub)
        if n == 0:
            continue
        y_true = sub[target_col].astype(int).values
        y_prob = sub[prob_col].astype(float).values
        y_pred_05 = (y_prob >= 0.5).astype(int)
        actions = sub[action_col].astype(str).tolist()
        # Per-cell realized cost under the default cost matrix
        eval_result = evaluate(
            f"system_cut_{value}",
            actions,
            y_true.tolist(),
            cost_matrix,
        )
        rows.append({
            "cut_column": cut_col,
            "cut_value": str(value),
            "n": int(n),
            "n_pos": int(y_true.sum()),
            "n_neg": int(n - y_true.sum()),
            "auc": _safe_auc(y_true, y_prob),
            "f1_at_0.5": _safe_f1(y_true, y_pred_05),
            "log_loss": _safe_log_loss(y_true, y_prob),
            "p_greenlight": float((np.array(actions) == "Greenlight").mean()),
            "p_pass": float((np.array(actions) == "Pass").mean()),
            "p_refer": float((np.array(actions) == "Refer").mean()),
            "total_cost": float(eval_result.total_cost),
            "cost_per_film_M": eval_result.total_cost / max(n, 1) / 1_000_000,
        })
    return pd.DataFrame(rows).sort_values(["cut_column", "cut_value"])


# ---------------------------------------------------------------------------
# Most-correct / most-wrong galleries
# ---------------------------------------------------------------------------


def gallery(
    df: pd.DataFrame,
    target_col: str,
    prob_col: str,
    *,
    direction: Literal["most_correct", "most_wrong"],
    top_n: int = 15,
) -> pd.DataFrame:
    """Per-film log-loss; sort by it and return the top N."""
    eps = 1e-7
    p = np.clip(df[prob_col].astype(float).values, eps, 1 - eps)
    y = df[target_col].astype(int).values
    per_film_ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    out = df.copy()
    out["per_film_log_loss"] = per_film_ll
    if direction == "most_correct":
        # Most-correct = lowest log loss among predicted positives
        # (the system was confident it was a hit and was right). The
        # pre-reg's intent is "best calibrated confidence rewarded by
        # truth"; restrict to predicted positives or to films with
        # high probability and matching label.
        out = out[out[target_col] == 1].sort_values("per_film_log_loss", ascending=True)
    else:
        out = out.sort_values("per_film_log_loss", ascending=False)
    return out.head(top_n).reset_index(drop=True)
