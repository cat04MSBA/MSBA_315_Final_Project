"""Layer-by-layer test-set metrics with bootstrap confidence intervals.

Implements Phase 8 pre-registration Section 4 (predictive
performance, calibration, decision quality, attribution stability).
Each metric is computed once on the test set; bootstrap CIs are
formed by resampling the 257 films with replacement (seed 42, 2,000
draws, percentile method).

The module is read-only against the four-layer artifacts: it loads
the per-film outputs produced by ``run_batch`` plus the test-set
labels and feature matrix, then computes everything in plain numpy
/ scikit-learn / scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.calibration.metrics import (
    expected_calibration_error,
    reliability_curve,
)
from src.decision.cost_matrix import (
    CostMatrix,
    DEFAULT_COST_MATRIX,
    refer_cost_variants,
)
from src.decision.evaluation import evaluate
from src.decision.baselines import BASELINES
from src.decision.rule import decide_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


BOOTSTRAP_N: int = 2_000
BOOTSTRAP_SEED: int = 42
DEFAULT_CONFIDENCE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    metric_fn,
    *args,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap (point, lower, upper) for a metric of (y_true, y_score).

    ``metric_fn`` takes positional arrays (y_true, y_score, ...) and
    returns a scalar. ``args`` is a tuple of arrays of equal length.
    Resampling is over the first axis (films).
    """
    arrays = [np.asarray(a) for a in args]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples[b] = float(metric_fn(*[a[idx] for a in arrays]))
        except (ValueError, ZeroDivisionError):
            samples[b] = float("nan")
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Layer 1 — predictive performance
# ---------------------------------------------------------------------------


def regression_metrics_with_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """MSE / RMSE / MAE / CVRMSE on the regression target with bootstrap CIs."""

    def mse(yt, yp):
        return mean_squared_error(yt, yp)

    def rmse(yt, yp):
        return float(np.sqrt(mean_squared_error(yt, yp)))

    def mae(yt, yp):
        return float(mean_absolute_error(yt, yp))

    def cvrmse(yt, yp):
        denom = float(np.abs(np.mean(yt)))
        if denom < 1e-12:
            return float("nan")
        return float(np.sqrt(mean_squared_error(yt, yp))) / denom

    out: dict[str, dict[str, float]] = {}
    for name, fn in [("mse", mse), ("rmse", rmse), ("mae", mae), ("cvrmse", cvrmse)]:
        p, lo, hi = _bootstrap_ci(fn, y_true, y_pred)
        out[name] = {"point": p, "lower": lo, "upper": hi}
    return out


def classification_metrics_with_ci(
    y_true: np.ndarray, y_score: np.ndarray,
) -> dict[str, dict[str, float]]:
    """AUC-ROC / PR-AUC / F1@0.5 / log-loss on a binary target with bootstrap CIs."""

    def auc(yt, ys):
        if len(np.unique(yt)) < 2:
            return float("nan")
        return float(roc_auc_score(yt, ys))

    def pr_auc(yt, ys):
        if len(np.unique(yt)) < 2:
            return float("nan")
        return float(average_precision_score(yt, ys))

    def f1_at_half(yt, ys):
        return float(f1_score(yt, (ys >= 0.5).astype(int), zero_division=0))

    def ll(yt, ys):
        eps = 1e-7
        ys_clip = np.clip(ys, eps, 1 - eps)
        try:
            return float(log_loss(yt, ys_clip, labels=[0, 1]))
        except ValueError:
            return float("nan")

    out: dict[str, dict[str, float]] = {}
    for name, fn in [("auc_roc", auc), ("pr_auc", pr_auc), ("f1", f1_at_half), ("log_loss", ll)]:
        p, lo, hi = _bootstrap_ci(fn, y_true, y_score)
        out[name] = {"point": p, "lower": lo, "upper": hi}
    return out


# ---------------------------------------------------------------------------
# Layer 2 — calibration & coverage
# ---------------------------------------------------------------------------


def classification_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> dict[str, float]:
    """ECE, Brier, log-loss on calibrated probability."""
    eps = 1e-7
    p_clip = np.clip(y_prob, eps, 1 - eps)
    try:
        ll = float(log_loss(y_true, p_clip, labels=[0, 1]))
    except ValueError:
        ll = float("nan")
    return {
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": ll,
        "n": int(len(y_true)),
    }


def conformal_classification_coverage(
    y_true: np.ndarray, set_size: np.ndarray, in_set: np.ndarray,
    level: float,
) -> dict[str, float]:
    """Empirical coverage + singleton/refer/empty rates at one level.

    ``set_size[i]`` is in {0, 1, 2}; ``in_set[i]`` is whether the
    true label is in the prediction set at this level.
    """
    return {
        "level": level,
        "empirical_coverage": float(in_set.mean()) if len(in_set) else float("nan"),
        "singleton_rate": float((set_size == 1).mean()) if len(set_size) else float("nan"),
        "refer_rate": float((set_size == 2).mean()) if len(set_size) else float("nan"),
        "empty_rate": float((set_size == 0).mean()) if len(set_size) else float("nan"),
        "n_eval": int(len(y_true)),
    }


def conformal_regression_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    level: float,
) -> dict[str, float]:
    """Empirical coverage + mean / median width at one level."""
    in_int = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    return {
        "level": level,
        "empirical_coverage": float(in_int.mean()),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "n_eval": int(len(y_true)),
    }


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin reliability data for figure rendering."""
    mean_pred, accuracy, counts = reliability_curve(
        y_true, y_prob, n_bins=n_bins, strategy="equal_size",
    )
    return pd.DataFrame({
        "bin": np.arange(n_bins),
        "mean_predicted": mean_pred,
        "empirical_accuracy": accuracy,
        "count": counts,
    })


# ---------------------------------------------------------------------------
# Layer 3 — decision quality
# ---------------------------------------------------------------------------


def decision_evaluation(
    imdb_ids: list[str],
    calibrated_probs: np.ndarray,
    true_labels: np.ndarray,
    genres: list[str],
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """System + 5 baselines under one cost matrix.

    Returns (system_dict, baselines_list).
    """
    decisions = decide_batch(imdb_ids, calibrated_probs.tolist(), cost_matrix)
    actions = [d.recommended_action for d in decisions]
    sys_eval = evaluate("system", actions, true_labels.tolist(), cost_matrix)
    sys_row = {
        "strategy": "system",
        "cost_matrix": cost_matrix.name,
        "total_cost": sys_eval.total_cost,
        "n_decisions": sys_eval.n_decisions,
        "p_greenlight": sys_eval.action_proportions["Greenlight"],
        "p_pass": sys_eval.action_proportions["Pass"],
        "p_refer": sys_eval.action_proportions["Refer"],
        "n_greenlight": sys_eval.action_counts["Greenlight"],
        "n_pass": sys_eval.action_counts["Pass"],
        "n_refer": sys_eval.action_counts["Refer"],
        "cost_per_film_M": sys_eval.total_cost / max(sys_eval.n_decisions, 1) / 1_000_000,
    }
    baseline_rows: list[dict[str, Any]] = []
    for bname, bfn in BASELINES.items():
        bactions = bfn(imdb_ids, genres=genres)
        beval = evaluate(bname, bactions, true_labels.tolist(), cost_matrix)
        baseline_rows.append({
            "strategy": bname,
            "cost_matrix": cost_matrix.name,
            "total_cost": beval.total_cost,
            "n_decisions": beval.n_decisions,
            "p_greenlight": beval.action_proportions["Greenlight"],
            "p_pass": beval.action_proportions["Pass"],
            "p_refer": beval.action_proportions["Refer"],
            "n_greenlight": beval.action_counts["Greenlight"],
            "n_pass": beval.action_counts["Pass"],
            "n_refer": beval.action_counts["Refer"],
            "cost_per_film_M": beval.total_cost / max(beval.n_decisions, 1) / 1_000_000,
        })
    return sys_row, baseline_rows


def refer_cost_sweep(
    imdb_ids: list[str],
    calibrated_probs: np.ndarray,
    true_labels: np.ndarray,
) -> pd.DataFrame:
    """System cost + action distribution under each refer-cost variant."""
    rows = []
    for cm in refer_cost_variants():
        decisions = decide_batch(imdb_ids, calibrated_probs.tolist(), cm)
        actions = [d.recommended_action for d in decisions]
        result = evaluate("system", actions, true_labels.tolist(), cm)
        rows.append({
            "cost_matrix_name": cm.name,
            "cost_refer_flop": cm.cost_refer_flop,
            "total_cost_system": result.total_cost,
            "p_greenlight": result.action_proportions["Greenlight"],
            "p_pass": result.action_proportions["Pass"],
            "p_refer": result.action_proportions["Refer"],
            "n_greenlight": result.action_counts["Greenlight"],
            "n_pass": result.action_counts["Pass"],
            "n_refer": result.action_counts["Refer"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 4 — attribution stability (test-set SHAP)
# ---------------------------------------------------------------------------


def shap_test_global_ranking(
    explainer_bundle: dict,
    pipeline,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute test-set TreeSHAP, return (ranked DataFrame, raw SHAP array)."""
    explainer = explainer_bundle["explainer"]
    feature_names = explainer_bundle["feature_names"]

    pre = pipeline.named_steps.get("pre")
    X_pre = np.asarray(pre.transform(X_test)) if pre is not None else X_test.values
    raw = explainer.shap_values(X_pre)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    mean_abs = np.mean(np.abs(arr), axis=0)
    mean_signed = np.mean(arr, axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, arr


def shap_vs_native_test(
    test_ranking: pd.DataFrame, target: str = "roi_gt_2",
) -> tuple[pd.DataFrame, float]:
    """Spearman ρ between test-set SHAP ranking and Phase 4 native importance."""
    from src.utils import paths
    native_path = paths.REPORTS_TABLES_DIR / f"phase4_importance_{target}.csv"
    if not native_path.is_file():
        logger.warning("Native importance not found at %s", native_path)
        return pd.DataFrame(), float("nan")
    native = pd.read_csv(native_path)[["feature", "importance"]].copy()
    native["native_rank"] = native["importance"].rank(
        ascending=False, method="min",
    ).astype(int)
    merged = test_ranking.merge(native, on="feature", how="outer")
    merged["shap_rank"] = merged["rank"]
    valid = merged.dropna(subset=["shap_rank", "native_rank"])
    if len(valid) < 5:
        rho = float("nan")
    else:
        r, _ = spearmanr(valid["shap_rank"], valid["native_rank"])
        rho = float(r)
    return merged.sort_values("shap_rank"), rho


def shap_test_vs_cal_overlap(
    test_ranking: pd.DataFrame, target: str = "roi_gt_2",
    top_k: int = 15,
) -> tuple[float, pd.DataFrame]:
    """Jaccard overlap of top-K SHAP features between Phase 7 cal and Phase 8 test."""
    from src.utils import paths
    cal_path = paths.REPORTS_TABLES_DIR / f"phase7_global_shap_{target}.csv"
    if not cal_path.is_file():
        return float("nan"), pd.DataFrame()
    cal = pd.read_csv(cal_path)
    cal_top = set(cal.head(top_k)["feature"].tolist())
    test_top = set(test_ranking.head(top_k)["feature"].tolist())
    union = cal_top | test_top
    if not union:
        return float("nan"), pd.DataFrame()
    jaccard = len(cal_top & test_top) / len(union)

    overlap_df = pd.DataFrame({
        "feature": sorted(union),
        "in_cal_top15": [f in cal_top for f in sorted(union)],
        "in_test_top15": [f in test_top for f in sorted(union)],
    })
    return float(jaccard), overlap_df
