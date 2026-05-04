"""Global SHAP-based feature ranking + comparison vs Phase 4 native importance.

Two responsibilities:

* :func:`global_shap_ranking` — given (n_samples, n_features) SHAP
  values and the feature names, return a DataFrame ranked by
  ``mean(|SHAP|)`` across samples.
* :func:`compare_to_native` — load the Phase 4
  ``phase4_importance_<target>.csv`` table and compute the Spearman
  rank correlation between the SHAP ranking and the native
  ``feature_importances_`` ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import paths


def global_shap_ranking(
    shap_vals: np.ndarray, feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean |SHAP|, descending."""
    if shap_vals.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP shape mismatch: {shap_vals.shape[1]} cols vs "
            f"{len(feature_names)} feature names",
        )
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_signed = np.mean(shap_vals, axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def compare_to_native(
    target: str, shap_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the SHAP ranking with Phase 4's native importance ranking."""
    native_path = paths.REPORTS_TABLES_DIR / f"phase4_importance_{target}.csv"
    if not native_path.is_file():
        raise FileNotFoundError(f"Native importance not found: {native_path}")
    native = pd.read_csv(native_path)
    native = native[["feature", "importance"]].copy()
    native["native_rank"] = (
        native["importance"].rank(ascending=False, method="min").astype(int)
    )
    merged = shap_ranking.merge(native, on="feature", how="outer")
    merged["shap_rank"] = merged["rank"]
    return merged[[
        "feature", "mean_abs_shap", "mean_signed_shap", "shap_rank",
        "importance", "native_rank",
    ]].sort_values("shap_rank").reset_index(drop=True)


def spearman_rank_correlation(comparison: pd.DataFrame) -> float:
    """Spearman correlation between shap_rank and native_rank, dropping NaN."""
    valid = comparison.dropna(subset=["shap_rank", "native_rank"])
    if len(valid) < 5:
        return float("nan")
    rho, _ = spearmanr(valid["shap_rank"], valid["native_rank"])
    return float(rho)
