"""Per-film attributions + rationale strings.

For each film, builds a structured per-film payload containing:

* The top-K features by absolute SHAP value (positive contributors
  pushing probability up; negative pulling it down).
* A natural-language rationale string concatenating the top
  positive and negative contributors with their effects.
* The Phase 6 recommended action (read from
  ``reports/tables/phase6_decisions.csv`` if available).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import paths


TOP_K: int = 5


def _format_shap_effect(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.3f}"


def _readable_feature(feature: str) -> str:
    """Light cosmetic improvement on raw feature names for the report."""
    if feature.startswith("genre_"):
        return f"Genre={feature.removeprefix('genre_')}"
    if feature.startswith("topic_") and feature.endswith("_proportion"):
        return f"Topic {feature.removeprefix('topic_').removesuffix('_proportion')} proportion"
    if feature.startswith("network_"):
        return feature.replace("network_", "Network: ")
    if feature.startswith("embed_pc_"):
        return f"Embedding PC {feature.removeprefix('embed_pc_')}"
    if feature.startswith("log_"):
        return f"log({feature.removeprefix('log_')})"
    return feature


def per_film_attributions(
    imdb_ids: list[str],
    shap_vals: np.ndarray,
    feature_names: list[str],
    base_value: float,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """For each film, list the top-k positive and top-k negative SHAP features."""
    n = shap_vals.shape[0]
    rows = []
    for i in range(n):
        sv = shap_vals[i]
        order_abs = np.argsort(-np.abs(sv))
        # Split into positive and negative contributors
        positives = [(feature_names[j], float(sv[j])) for j in order_abs if sv[j] > 0]
        negatives = [(feature_names[j], float(sv[j])) for j in order_abs if sv[j] < 0]
        rows.append({
            "imdb_id": imdb_ids[i],
            "shap_sum": float(sv.sum()),
            "base_value": base_value,
            "top_pos_features": "; ".join(
                f"{_readable_feature(n)}={_format_shap_effect(v)}"
                for n, v in positives[:top_k]
            ),
            "top_neg_features": "; ".join(
                f"{_readable_feature(n)}={_format_shap_effect(v)}"
                for n, v in negatives[:top_k]
            ),
            "rationale_features": _build_rationale(
                base_value, positives[:top_k], negatives[:top_k],
            ),
        })
    return pd.DataFrame(rows)


def _build_rationale(
    base_value: float,
    positives: list[tuple[str, float]],
    negatives: list[tuple[str, float]],
) -> str:
    """Concatenate the top contributors into a natural-language sentence."""
    parts: list[str] = []
    if positives:
        pos_strs = ", ".join(
            f"{_readable_feature(n)} ({_format_shap_effect(v)} log-odds)"
            for n, v in positives[:3]
        )
        parts.append(f"Top features pushing probability up: {pos_strs}")
    if negatives:
        neg_strs = ", ".join(
            f"{_readable_feature(n)} ({_format_shap_effect(v)} log-odds)"
            for n, v in negatives[:3]
        )
        parts.append(f"Top features pulling probability down: {neg_strs}")
    return ". ".join(parts) + "." if parts else "No notable feature contributions."


def merge_with_phase6(per_film_df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 6 recommended_action and probability when available."""
    p6 = paths.REPORTS_TABLES_DIR / "phase6_decisions.csv"
    if not p6.is_file():
        return per_film_df
    decisions = pd.read_csv(p6)[
        ["imdb_id", "calibrated_probability", "recommended_action", "rationale", "true_label", "genre"]
    ].rename(columns={"rationale": "phase6_rationale"})
    return per_film_df.merge(decisions, on="imdb_id", how="left")
