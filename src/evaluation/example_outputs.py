"""Five-film curated example gallery for the Phase 9 presentation.

Selection rules (locked in pre-registration Section 4.7):

1. **High-confidence Greenlight** — film with highest calibrated
   probability that the system actually recommends Greenlight.
2. **High-confidence Pass** — if any exist; otherwise the lowest-
   probability Refer.
3. **High-uncertainty Refer near 0.50** — film closest to 0.5
   calibrated probability.
4. **Adventure / Fantasy / Sci-Fi true positive** — a positive
   ``roi_gt_2`` film in the genre-tractable cluster correctly
   identified.
5. **Drama / Comedy / Romance defer** — a film in the genre-
   intractable cluster the system correctly defers on.

If a category yields no eligible film on the test set, the
substitution is documented in the rendered output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


GENRE_TRACTABLE: tuple[str, ...] = ("Adventure", "Fantasy", "Science Fiction")
GENRE_INTRACTABLE: tuple[str, ...] = ("Drama", "Comedy", "Romance")


@dataclass(frozen=True)
class ExampleSelection:
    """Single example film with category and explanatory note."""

    category: str
    imdb_id: str
    movie_name: str
    note: str


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _safe_lookup(df: pd.DataFrame, mask: pd.Series, sort_col: str, ascending: bool):
    """Return the first row matching mask, sorted by sort_col, or None."""
    sub = df[mask].sort_values(sort_col, ascending=ascending)
    if sub.empty:
        return None
    return sub.iloc[0]


def select_examples(
    per_film_df: pd.DataFrame,
    *,
    target_col: str = "true_label",
    prob_col: str = "calibrated_probability_roi_gt_2",
    action_col: str = "recommended_action",
    genre_col: str = "primary_genre_bucketed",
    name_col: str = "movie_name",
) -> list[ExampleSelection]:
    """Apply the five selection rules to ``per_film_df`` and return the picks.

    The DataFrame is expected to already carry per-film outputs from
    :func:`src.evaluation.pipeline.run_batch` joined with the
    test-set ground truth.
    """
    selections: list[ExampleSelection] = []

    # Rule 1 — high-confidence Greenlight
    rule1 = _safe_lookup(
        per_film_df,
        per_film_df[action_col] == "Greenlight",
        prob_col, ascending=False,
    )
    if rule1 is not None:
        selections.append(ExampleSelection(
            category="High-confidence Greenlight",
            imdb_id=str(rule1["imdb_id"]),
            movie_name=str(rule1[name_col]),
            note=f"Highest-probability Greenlight (P={rule1[prob_col]:.3f}).",
        ))
    else:
        # Substitution: closest-to-Greenlight Refer film (highest probability).
        sub = per_film_df.sort_values(prob_col, ascending=False).iloc[0]
        selections.append(ExampleSelection(
            category="High-confidence Greenlight (substituted)",
            imdb_id=str(sub["imdb_id"]),
            movie_name=str(sub[name_col]),
            note=(
                "No Greenlight recommendation on test set; substituting the "
                f"highest-probability Refer (P={sub[prob_col]:.3f})."
            ),
        ))

    # Rule 2 — high-confidence Pass (if any) else lowest-probability Refer
    rule2 = _safe_lookup(
        per_film_df,
        per_film_df[action_col] == "Pass",
        prob_col, ascending=True,
    )
    if rule2 is not None:
        selections.append(ExampleSelection(
            category="High-confidence Pass",
            imdb_id=str(rule2["imdb_id"]),
            movie_name=str(rule2[name_col]),
            note=f"Lowest-probability Pass (P={rule2[prob_col]:.3f}).",
        ))
    else:
        sub = (
            per_film_df[per_film_df[action_col] == "Refer"]
            .sort_values(prob_col, ascending=True)
            .iloc[0]
        )
        selections.append(ExampleSelection(
            category="High-confidence Pass (substituted)",
            imdb_id=str(sub["imdb_id"]),
            movie_name=str(sub[name_col]),
            note=(
                "No Pass recommendations on test set (Phase 6 trigger #1 "
                "fired in Phase 6 cal); substituting the lowest-probability "
                f"Refer (P={sub[prob_col]:.3f})."
            ),
        ))

    # Rule 3 — refer closest to 0.50
    refers = per_film_df[per_film_df[action_col] == "Refer"].copy()
    if not refers.empty:
        refers["dist_to_0.5"] = (refers[prob_col] - 0.5).abs()
        rule3 = refers.sort_values("dist_to_0.5", ascending=True).iloc[0]
        selections.append(ExampleSelection(
            category="High-uncertainty Refer near 0.50",
            imdb_id=str(rule3["imdb_id"]),
            movie_name=str(rule3[name_col]),
            note=f"Closest-to-0.5 Refer (P={rule3[prob_col]:.3f}).",
        ))
    else:
        selections.append(ExampleSelection(
            category="High-uncertainty Refer (skipped)",
            imdb_id="",
            movie_name="",
            note="No Refer recommendations on test set.",
        ))

    # Rule 4 — Adventure / Fantasy / Sci-Fi true positive correctly identified
    mask4 = (
        per_film_df[genre_col].isin(GENRE_TRACTABLE)
        & (per_film_df[target_col] == 1)
        & (per_film_df[prob_col] >= 0.5)
    )
    rule4 = _safe_lookup(per_film_df, mask4, prob_col, ascending=False)
    if rule4 is not None:
        selections.append(ExampleSelection(
            category="Genre-tractable true positive",
            imdb_id=str(rule4["imdb_id"]),
            movie_name=str(rule4[name_col]),
            note=(
                f"{rule4[genre_col]} hit correctly recognized "
                f"(P={rule4[prob_col]:.3f}; action={rule4[action_col]})."
            ),
        ))
    else:
        # Substitute: any positive film among the tractable cluster
        mask = (
            per_film_df[genre_col].isin(GENRE_TRACTABLE)
            & (per_film_df[target_col] == 1)
        )
        sub = _safe_lookup(per_film_df, mask, prob_col, ascending=False)
        if sub is not None:
            selections.append(ExampleSelection(
                category="Genre-tractable true positive (substituted)",
                imdb_id=str(sub["imdb_id"]),
                movie_name=str(sub[name_col]),
                note=(
                    f"No tractable-cluster hit with P >= 0.5 on test; "
                    f"substituting the highest-probability one "
                    f"(P={sub[prob_col]:.3f})."
                ),
            ))
        else:
            selections.append(ExampleSelection(
                category="Genre-tractable true positive (skipped)",
                imdb_id="",
                movie_name="",
                note="No Adventure/Fantasy/Sci-Fi positives on test set.",
            ))

    # Rule 5 — Drama / Comedy / Romance correctly deferred
    mask5 = (
        per_film_df[genre_col].isin(GENRE_INTRACTABLE)
        & (per_film_df[action_col] == "Refer")
    )
    rule5_pool = per_film_df[mask5].copy()
    if not rule5_pool.empty:
        rule5_pool["dist_to_0.5"] = (rule5_pool[prob_col] - 0.5).abs()
        rule5 = rule5_pool.sort_values("dist_to_0.5", ascending=True).iloc[0]
        selections.append(ExampleSelection(
            category="Genre-intractable defer",
            imdb_id=str(rule5["imdb_id"]),
            movie_name=str(rule5[name_col]),
            note=(
                f"{rule5[genre_col]} film correctly deferred "
                f"(P={rule5[prob_col]:.3f}; near-0.5 uncertainty)."
            ),
        ))
    else:
        selections.append(ExampleSelection(
            category="Genre-intractable defer (skipped)",
            imdb_id="",
            movie_name="",
            note="No Drama/Comedy/Romance Refers on test set.",
        ))

    return selections


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_gallery_markdown(
    selections: list[ExampleSelection],
    per_film_df: pd.DataFrame,
    *,
    target_col: str = "true_label",
) -> str:
    """Compose the deliverable Markdown gallery."""
    lines: list[str] = [
        "# Phase 8 — Curated Example Gallery (Test Set)",
        "",
        (
            "Five films selected per the locked rules in "
            "``docs/proposals/phase8_preregistration.md`` Section 4.7. "
            "All films come from the held-out 257-film test set."
        ),
        "",
    ]
    for sel in selections:
        lines.append(f"## {sel.category}")
        lines.append("")
        if not sel.imdb_id:
            lines.append(f"_Skipped: {sel.note}_")
            lines.append("")
            continue
        row = per_film_df[per_film_df["imdb_id"] == sel.imdb_id]
        if row.empty:
            lines.append(f"_Selection {sel.imdb_id} not found in per-film table._")
            lines.append("")
            continue
        r = row.iloc[0]
        true_label = int(r.get(target_col, -1)) if not pd.isna(r.get(target_col, np.nan)) else None
        lines.extend([
            f"**Film:** {sel.movie_name}  ",
            f"**IMDb ID:** {sel.imdb_id}  ",
            f"**Genre:** {r.get('primary_genre_bucketed', 'n/a')}  ",
            f"**Release year:** {int(r.get('release_year_parsed', 0)) if not pd.isna(r.get('release_year_parsed', np.nan)) else 'n/a'}  ",
            (
                f"**True ROI > 2x:** {bool(true_label)}  "
                if true_label is not None else
                "**True ROI > 2x:** n/a  "
            ),
            "",
            f"**Note:** {sel.note}",
            "",
            "### Layered triage output",
            "",
            f"| Layer | Output |",
            f"|---|---|",
            f"| 1 — log_roi point prediction | {r['log_roi_point_prediction']:.3f} |",
            f"| 1 — uncalibrated P(ROI > 2x) | {r['roi_gt_2_uncalibrated_probability']:.3f} |",
            f"| 2 — **calibrated P(ROI > 2x)** | **{r['calibrated_probability_roi_gt_2']:.3f}** |",
            f"| 2 — log_roi 90% interval | [{r.get('log_roi_lower_0.9', float('nan')):.2f}, {r.get('log_roi_upper_0.9', float('nan')):.2f}] |",
            f"| 2 — conformal set @ 0.90 (size) | {int(r.get('conf_roi_gt_2_set_size_0.9', -1))} |",
            f"| 3 — recommended action | **{r['recommended_action']}** |",
            f"| 3 — expected cost (Greenlight / Pass / Refer) | ${r['expected_cost_greenlight']/1e6:.2f}M / ${r['expected_cost_pass']/1e6:.2f}M / ${r['expected_cost_refer']/1e3:.1f}K |",
            "",
            "**Decision rationale:**",
            "",
            f"> {r['decision_rationale']}",
            "",
            "**SHAP rationale (Layer 4):**",
            "",
            f"> {r['shap_rationale']}",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)
