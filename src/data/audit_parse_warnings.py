"""Phase 2 audit: empirical validation of parser recovery rules.

Validates that the parser's recovery rules do not introduce systematic
bias in the structural metrics, and dumps source XML alongside parsed
output for the worst-case films to confirm the recoveries did the right
thing in practice.

The audit answers two questions:

1. Does ``parse_warning_count`` correlate with structural metrics in
   the way the case analysis predicts? Specifically:

   * Negative correlation expected with
     ``mean_dialogue_line_length = total_dialogue_chars /
     n_dialogue_lines``. Per the case analysis (Cases 5-8 in the
     parser), warning-heavy films include more empty-string-text
     dialogue placeholders, dragging the mean line length downward.
   * Approximately uncorrelated with ``n_scenes``,
     ``n_unique_characters``, ``n_dialogue_lines``,
     ``total_dialogue_chars``, ``dialogue_to_total_text_ratio``,
     ``script_char_len``, decade, and ``primary_genre``. Anything
     with ``|r| > 0.3`` (or ``η² > 0.09`` for the categorical) is
     flagged as a potential systematic bias.

2. On the five films with the highest ``parse_warning_count``, do the
   parser's recoveries produce sensible output? The audit dumps the
   raw XML for the scenes named in the first warnings and the
   corresponding parsed ``Scene`` so the recovery can be inspected
   manually.

Outputs:

* ``reports/tables/phase2_parse_warning_audit.csv`` (correlation table)
* ``reports/figures/phase2_parse_warning_correlations.png`` (scatter grid)
* ``reports/tables/phase2_top5_warnings_inspection.md`` (XML vs parsed)

Run from the project root::

    python -m src.data.audit_parse_warnings
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python3 -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.load_moviesum import load_moviesum
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Threshold for flagging a structural metric as systematically biased.
# |Pearson r| > 0.3 corresponds to r^2 > 0.09; for categorical η² we use
# the same 0.09 threshold (η² is comparable to r²).
CORR_FLAG_THRESHOLD = 0.30
ETA_SQ_FLAG_THRESHOLD = 0.09

# Numeric metrics whose relationship with parse_warning_count we test.
NUMERIC_METRICS = (
    "n_scenes",
    "n_unique_characters",
    "n_dialogue_lines",
    "total_dialogue_chars",
    "dialogue_to_total_text_ratio",
    "mean_dialogue_line_length",
    "script_char_len",
    "decade",
)

# Number of films to inspect by hand at the top of the warning distribution.
TOP_K_INSPECTION = 5


# ---------------------------------------------------------------------------
# Loading + derived columns
# ---------------------------------------------------------------------------

def load_corpus_with_derived_columns() -> pd.DataFrame:
    """Load the master parquet and add the audit-only derived columns."""
    parquet = paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    df = pd.read_parquet(parquet)

    # mean_dialogue_line_length is the audit's primary expected-correlate.
    # `clip(lower=1)` guards the divide-by-zero edge case (a film with 0
    # dialogue lines, which the validator's hard_asserts already excludes
    # but is defensive).
    df["mean_dialogue_line_length"] = (
        df["total_dialogue_chars"] / df["n_dialogue_lines"].clip(lower=1)
    )

    # Decade as a numeric correlate (films share the decade ordering;
    # treating it as continuous Pearson/Spearman target is appropriate
    # since the spacing is uniform).
    df["decade"] = (df["release_year_parsed"] // 10 * 10).astype("Int64")

    logger.info("Loaded %s films for audit", f"{len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------

def numeric_correlation_row(
    df: pd.DataFrame, metric: str, target: str = "parse_warning_count"
) -> list[dict]:
    """Compute Pearson and Spearman for one numeric metric vs the target.

    Returns two rows (one per test) for the audit CSV.
    """
    valid = df[[metric, target]].dropna()
    if len(valid) < 30:
        return [
            {"variable": metric, "test": test, "statistic": float("nan"),
             "p_value": float("nan"), "flagged": False,
             "notes": f"n={len(valid)} insufficient"}
            for test in ("pearson_r", "spearman_rho")
        ]

    r, p_pearson = stats.pearsonr(valid[target], valid[metric])
    rho, p_spearman = stats.spearmanr(valid[target], valid[metric])

    return [
        {"variable": metric, "test": "pearson_r", "statistic": float(r),
         "p_value": float(p_pearson),
         "flagged": abs(r) > CORR_FLAG_THRESHOLD,
         "notes": f"n={len(valid)}"},
        {"variable": metric, "test": "spearman_rho", "statistic": float(rho),
         "p_value": float(p_spearman),
         "flagged": abs(rho) > CORR_FLAG_THRESHOLD,
         "notes": f"n={len(valid)}"},
    ]


def categorical_eta_squared_row(
    df: pd.DataFrame, group_col: str, target: str = "parse_warning_count"
) -> dict:
    """One-way ANOVA + η² for a categorical grouping variable."""
    valid = df[[group_col, target]].dropna()
    groups = [g[target].values for _, g in valid.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)

    overall_mean = valid[target].mean()
    ss_total = float(((valid[target] - overall_mean) ** 2).sum())
    ss_between = float(sum(
        len(g) * (g[target].mean() - overall_mean) ** 2
        for _, g in valid.groupby(group_col)
    ))
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

    return {
        "variable": group_col,
        "test": "eta_squared",
        "statistic": eta_sq,
        "p_value": float(p_value),
        "flagged": eta_sq > ETA_SQ_FLAG_THRESHOLD,
        "notes": f"F={f_stat:.2f}, n_groups={valid[group_col].nunique()}",
    }


def build_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the audit CSV with correlation and ANOVA results."""
    rows: list[dict] = []
    for metric in NUMERIC_METRICS:
        rows.extend(numeric_correlation_row(df, metric))
    rows.append(categorical_eta_squared_row(df, "primary_genre_bucketed"))

    audit_df = pd.DataFrame(rows)
    # Round for readability while keeping enough precision to interpret.
    audit_df["statistic"] = audit_df["statistic"].round(4)
    audit_df["p_value"] = audit_df["p_value"].round(6)
    return audit_df[["variable", "test", "statistic", "p_value",
                     "flagged", "notes"]]


# ---------------------------------------------------------------------------
# Plot grid
# ---------------------------------------------------------------------------

def plot_correlation_grid(df: pd.DataFrame, out_path) -> None:
    """3x3 grid: 8 numeric scatter panels + 1 boxplot per genre.

    Annotates each numeric panel with the Spearman ρ (more robust to
    the heavy-tailed warning distribution than Pearson r).
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    target = "parse_warning_count"

    for i, metric in enumerate(NUMERIC_METRICS):
        ax = axes.flat[i]
        valid = df[[target, metric]].dropna()
        ax.scatter(valid[target], valid[metric], alpha=0.25, s=10,
                   color="steelblue", edgecolors="none")
        rho, p = stats.spearmanr(valid[target], valid[metric])
        flag = "  [FLAGGED]" if abs(rho) > CORR_FLAG_THRESHOLD else ""
        ax.set(xlabel=target, ylabel=metric,
               title=f"{metric}\nSpearman ρ = {rho:+.3f}, p = {p:.2g}{flag}")
        ax.grid(linestyle=":", alpha=0.5)

    # 9th panel: per-genre boxplot.
    ax = axes.flat[8]
    valid = df[[target, "primary_genre_bucketed"]].dropna()
    genre_order = (
        valid.groupby("primary_genre_bucketed")[target]
        .median().sort_values().index.tolist()
    )
    data_per_genre = [
        valid.loc[valid["primary_genre_bucketed"] == g, target].values
        for g in genre_order
    ]
    eta_sq_row = categorical_eta_squared_row(df, "primary_genre_bucketed")
    flag = "  [FLAGGED]" if eta_sq_row["flagged"] else ""
    ax.boxplot(data_per_genre, tick_labels=genre_order, showfliers=False)
    ax.set(ylabel=target,
           title=f"By primary_genre_bucketed\nη² = {eta_sq_row['statistic']:.3f}{flag}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Phase 2 parse-warning audit: correlations with structural metrics\n"
        f"(|ρ| > {CORR_FLAG_THRESHOLD} flags potential systematic bias)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.info("Saved correlation grid")


# ---------------------------------------------------------------------------
# Top-5 inspection: XML vs parsed output
# ---------------------------------------------------------------------------

_SCENE_NUMBER_PATTERN = re.compile(r"scene (\d+):")


def _extract_scene_numbers_from_warnings(warnings: tuple[str, ...]) -> list[int]:
    """Pull the scene numbers referenced by a list of warning strings."""
    seen: list[int] = []
    for w in warnings:
        match = _SCENE_NUMBER_PATTERN.search(w)
        if match:
            n = int(match.group(1))
            if n not in seen:
                seen.append(n)
    return seen


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def build_top5_inspection_report(
    df: pd.DataFrame,
    parsed_screenplays: dict,
    moviesum_df: pd.DataFrame,
) -> str:
    """Generate the top-5 markdown inspection report as a single string."""
    top5 = df.nlargest(TOP_K_INSPECTION, "parse_warning_count")

    out: list[str] = [
        f"# Phase 2 — Top {TOP_K_INSPECTION} films by `parse_warning_count`: "
        "raw XML vs parsed output\n",
        "This report dumps source XML alongside the parser's output for the "
        "five films with the highest warning counts in the corpus. Its "
        "purpose is to confirm that the parser's recovery rules produced "
        "sensible output in the worst empirical cases.\n",
        "Each entry shows: the film and its top-line counts, the first "
        "five warnings emitted during parsing, and for each of the first "
        "two scenes named in those warnings, the raw XML alongside the "
        "parsed `Scene` object's output.\n",
    ]

    moviesum_indexed = moviesum_df.set_index("imdb_id")

    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        imdb_id = row["imdb_id"]
        parsed = parsed_screenplays[imdb_id]

        try:
            raw_xml = moviesum_indexed.loc[imdb_id, "script"]
        except KeyError:
            raw_xml = None

        out.append(
            f"\n## {rank}. {row['movie_name']!r} (`{imdb_id}`) — "
            f"{int(row['parse_warning_count'])} warnings\n"
        )
        out.append(
            f"- `n_scenes`: {int(row['n_scenes'])}\n"
            f"- `n_dialogue_lines`: {int(row['n_dialogue_lines']):,}\n"
            f"- `n_unique_characters`: {int(row['n_unique_characters'])}\n"
            f"- `total_dialogue_chars`: {int(row['total_dialogue_chars']):,}\n"
            f"- `mean_dialogue_line_length`: "
            f"{row['total_dialogue_chars'] / max(row['n_dialogue_lines'], 1):.1f}\n"
        )

        out.append("\n### First 5 warnings\n")
        for w in parsed.parse_warnings[:5]:
            out.append(f"- {w}")

        scene_nums = _extract_scene_numbers_from_warnings(
            parsed.parse_warnings[:10]
        )[:2]
        if not scene_nums or raw_xml is None:
            out.append(
                "\n_(No scene-level warnings to dump, or raw XML unavailable.)_\n"
            )
            continue

        try:
            root = ET.fromstring(raw_xml)
            scene_elements = root.findall("scene")
        except ET.ParseError as exc:
            out.append(f"\n_(XML parse failed re-loading raw script: {exc})_\n")
            continue

        for sn in scene_nums:
            if not (1 <= sn <= len(scene_elements)):
                continue
            xml_str = ET.tostring(scene_elements[sn - 1], encoding="unicode")
            parsed_scene = parsed.scenes[sn - 1]

            out.append(f"\n### Scene {sn} — raw XML\n")
            out.append("```xml")
            out.append(_truncate(xml_str, 2000))
            out.append("```")

            out.append(f"\n### Scene {sn} — parsed output\n")
            out.append("```python")
            out.append(
                f"stage_direction:   {parsed_scene.stage_direction[:200]!r}"
            )
            out.append(
                f"scene_description: {parsed_scene.scene_description[:200]!r}"
            )
            out.append(f"n_dialogue_units:  {len(parsed_scene.dialogue_units)}")
            out.append("First 8 dialogue_units:")
            for char, line in parsed_scene.dialogue_units[:8]:
                out.append(f"  ({char!r}, {line[:80]!r})")
            out.append("```")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run the full audit and write all three artifacts."""
    paths.ensure_dirs()

    df = load_corpus_with_derived_columns()
    audit_df = build_audit_table(df)

    audit_csv = paths.REPORTS_TABLES_DIR / "phase2_parse_warning_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    logger.info("Saved audit table")

    plot_correlation_grid(
        df, paths.REPORTS_FIGURES_DIR / "phase2_parse_warning_correlations.png"
    )

    # Top-5 inspection requires raw XML, so reload MovieSum.
    moviesum_df = load_moviesum(include_script=True)
    pkl_path = paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl"
    with pkl_path.open("rb") as f:
        parsed_screenplays = pickle.load(f)

    inspection_md = build_top5_inspection_report(
        df, parsed_screenplays, moviesum_df
    )
    inspection_path = paths.REPORTS_TABLES_DIR / "phase2_top5_warnings_inspection.md"
    inspection_path.write_text(inspection_md, encoding="utf-8")
    logger.info("Saved top-%d inspection report", TOP_K_INSPECTION)

    # Print the audit table to stdout in a form fit for the phase summary.
    print("\n=== Phase 2 parse-warning audit ===")
    print(audit_df.to_string(index=False))
    print()
    flagged = audit_df[audit_df["flagged"]]
    if flagged.empty:
        print("No correlations exceeded the |r| > 0.3 (or η² > 0.09) flag "
              "threshold.")
    else:
        print(f"FLAGGED ({len(flagged)} entries):")
        print(flagged.to_string(index=False))
    return audit_df


if __name__ == "__main__":
    main()
