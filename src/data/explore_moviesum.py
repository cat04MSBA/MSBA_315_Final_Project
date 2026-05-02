"""Phase 1, Task 3 — MovieSum exploratory profile.

Loads all three MovieSum splits, confirms the documented count and IMDb-ID
coverage, parses one screenplay's XML to verify the structure, spot-checks
five randomly sampled screenplays, and saves a length-distribution plot.

Run from the project root:

    python -m src.data.explore_moviesum

Outputs:

- ``reports/figures/phase1_moviesum_length_distribution.png``
- ``reports/tables/phase1_moviesum_summary.csv``
- Random spot-check excerpts logged to stderr (also captured in the
  phase summary).
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_moviesum import imdb_id_validity, load_moviesum
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Per the MovieSum README: 1800 train + 200 val + 200 test = 2200 screenplays.
EXPECTED_TOTAL = 2200
RANDOM_SEED = 42  # Project-wide standard seed (CLAUDE_CODE_GUIDELINES Section 3).


def _parse_one_script(xml_string: str) -> dict[str, int]:
    """Parse a screenplay's XML and count its structural elements.

    A confirmation that the on-disk format matches the structure documented
    in ``PROJECT_CONTEXT.txt`` Section 4. Counts scenes, character speakers,
    dialogue lines, stage directions, and scene descriptions.

    Returns ``{"_parse_error": 1}`` if the XML cannot be parsed.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        logger.warning("XML parse failed: %s", exc)
        return {"_parse_error": 1}

    counts: Counter[str] = Counter()
    counts["scene"] = len(root.findall("scene"))
    for tag in ("stage_direction", "scene_description", "character", "dialogue"):
        counts[tag] = sum(len(scene.findall(tag)) for scene in root.findall("scene"))
    counts["unique_characters"] = len(
        {c.text for scene in root.findall("scene") for c in scene.findall("character") if c.text}
    )
    return dict(counts)


def _plot_length_distribution(df: pd.DataFrame, out_path) -> None:
    """Two-panel length plot: raw character count and log10 character count."""
    lengths = df["script_char_len"]
    fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_raw.hist(lengths / 1000.0, bins=40, edgecolor="black", alpha=0.85)
    ax_raw.set_xlabel("Screenplay length (thousand chars)")
    ax_raw.set_ylabel("Number of screenplays")
    ax_raw.set_title("Raw scale")
    ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

    ax_log.hist(lengths / 1000.0, bins=40, log=True, edgecolor="black", alpha=0.85)
    ax_log.set_xlabel("Screenplay length (thousand chars)")
    ax_log.set_ylabel("Number of screenplays (log scale)")
    ax_log.set_title("Log y-axis (heavy-tail visibility)")
    ax_log.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"MovieSum screenplay length — N={len(df):,}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _spot_check(df: pd.DataFrame, n: int = 5, seed: int = RANDOM_SEED) -> None:
    """Sample screenplays at random and log a short excerpt of each.

    Confirms by eye that the data on disk looks like real screenplays
    (scene boundaries, character names, dialogue) — required by the brief.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), n)
    logger.info("Spot-checking %d random screenplays (seed=%d):", n, seed)
    for i in indices:
        row = df.iloc[i]
        head = (row["script"] or "")[:600].replace("\n", " ⏎ ")
        logger.info(
            "  [%d] %s (%s, len=%d chars, split=%s)\n    %s ...",
            i, row["movie_name"], row["imdb_id"], row["script_char_len"], row["origin_split"], head,
        )


def main() -> None:
    paths.ensure_dirs()

    df = load_moviesum()
    logger.info("Total screenplays loaded: %d (expected %d)", len(df), EXPECTED_TOTAL)
    if len(df) != EXPECTED_TOTAL:
        logger.warning("MovieSum total mismatch: got %d, README claims %d", len(df), EXPECTED_TOTAL)

    validity = imdb_id_validity(df)
    logger.info("IMDb ID validity: %s", validity)

    split_counts = df["origin_split"].value_counts().to_dict()
    logger.info("Origin-split breakdown: %s", split_counts)

    year_summary = df["year_in_title"].dropna().describe()
    logger.info("Year-in-title summary:\n%s", year_summary.to_string())

    length_summary = df["script_char_len"].describe()
    logger.info("Screenplay length (chars) summary:\n%s", length_summary.to_string())

    # Parse one screenplay end-to-end to verify the documented XML schema holds.
    sample_idx = 0
    sample_counts = _parse_one_script(df.iloc[sample_idx]["script"])
    logger.info(
        "Parsed sample screenplay (idx=%d, %s) structural counts: %s",
        sample_idx, df.iloc[sample_idx]["movie_name"], sample_counts,
    )

    _spot_check(df, n=5)

    _plot_length_distribution(
        df, paths.REPORTS_FIGURES_DIR / "phase1_moviesum_length_distribution.png"
    )

    summary_table = pd.DataFrame(
        {
            "metric": [
                "total_screenplays",
                "valid_imdb_id",
                "invalid_or_missing_imdb_id",
                "unique_valid_imdb_ids",
                "split_train", "split_val", "split_test",
                "length_mean_chars", "length_median_chars", "length_min_chars", "length_max_chars",
                "year_min", "year_max",
            ],
            "value": [
                len(df),
                validity["valid_imdb_id"],
                validity["invalid_or_missing"],
                validity["unique_valid_ids"],
                split_counts.get("train", 0), split_counts.get("val", 0), split_counts.get("test", 0),
                int(length_summary["mean"]), int(length_summary["50%"]),
                int(length_summary["min"]), int(length_summary["max"]),
                int(df["year_in_title"].min()) if df["year_in_title"].notna().any() else None,
                int(df["year_in_title"].max()) if df["year_in_title"].notna().any() else None,
            ],
        }
    )
    table_path = paths.REPORTS_TABLES_DIR / "phase1_moviesum_summary.csv"
    summary_table.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)


if __name__ == "__main__":
    main()
