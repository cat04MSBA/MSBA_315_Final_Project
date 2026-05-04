"""Phase 4 Tier-A2 diagnostic: genre + era-stratified OOF AUC.

When four model families with very different inductive biases all
converge to the same 0.60 to 0.63 OOF AUC range on ``roi_gt_2``, the
ceiling could be either uniform across the corpus (every film is
about equally hard) or non-uniform (some sub-corpora are tractable,
others are not). The two interpretations have very different report
implications.

This module slices the saved OOF predictions per (matrix, family) by
``primary_genre_bucketed`` and ``decade_bucket`` and recomputes
AUC-ROC per slice. The output is two tables (per-genre, per-decade)
and a heatmap-style figure.

Requires the metrics.json files written by the Phase 4 benchmark
(per-row OOF score and y_true).
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.phase4.figures import (
    HEADLINE_TARGET,
    PRIMARY_FAMILY_NAMES,
    latest_per_cell_dir,
)
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def _decade_bucket(year: int) -> str:
    """Match the bucketing convention used in Phase 3 split assignments."""
    if year < 1980:
        return "pre_1980s"
    if year < 1990:
        return "1980s"
    if year < 2000:
        return "1990s"
    if year < 2010:
        return "2000s"
    return "2010s_2020s"


def load_meta() -> pd.DataFrame:
    """Load the master corpus restricted to the train split, with bucket columns."""
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df = df[df["imdb_id"].isin(train_ids)].reset_index(drop=True)
    df["decade_bucket"] = df["release_year_parsed"].astype(int).apply(_decade_bucket)
    return df[["imdb_id", "primary_genre_bucketed", "decade_bucket"]]


def _load_oof(run_dir: Path, target: str) -> pd.DataFrame | None:
    """Load OOF score + y_true for one (matrix, family, target) cell.

    Returns DataFrame with columns ``imdb_id``, ``oof_score``, ``y_true``
    or ``None`` if the cell is missing the requested target.
    """
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)
    pt = m.get("per_target", {})
    if target not in pt:
        return None
    imdb_ids = pt.get("_imdb_ids")
    if imdb_ids is None:
        return None
    return pd.DataFrame({
        "imdb_id": imdb_ids,
        "oof_score": pt[target]["oof_score"],
        "y_true": pt[target]["y_true"],
    })


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def stratified_auc_table(
    target: str = HEADLINE_TARGET,
    matrix: str = "standalone_positive_union",
    families: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-genre and per-decade OOF AUC for the chosen cell.

    Returns (per-genre table, per-decade table). Each table has rows
    indexed by slice and columns one per family, plus a count column.
    """
    families = families or PRIMARY_FAMILY_NAMES
    meta = load_meta()
    phase_dir = paths.RUNS_DIR / "phase_4"

    per_genre_rows: dict[tuple[str, str], dict] = {}
    per_decade_rows: dict[tuple[str, str], dict] = {}

    for fam in families:
        run_dir = latest_per_cell_dir(phase_dir, matrix, fam)
        if run_dir is None:
            logger.warning("No run for matrix=%s family=%s", matrix, fam)
            continue
        oof = _load_oof(run_dir, target)
        if oof is None:
            logger.warning("No %s data in %s", target, run_dir)
            continue
        merged = meta.merge(oof, on="imdb_id", how="inner")

        for genre, sub in merged.groupby("primary_genre_bucketed"):
            key = ("genre", genre)
            row = per_genre_rows.setdefault(key, {"slice": genre, "n": int(len(sub)),
                                                   "n_pos": int(sub["y_true"].sum())})
            row[fam] = _safe_auc(sub["y_true"].values, sub["oof_score"].values)
        for decade, sub in merged.groupby("decade_bucket"):
            key = ("decade", decade)
            row = per_decade_rows.setdefault(key, {"slice": decade, "n": int(len(sub)),
                                                    "n_pos": int(sub["y_true"].sum())})
            row[fam] = _safe_auc(sub["y_true"].values, sub["oof_score"].values)

    genre_df = pd.DataFrame(list(per_genre_rows.values())).set_index("slice")
    decade_df = pd.DataFrame(list(per_decade_rows.values())).set_index("slice")
    # Sort decades chronologically.
    decade_order = ["pre_1980s", "1980s", "1990s", "2000s", "2010s_2020s"]
    decade_df = decade_df.reindex([d for d in decade_order if d in decade_df.index])
    return genre_df, decade_df


def plot_stratified(
    genre_df: pd.DataFrame,
    decade_df: pd.DataFrame,
    out_path: Path,
    target: str,
    matrix: str,
    families: list[str],
) -> Path:
    """Render the diagnostic as two heatmap panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * len(genre_df))))

    for ax, df, title in [
        (axes[0], genre_df, "Per primary_genre"),
        (axes[1], decade_df, "Per decade_bucket"),
    ]:
        mat = df[families].values
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.85)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=30, ha="right")
        ax.set_yticks(range(len(df)))
        labels = [f"{idx} (n={int(df.loc[idx, 'n'])})" for idx in df.index]
        ax.set_yticklabels(labels)
        ax.set_title(title)
        for i in range(len(df)):
            for j in range(len(families)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.5 < v < 0.75 else "white", fontsize=8,
                    )
        fig.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(
        f"Phase 4 stratified OOF AUC on {target} ({matrix})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=HEADLINE_TARGET)
    parser.add_argument(
        "--matrix", default="standalone_positive_union",
        help="Matrix to slice (default the headline winner's matrix).",
    )
    parser.add_argument(
        "--out-genre", type=Path,
        default=paths.REPORTS_TABLES_DIR / "phase4_per_genre_auc.csv",
    )
    parser.add_argument(
        "--out-decade", type=Path,
        default=paths.REPORTS_TABLES_DIR / "phase4_per_decade_auc.csv",
    )
    parser.add_argument(
        "--out-figure", type=Path,
        default=paths.REPORTS_FIGURES_DIR / "phase4_stratified_auc.png",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    set_log_level(args.log_level)
    paths.ensure_dirs()

    genre_df, decade_df = stratified_auc_table(
        target=args.target, matrix=args.matrix,
    )

    genre_df.to_csv(args.out_genre)
    decade_df.to_csv(args.out_decade)
    logger.info("Wrote %s (%d rows)", args.out_genre, len(genre_df))
    logger.info("Wrote %s (%d rows)", args.out_decade, len(decade_df))

    plot_stratified(
        genre_df, decade_df, args.out_figure,
        target=args.target, matrix=args.matrix, families=PRIMARY_FAMILY_NAMES,
    )

    print()
    print("PER GENRE OOF AUC:")
    cols = ["n", "n_pos"] + PRIMARY_FAMILY_NAMES
    print(genre_df[cols].round(3).to_string())
    print()
    print("PER DECADE OOF AUC:")
    print(decade_df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
