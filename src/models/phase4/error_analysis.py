"""Phase 4 per-film error analysis.

Loads the OOF predictions saved in
``runs/phase_4/<timestamp>_<matrix>_<family>/metrics.json`` for the
headline winner per target and answers the more useful interpretive
question than feature importance: **which films does the model get
right, and which does it get wrong?**

Three outputs per target:

* ``phase4_predictions_<target>.csv``: every train film with
  ``imdb_id``, ``movie_name``, genre, decade, ``budget``,
  ``effective_rating``, ``y_true``, ``oof_score``, ``residual``,
  predicted-class confidence (classification) or |residual|
  (regression).
* ``phase4_top_correct_<target>.md``: top 20 most-correctly
  predicted films with metadata, formatted for readability.
* ``phase4_top_wrong_<target>.md``: top 20 most-wrongly predicted
  films with metadata. The "where the model fails" gallery.

Also produces a summary figure
``phase4_error_by_genre.png`` showing per-genre mean absolute
residual / classification error rate, complementary to the
stratified AUC heatmap.
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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.targets import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
)
from src.models.phase4.diagnostic import _decade_bucket
from src.models.phase4.figures import latest_per_cell_dir
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)

ALL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


def _load_meta() -> pd.DataFrame:
    """Train-split films with the metadata columns useful for error narration."""
    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df = df[df["imdb_id"].isin(train_ids)].reset_index(drop=True)
    df["decade_bucket"] = df["release_year_parsed"].astype(int).apply(_decade_bucket)
    keep = [
        "imdb_id", "movie_name", "release_year_parsed",
        "primary_genre_bucketed", "decade_bucket",
        "budget", "revenue", "effective_rating",
        "n_scenes", "n_unique_characters", "n_dialogue_lines",
        "data_quality_flag",
    ]
    return df[keep]


def _load_oof_for_winner(target: str) -> pd.DataFrame:
    """Load OOF predictions for the canonical winner cell of ``target``."""
    artifact = paths.DATA_PROCESSED_DIR / f"phase4_primary_model_{target}.joblib"
    bundle = joblib.load(artifact)
    matrix_name = bundle["matrix"]
    family_name = bundle["family"]

    run_dir = latest_per_cell_dir(paths.RUNS_DIR / "phase_4", matrix_name, family_name)
    if run_dir is None:
        raise RuntimeError(
            f"No run directory for matrix={matrix_name} family={family_name}; "
            f"the benchmark may need to be re-run after a finalize."
        )
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)
    pt = m["per_target"]
    if target not in pt:
        raise RuntimeError(f"No {target} OOF data in {run_dir}")
    payload = pt[target]
    df = pd.DataFrame({
        "imdb_id": pt["_imdb_ids"],
        "y_true": payload["y_true"],
        "oof_score": payload["oof_score"],
        "in_sample_score": payload["in_sample_score"],
    })
    if payload.get("oof_hard") is not None:
        df["oof_hard"] = payload["oof_hard"]
    df.attrs["matrix"] = matrix_name
    df.attrs["family"] = family_name
    df.attrs["task"] = payload["task"]
    return df


def _attach_error_columns(
    df: pd.DataFrame, task: str,
) -> pd.DataFrame:
    """Add residual / |residual| / correctness columns."""
    df = df.copy()
    if task == "classification":
        # |error| in probability space
        df["abs_error"] = np.abs(df["y_true"] - df["oof_score"])
        # confidence in the predicted class
        df["pred_class"] = (df["oof_score"] >= 0.5).astype(int)
        df["correct"] = (df["pred_class"] == df["y_true"]).astype(int)
    else:
        df["residual"] = df["y_true"] - df["oof_score"]
        df["abs_error"] = df["residual"].abs()
    return df


def _format_markdown_gallery(
    df: pd.DataFrame, target: str, task: str, kind: str, top_k: int = 20,
) -> str:
    """Pretty markdown table for the top-K most-correct or most-wrong films."""
    head = f"# Phase 4 {kind} predictions for {target}\n\n"
    head += f"Showing top {top_k} films sorted by absolute error "
    head += f"(ascending for 'most correct', descending for 'most wrong').\n\n"

    cols = [
        "movie_name", "release_year_parsed", "primary_genre_bucketed",
        "y_true", "oof_score",
    ]
    if task == "classification":
        cols += ["pred_class", "correct", "abs_error"]
    else:
        cols += ["residual", "abs_error"]
    cols += ["budget", "revenue", "effective_rating"]

    return head + df[cols].head(top_k).to_markdown(index=False, floatfmt=".3f") + "\n"


def per_genre_error_summary(
    df: pd.DataFrame, task: str,
) -> pd.DataFrame:
    """Per-genre mean absolute error + sample size."""
    if task == "classification":
        grouped = df.groupby("primary_genre_bucketed").agg(
            n=("imdb_id", "count"),
            n_pos=("y_true", "sum"),
            mean_abs_error=("abs_error", "mean"),
            accuracy=("correct", "mean"),
        )
    else:
        grouped = df.groupby("primary_genre_bucketed").agg(
            n=("imdb_id", "count"),
            mean_residual=("residual", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
    return grouped.sort_values("n", ascending=False)


def plot_per_genre_error(
    summaries: dict[str, pd.DataFrame],
    out_path: Path,
) -> Path:
    """Bar chart of per-genre mean absolute error per target."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 5))
    if len(summaries) == 1:
        axes = [axes]
    for ax, (target, summary) in zip(axes, summaries.items()):
        s = summary.copy()
        # Drop genres with fewer than 5 films for stability.
        s = s[s["n"] >= 5]
        s = s.sort_values("mean_abs_error", ascending=True)
        ax.barh(s.index.astype(str), s["mean_abs_error"], color="indianred")
        for i, (idx, row) in enumerate(s.iterrows()):
            ax.text(
                row["mean_abs_error"] + 0.005, i,
                f" n={int(row['n'])}",
                va="center", fontsize=8,
            )
        ax.set_title(target)
        ax.set_xlabel("mean |y_true - oof_score|")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Phase 4 per-genre absolute error on the headline winner",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def analyze_target(target: str, meta: pd.DataFrame, top_k: int = 20) -> dict:
    """End-to-end analysis for one target."""
    oof = _load_oof_for_winner(target)
    task = oof.attrs["task"]
    matrix_name = oof.attrs["matrix"]
    family_name = oof.attrs["family"]

    merged = oof.merge(meta, on="imdb_id", how="inner")
    merged = _attach_error_columns(merged, task)

    by_correct = merged.sort_values("abs_error", ascending=True)
    by_wrong = merged.sort_values("abs_error", ascending=False)

    out_csv = paths.REPORTS_TABLES_DIR / f"phase4_predictions_{target}.csv"
    merged.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(merged))

    out_correct = paths.REPORTS_TABLES_DIR / f"phase4_top_correct_{target}.md"
    out_correct.write_text(_format_markdown_gallery(
        by_correct, target, task, kind="most-correct", top_k=top_k,
    ))
    logger.info("Wrote %s", out_correct)

    out_wrong = paths.REPORTS_TABLES_DIR / f"phase4_top_wrong_{target}.md"
    out_wrong.write_text(_format_markdown_gallery(
        by_wrong, target, task, kind="most-wrong", top_k=top_k,
    ))
    logger.info("Wrote %s", out_wrong)

    summary = per_genre_error_summary(merged, task)
    out_summary = paths.REPORTS_TABLES_DIR / f"phase4_error_by_genre_{target}.csv"
    summary.to_csv(out_summary)
    logger.info("Wrote %s (%d rows)", out_summary, len(summary))

    return {
        "target": target,
        "matrix": matrix_name,
        "family": family_name,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    set_log_level(args.log_level)
    paths.ensure_dirs()

    meta = _load_meta()

    summaries: dict[str, pd.DataFrame] = {}
    for target in ALL_TARGETS:
        try:
            result = analyze_target(target, meta, top_k=args.top_k)
        except RuntimeError as exc:
            logger.warning("Skip %s: %s", target, exc)
            continue
        summaries[target] = result["summary"]

    plot_per_genre_error(
        summaries,
        paths.REPORTS_FIGURES_DIR / "phase4_error_by_genre.png",
    )


if __name__ == "__main__":
    main()
