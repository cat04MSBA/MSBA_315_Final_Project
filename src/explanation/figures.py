"""Phase 7 figure generation.

Reads the saved tables and produces the four pre-registered figures:

* ``phase7_global_shap.png`` — top-20 SHAP ranking for both
  supported targets, side-by-side.
* ``phase7_shap_vs_native.png`` — scatter of SHAP rank vs Phase 4
  native importance rank.
* ``phase7_per_film_examples.png`` — per-film waterfall-style top
  contributors for 4 representative films.
* ``phase7_scene_level_example.png`` — per-scene contribution bar
  chart for one example film.
"""

from __future__ import annotations

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

from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def plot_global_shap(out_path: Path, top_k: int = 20) -> Path:
    """Side-by-side top-K global SHAP ranking for the two supported targets."""
    targets = []
    for name in ("roi_gt_2", "log_roi"):
        p = paths.REPORTS_TABLES_DIR / f"phase7_global_shap_{name}.csv"
        if p.is_file():
            targets.append((name, pd.read_csv(p)))
    if not targets:
        return out_path

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 8), sharex=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, targets):
        top = df.head(top_k).iloc[::-1]
        ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")
        ax.set_title(f"{name}\n(top {top_k} by mean |SHAP|)")
        ax.set_xlabel("mean |SHAP|")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Phase 7 global SHAP feature ranking", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_shap_vs_native(out_path: Path) -> Path:
    """Scatter of SHAP rank vs native importance rank with the identity line."""
    p = paths.REPORTS_TABLES_DIR / "phase7_shap_vs_native_importance.csv"
    if not p.is_file():
        return out_path
    df = pd.read_csv(p).dropna(subset=["shap_rank", "native_rank"])
    if df.empty:
        return out_path
    from scipy.stats import spearmanr
    rho, _ = spearmanr(df["shap_rank"], df["native_rank"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["native_rank"], df["shap_rank"], alpha=0.6, color="tab:blue")
    n_ranks = max(df["native_rank"].max(), df["shap_rank"].max())
    ax.plot([1, n_ranks], [1, n_ranks], "k--", linewidth=0.8, label="identity")
    # Annotate top-10 by SHAP rank
    top = df.nsmallest(10, "shap_rank")
    for _, row in top.iterrows():
        ax.annotate(
            row["feature"],
            (row["native_rank"], row["shap_rank"]),
            fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points",
        )
    ax.set_xlabel("Phase 4 native importance rank")
    ax.set_ylabel("Phase 7 SHAP rank (mean |SHAP|)")
    ax.set_title(
        f"Phase 7: SHAP global ranking vs Phase 4 native importance\n"
        f"(roi_gt_2 winner XGBoost; Spearman ρ = {rho:.3f})",
        fontsize=11,
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_per_film_examples(out_path: Path, n_examples: int = 4) -> Path:
    """Top SHAP contributors for a few example films from the per-film table."""
    p = paths.REPORTS_TABLES_DIR / "phase7_per_film_rationale.csv"
    if not p.is_file():
        return out_path
    df = pd.read_csv(p)
    if "recommended_action" not in df.columns:
        return out_path

    # Pick one film per Phase 6 action plus a misclassified film
    examples: list[pd.Series] = []
    for action in ("Greenlight", "Pass", "Refer"):
        sub = df[df["recommended_action"] == action]
        if not sub.empty:
            examples.append(sub.iloc[0])
    # And a high-uncertainty Refer film
    if "calibrated_probability" in df.columns:
        df["dist_from_half"] = (df["calibrated_probability"] - 0.5).abs()
        uncertain = df[df["recommended_action"] == "Refer"].nsmallest(1, "dist_from_half")
        if not uncertain.empty:
            examples.append(uncertain.iloc[0])
    examples = examples[:n_examples]

    if not examples:
        return out_path

    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 6))
    if len(examples) == 1:
        axes = [axes]
    for ax, row in zip(axes, examples):
        # Parse top_pos and top_neg back into bar lists
        pos = _parse_top_features(row.get("top_pos_features", ""))
        neg = _parse_top_features(row.get("top_neg_features", ""))
        labels = [n for n, _ in pos][::-1] + [n for n, _ in neg]
        values = [v for _, v in pos][::-1] + [v for _, v in neg]
        colors = ["tab:green"] * len(pos) + ["tab:red"] * len(neg)
        if not labels:
            ax.text(0.5, 0.5, "no features", ha="center", va="center")
            continue
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("SHAP contribution (log-odds)")
        title = (
            f"{row['imdb_id']} ({row.get('genre', 'NA')})\n"
            f"{row.get('recommended_action', 'NA')} | "
            f"P={row.get('calibrated_probability', float('nan')):.3f}"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle("Phase 7 per-film SHAP contributors (4 example films)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def _parse_top_features(s: str) -> list[tuple[str, float]]:
    """Parse 'feature_name=+0.123; feature_name=-0.045' style strings."""
    if not isinstance(s, str) or not s:
        return []
    items = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        feat, val = chunk.rsplit("=", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        items.append((feat.strip(), v))
    return items


def plot_scene_level_example(out_path: Path, top_k_each: int = 5) -> Path:
    """Per-scene contribution bar chart for the first example film with >5 scenes."""
    json_path = paths.DATA_PROCESSED_DIR / "phase7_scene_level_examples.json"
    if not json_path.is_file():
        return out_path
    with open(json_path) as f:
        examples = json.load(f)
    # Pick the first film with >= 5 scenes for a meaningful chart
    chosen = None
    for ex in examples:
        if ex["n_scenes"] >= 5:
            chosen = ex
            break
    if chosen is None:
        return out_path

    pos = chosen["top_positive_scenes"]
    neg = chosen["top_negative_scenes"]
    labels = [
        f"Scene {s['scene_index']+1}: {s['heading'][:40]}"
        for s in pos[::-1]
    ] + [
        f"Scene {s['scene_index']+1}: {s['heading'][:40]}"
        for s in neg
    ]
    values = [s["contribution"] for s in pos[::-1]] + [s["contribution"] for s in neg]
    colors = ["tab:green"] * len(pos) + ["tab:red"] * len(neg)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(
        "Scene contribution to predicted P(roi_gt_2 hit) "
        "(positive = scene pushes hit-probability up)",
        fontsize=9,
    )
    ax.set_title(
        f"Phase 7 scene-level attribution example: {chosen['imdb_id']}\n"
        f"(P={chosen['calibrated_probability']:.3f}; "
        f"action={chosen['recommended_action']}; "
        f"n_scenes={chosen['n_scenes']})",
        fontsize=10,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    set_log_level(args.log_level)
    paths.ensure_dirs()
    plot_global_shap(paths.REPORTS_FIGURES_DIR / "phase7_global_shap.png")
    plot_shap_vs_native(paths.REPORTS_FIGURES_DIR / "phase7_shap_vs_native.png")
    plot_per_film_examples(paths.REPORTS_FIGURES_DIR / "phase7_per_film_examples.png")
    plot_scene_level_example(paths.REPORTS_FIGURES_DIR / "phase7_scene_level_example.png")


if __name__ == "__main__":
    main()
