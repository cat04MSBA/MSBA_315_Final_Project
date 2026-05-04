"""Phase 4 figure generation.

Reads from the per-cell ``metrics.json`` files written by
:mod:`src.models.phase4.benchmark` and produces the two pre-registered
Phase 4 figures:

* ``phase4_train_oof_gap.png``: train-versus-OOF gap per family per
  target on both input matrices. Bar chart that surfaces over- and
  underfit at a glance. The brief's reason for adding this is the
  Phase 3 finding that HistGB has a 0.20-0.27 train-OOF AUC gap on
  classification at conservative defaults; the figure makes the
  family-by-family comparison legible.

* ``phase4_calibration_pre.png``: OOF reliability diagrams for the
  primary-tier classifiers on the headline target ``roi_gt_2``.
  Bridges to Phase 5 by showing in advance which candidates produce
  well-calibrated probability outputs and which need Platt scaling
  (SVM-RBF in particular emits decision-function scores rather than
  probabilities).

Both figures are pre-registered as Phase 4 deliverables in
``docs/proposals/phase4_preregistration.md`` Section 11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.models.phase4.families import FAMILIES, primary_tier
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


PRIMARY_FAMILY_NAMES: list[str] = [f.name for f in primary_tier()]
ALL_TARGETS: tuple[str, ...] = ("log_roi", "roi_gt_1", "roi_gt_2")
HEADLINE_TARGET: str = "roi_gt_2"


@dataclass
class CellMetrics:
    """One cell's per-target metrics loaded from the run directory."""
    matrix: str
    family: str
    per_target: dict[str, dict]


def _load_run_metrics(run_dir: Path) -> CellMetrics:
    """Load one ``metrics.json`` and tag it with matrix + family from params."""
    with open(run_dir / "params.json") as f:
        params = json.load(f)
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)
    per_target = {
        k: v for k, v in m.get("per_target", {}).items()
        if not k.startswith("_")
    }
    return CellMetrics(
        matrix=params["matrix"],
        family=params["family"],
        per_target=per_target,
    )


def latest_per_cell_dir(
    phase_dir: Path,
    matrix: str,
    family: str,
) -> Path | None:
    """Return the most recent run directory matching (matrix, family).

    Run-directory names follow the convention
    ``YYYYMMDD_HHMM_<matrix>_<family>``; lexicographic sort is
    chronological. Smoke-test directories are excluded.
    """
    matches: list[Path] = []
    for run_dir in sorted(phase_dir.glob(f"*_{matrix}_{family}")):
        if not (run_dir / "metrics.json").is_file():
            continue
        with open(run_dir / "params.json") as f:
            params = json.load(f)
        if params.get("mode") == "smoke":
            continue
        if params.get("matrix") != matrix or params.get("family") != family:
            continue
        matches.append(run_dir)
    return matches[-1] if matches else None


def load_all_cells(phase_dir: Path | None = None) -> list[CellMetrics]:
    """Load every Phase 4 benchmark cell from ``runs/phase_4/*/`` directories.

    Skips smoke-test runs, sensitivity runs, and stacking runs. Only
    standard (matrix x family) primary/secondary benchmark cells are
    loaded; identified by the presence of both ``matrix`` and
    ``family`` keys in ``params.json``.
    """
    phase_dir = phase_dir or paths.RUNS_DIR / "phase_4"
    cells: list[CellMetrics] = []
    for run_dir in sorted(phase_dir.glob("*/")):
        params_file = run_dir / "params.json"
        metrics_file = run_dir / "metrics.json"
        if not (params_file.is_file() and metrics_file.is_file()):
            continue
        with open(params_file) as f:
            params = json.load(f)
        if params.get("mode") in {"smoke", "sensitivity"}:
            continue
        # Skip auxiliary runs (stacking, sensitivity) that don't carry
        # the benchmark-cell shape.
        if "matrix" not in params or "family" not in params:
            continue
        cells.append(_load_run_metrics(run_dir))
    logger.info("Loaded %d Phase 4 benchmark cells", len(cells))
    return cells


# ---------------------------------------------------------------------------
# Figure: train-vs-OOF gap
# ---------------------------------------------------------------------------


def _select_gap_metric(target: str) -> str:
    """Select the metric whose train-OOF gap is plotted for a target."""
    return "rmse" if target == "log_roi" else "auc_roc"


def plot_train_oof_gap(
    cells: list[CellMetrics],
    out_path: Path,
    families: list[str] | None = None,
) -> Path:
    """Render the ``phase4_train_oof_gap.png`` figure.

    Three subplots (one per target). Each subplot has grouped bars
    showing the |train - OOF| gap per family, two bars per family (one
    per matrix). For regression the metric is RMSE; for classification
    AUC-ROC. Smaller gap = better generalization.
    """
    fams = families or PRIMARY_FAMILY_NAMES
    matrices = sorted({c.matrix for c in cells})

    fig, axes = plt.subplots(1, len(ALL_TARGETS), figsize=(15, 4.5), sharey=False)
    if len(ALL_TARGETS) == 1:
        axes = [axes]

    n_matrices = len(matrices)
    bar_width = 0.8 / n_matrices
    offsets = [(-0.4) + bar_width * (i + 0.5) for i in range(n_matrices)]

    for ax, target in zip(axes, ALL_TARGETS):
        metric = _select_gap_metric(target)
        x = np.arange(len(fams))
        for offset, matrix in zip(offsets, matrices):
            values: list[float] = []
            for fam in fams:
                cell = next(
                    (c for c in cells if c.matrix == matrix and c.family == fam),
                    None,
                )
                if cell is None or target not in cell.per_target:
                    values.append(np.nan)
                    continue
                pt = cell.per_target[target]
                tr = pt["in_sample_metrics"].get(metric, np.nan)
                oof = pt["oof_metrics_global"].get(metric, np.nan)
                if metric == "auc_roc":
                    gap = float(tr - oof)
                else:
                    # Lower-is-better metric: positive gap means OOF is worse than train.
                    gap = float(oof - tr)
                values.append(gap)
            ax.bar(x + offset, values, width=bar_width, label=matrix)

        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{target} ({metric})")
        ax.set_ylabel(f"train - OOF" if metric == "auc_roc" else "OOF - train")
        if target == ALL_TARGETS[-1]:
            ax.legend(title="matrix", fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Phase 4 train-vs-OOF gap per family per matrix "
        "(positive = train better than OOF, the overfit direction)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure: pre-Phase-5 calibration diagrams
# ---------------------------------------------------------------------------


def _normalize_score(score: np.ndarray, score_method: str) -> np.ndarray:
    """Map decision-function scores to [0, 1] via the logistic for plotting.

    Reliability diagrams need a probability-style x-axis. The SVM-RBF
    family emits raw signed margins; we apply ``1 / (1 + exp(-z))`` to
    map them to [0, 1]. The transform is monotonic so the binning order
    of the calibration curve is unaffected; the resulting figure shows
    that the SVM curve sits far from the diagonal even after the
    sigmoid rescale, which is the Phase 5 motivation.
    """
    if score_method == "predict_proba":
        return score
    return 1.0 / (1.0 + np.exp(-score))


def plot_calibration_pre(
    cells: list[CellMetrics],
    out_path: Path,
    target: str = HEADLINE_TARGET,
    matrix: str = "all_five",
    n_bins: int = 10,
) -> Path:
    """Render OOF reliability diagrams for primary-tier classifiers.

    One panel per family on the chosen target and matrix. Each panel
    shows fraction-of-positives vs predicted probability with the
    diagonal reference. The plot is the bridge to Phase 5: families
    with curves close to the diagonal need little calibration, families
    with curves far from the diagonal will need Platt scaling or an
    equivalent post-hoc step.
    """
    primary = [c for c in cells if c.matrix == matrix]
    fig, axes = plt.subplots(1, len(PRIMARY_FAMILY_NAMES), figsize=(16, 4.2), sharey=True)

    for ax, fam in zip(axes, PRIMARY_FAMILY_NAMES):
        cell = next((c for c in primary if c.family == fam), None)
        if cell is None or target not in cell.per_target:
            ax.set_title(f"{fam}\n(no data)")
            ax.set_axis_off()
            continue
        pt = cell.per_target[target]
        score = np.asarray(pt["oof_score"], dtype=float)
        y_true = np.asarray(pt["y_true"], dtype=int)
        spec = FAMILIES[fam]
        prob = _normalize_score(score, spec.score_method)

        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.3, label=fam)
        ax.set_title(f"{fam}\n({spec.score_method})")
        ax.set_xlabel("predicted probability")
        if ax is axes[0]:
            ax.set_ylabel("observed fraction positive")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Phase 4 OOF calibration on {target} ({matrix}) — pre-Phase-5 reliability",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure: per-matrix winner summary table (markdown helper, not a plot)
# ---------------------------------------------------------------------------


def winners_summary_table(cells: list[CellMetrics]) -> pd.DataFrame:
    """Build the per-target winner summary as a small DataFrame.

    Selection: AUC-ROC max for classification, RMSE min for regression.
    Includes both matrices' best per family + the overall winner.
    """
    rows = []
    for target in ALL_TARGETS:
        metric = _select_gap_metric(target)
        for cell in cells:
            if target not in cell.per_target:
                continue
            pt = cell.per_target[target]
            value = pt["oof_metrics_global"].get(metric, np.nan)
            rows.append({
                "target": target,
                "matrix": cell.matrix,
                "family": cell.family,
                "metric": metric,
                "oof_value": value,
                "in_sample_value": pt["in_sample_metrics"].get(metric, np.nan),
                "best_params": pt["best_params"],
            })
    return pd.DataFrame(rows)
