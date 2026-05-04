"""Phase 4 Tier-A1 step: stacking ensemble of the four primary families.

The Phase 4 paired Bayesian comparison declared no statistical
winner among (linear, histgb, random_forest, svm_rbf) on
``roi_gt_2`` AUC. They sit within 0.04 AUC of each other but make
different errors (different inductive biases on borderline films).
A standard remedy is stacking: train a meta-learner on the OOF
predictions of the four base models. The meta-learner combines them
into a single score that often beats any one individual.

Implementation. The base models' OOF predictions are already saved
in ``runs/phase_4/<timestamp>_<matrix>_<family>/metrics.json`` under
``per_target.<target>.oof_score``. Each is a 1199-length vector
indexed by the train-split imdb_ids; the i-th entry is the
out-of-fold prediction averaged across the three repeated-CV
repetitions. Stacking trains a Ridge / Logistic-L2 meta-learner
where the inputs are these four columns of OOF predictions and the
output is the target.

Methodologically the meta-learner needs its own cross-validation
to avoid optimism. The OOF inputs are themselves out-of-fold (not
trained on the same row they are predicting), so a simple 5-fold CV
on (OOF_features) -> y is honest. The meta-learner's CV folds are
seeded to match the base models' outer-CV scheme so the comparison
is paired by row.

Outputs:
    reports/tables/phase4_stacking.csv
    reports/figures/phase4_stacking.png  (lift over best base per cell)
    runs/phase_4/<timestamp>_stacking_<matrix>/  (per-cell save_run)
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold

from src.experiments.save_run import save_run
from src.models.phase4.benchmark import task_for_target
from src.models.phase4.figures import (
    ALL_TARGETS,
    PRIMARY_FAMILY_NAMES,
    latest_per_cell_dir,
)
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)

SEED: int = 42
N_FOLDS: int = 5
N_REPEATS: int = 3


def _load_oof(run_dir: Path, target: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)
    pt = m.get("per_target", {})
    if target not in pt:
        return None
    return (
        np.asarray(pt[target]["oof_score"], dtype=float),
        np.asarray(pt[target]["y_true"], dtype=float),
        list(pt["_imdb_ids"]),
    )


def _scale_score(score: np.ndarray, family: str) -> np.ndarray:
    """SVM emits decision-function scores; logistic-rescale them.

    The meta-learner is rank-invariant under monotonic transforms,
    but for log-loss reporting we need probability-style inputs.
    Logistic rescale is standard and harmless for AUC.
    """
    if family == "svm_rbf":
        return 1.0 / (1.0 + np.exp(-score))
    return score


def collect_meta_features(
    matrix: str, target: str,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Build (n_train x 4) meta-feature matrix from the four base families."""
    phase_dir = paths.RUNS_DIR / "phase_4"
    pieces: dict[str, np.ndarray] = {}
    y_ref: np.ndarray | None = None
    ids_ref: list[str] | None = None

    for fam in PRIMARY_FAMILY_NAMES:
        run_dir = latest_per_cell_dir(phase_dir, matrix, fam)
        if run_dir is None:
            logger.warning("Skip stacking for matrix=%s target=%s: missing %s",
                           matrix, target, fam)
            return None
        loaded = _load_oof(run_dir, target)
        if loaded is None:
            return None
        score, y_true, ids = loaded
        if y_ref is None:
            y_ref = y_true
            ids_ref = ids
        pieces[fam] = _scale_score(score, fam)

    X = pd.DataFrame(pieces, index=ids_ref)
    return X, y_ref


def _classification_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    hard = (score >= 0.5).astype(int)
    out = {
        "auc_roc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "f1": float(f1_score(y_true, hard)),
        "log_loss": float(log_loss(y_true, np.clip(score, 1e-7, 1 - 1e-7))),
    }
    return out


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    abs_mean = float(np.mean(np.abs(y_true)))
    return {
        "mse": float(mean_squared_error(y_true, pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, pred)),
        "cvrmse": rmse / abs_mean if abs_mean > 0 else float("nan"),
    }


def stack_one_cell(
    matrix: str, target: str,
) -> dict | None:
    """Train + evaluate the meta-learner for one (matrix, target) cell."""
    collected = collect_meta_features(matrix, target)
    if collected is None:
        return None
    X, y = collected
    task = task_for_target(target)

    if task == "classification":
        cv = RepeatedStratifiedKFold(
            n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED,
        )
        meta_factory = lambda: LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000, random_state=SEED,
        )
    else:
        cv = RepeatedKFold(
            n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED,
        )
        meta_factory = lambda: Ridge(alpha=1.0, random_state=SEED)

    n = len(y)
    score_accum = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    per_fold: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X.values, y):
        meta = meta_factory()
        meta.fit(X.values[train_idx], y[train_idx])
        if task == "classification":
            pred = meta.predict_proba(X.values[test_idx])[:, 1]
        else:
            pred = meta.predict(X.values[test_idx])
        score_accum[test_idx] += pred
        counts[test_idx] += 1
        if task == "classification":
            per_fold.append(_classification_metrics(y[test_idx], pred))
        else:
            per_fold.append(_regression_metrics(y[test_idx], pred))

    safe_counts = np.where(counts == 0, 1, counts)
    oof_score = score_accum / safe_counts

    if task == "classification":
        global_metrics = _classification_metrics(y, oof_score)
    else:
        global_metrics = _regression_metrics(y, oof_score)

    # Compute per-fold mean / std.
    per_fold_df = pd.DataFrame(per_fold)
    fold_summary = {
        f"{m}_mean": float(per_fold_df[m].mean()) for m in per_fold_df.columns
    }
    fold_summary.update({
        f"{m}_std": float(per_fold_df[m].std(ddof=1)) for m in per_fold_df.columns
    })

    # Train final meta on all OOF (this is the model used at inference).
    meta_final = meta_factory()
    meta_final.fit(X.values, y)
    coefs = (
        meta_final.coef_.flatten().tolist() if hasattr(meta_final, "coef_") else []
    )
    intercept = (
        float(meta_final.intercept_) if np.isscalar(meta_final.intercept_)
        else float(np.atleast_1d(meta_final.intercept_)[0])
    )

    return {
        "matrix": matrix,
        "target": target,
        "task": task,
        "oof_metrics_global": global_metrics,
        "per_fold_metrics": {m: per_fold_df[m].tolist() for m in per_fold_df.columns},
        "per_fold_summary": fold_summary,
        "meta_coefs": dict(zip(PRIMARY_FAMILY_NAMES, coefs)),
        "meta_intercept": intercept,
        "n_train": int(n),
        "oof_score": oof_score.tolist(),
        "y_true": y.tolist(),
        "imdb_ids": X.index.tolist(),
    }


def find_best_base_metric(matrix: str, target: str, metric: str) -> float:
    """Lookup the best base-family OOF value on this metric for comparison.

    Looks across both the canonical primary benchmark CSV and the mpnet
    follow-up CSV so that mpnet matrices have a comparable best-base.
    """
    bench_paths = [
        paths.REPORTS_TABLES_DIR / "phase4_benchmark.csv",
        paths.REPORTS_TABLES_DIR / "phase4_benchmark_mpnet.csv",
    ]
    frames = []
    for p in bench_paths:
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        return float("nan")
    bench = pd.concat(frames, ignore_index=True)
    sub = bench[
        (bench["matrix"] == matrix)
        & (bench["target"] == target)
        & (bench["metric"] == metric)
        & (bench["eval_set"] == "oof_global")
        & (bench["tier"] == "primary")
    ]
    if sub.empty:
        return float("nan")
    if metric in ("auc_roc", "pr_auc", "f1"):
        return float(sub["value"].max())
    return float(sub["value"].min())


def run_stacking_all() -> pd.DataFrame:
    """Stack across all available matrices and all three targets.

    Iterates over every matrix that has cells for all four primary
    families on disk (so mpnet matrices are picked up automatically
    once the mpnet benchmark has run).
    """
    results: list[dict] = []
    rows: list[dict] = []
    matrices = (
        "all_five", "standalone_positive_union",
        "all_five_mpnet", "standalone_positive_union_mpnet",
    )
    for matrix in matrices:
        for target in ALL_TARGETS:
            r = stack_one_cell(matrix, target)
            if r is None:
                continue
            results.append(r)
            primary_metric = "auc_roc" if r["task"] == "classification" else "rmse"
            best_base = find_best_base_metric(matrix, target, primary_metric)
            stack_value = r["oof_metrics_global"][primary_metric]
            lift = (
                stack_value - best_base if primary_metric != "rmse"
                else best_base - stack_value
            )
            logger.info(
                "Stacking %s %s: %s = %.4f (best base %.4f, lift %+.4f)",
                matrix, target, primary_metric, stack_value, best_base, lift,
            )
            for metric, value in r["oof_metrics_global"].items():
                rows.append({
                    "matrix": matrix,
                    "target": target,
                    "task": r["task"],
                    "metric": metric,
                    "stack_value": value,
                    "best_base_value": (
                        find_best_base_metric(matrix, target, metric)
                        if metric in ("auc_roc", "pr_auc", "f1", "rmse", "mse", "mae", "cvrmse", "log_loss")
                        else float("nan")
                    ),
                    "n_train": r["n_train"],
                    "meta_coefs": str(r["meta_coefs"]),
                    "meta_intercept": r["meta_intercept"],
                })

    df = pd.DataFrame(rows)
    out = paths.REPORTS_TABLES_DIR / "phase4_stacking.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))

    # Persist a save_run per (matrix, target) for the audit trail.
    for r in results:
        with save_run(
            phase="phase_4",
            name=f"stacking_{r['matrix']}_{r['target']}",
            params={
                "meta_learner": "Logistic" if r["task"] == "classification" else "Ridge",
                "base_families": list(PRIMARY_FAMILY_NAMES),
                "n_folds": N_FOLDS,
                "n_repeats": N_REPEATS,
                "task": r["task"],
            },
            preprocessing={"meta_input": "OOF predictions of base families"},
            features=list(PRIMARY_FAMILY_NAMES),
        ) as run:
            run.record_metrics({
                "matrix": r["matrix"],
                "target": r["target"],
                "oof_metrics_global": r["oof_metrics_global"],
                "per_fold_summary": r["per_fold_summary"],
                "meta_coefs": r["meta_coefs"],
                "meta_intercept": r["meta_intercept"],
            })

    return df


def plot_stacking_lift(out_path: Path) -> Path:
    """Bar chart of stacking lift per (matrix, target) on the headline metric."""
    df = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase4_stacking.csv")
    headline = df[
        ((df["task"] == "classification") & (df["metric"] == "auc_roc"))
        | ((df["task"] == "regression") & (df["metric"] == "rmse"))
    ].copy()
    headline["lift"] = headline.apply(
        lambda r: (r["stack_value"] - r["best_base_value"])
        if r["metric"] != "rmse" else (r["best_base_value"] - r["stack_value"]),
        axis=1,
    )

    matrices = sorted(headline["matrix"].unique())
    targets = list(ALL_TARGETS)
    width = 0.35
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, matrix in zip([-width / 2, width / 2], matrices):
        vals = []
        for t in targets:
            row = headline[(headline["matrix"] == matrix) & (headline["target"] == t)]
            vals.append(float(row["lift"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, vals, width=width, label=matrix)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({'AUC' if t.startswith('roi_gt') else 'RMSE'})" for t in targets])
    ax.set_ylabel("Stacking lift over best base family\n(positive = stacking better)")
    ax.set_title("Phase 4 stacking lift per (matrix, target)")
    ax.legend(title="matrix", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
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
    run_stacking_all()
    plot_stacking_lift(paths.REPORTS_FIGURES_DIR / "phase4_stacking_lift.png")


if __name__ == "__main__":
    main()
