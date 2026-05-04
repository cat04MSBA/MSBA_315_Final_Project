"""v1 vs v2 headline comparison report.

Reads:
* ``reports/tables/phase4_benchmark.csv`` and ``phase4_benchmark_mpnet.csv``
  (v1 primary-tier benchmark across both encoder variants)
* ``reports/tables/phase4_benchmark_v2.csv`` and
  ``phase4_benchmark_mpnet_v2.csv`` (v2 same)

Writes:
* ``reports/tables/phase4_v1_vs_v2_comparison.csv``: per-target side-by-side
  best primary model on each (matrix, encoder), with v1 → v2 delta.
* Console headline summary.

Headline metric:
* ``roi_gt_2`` AUC-ROC (the project's headline classification target)
* ``roi_gt_1`` AUC-ROC
* ``log_roi`` RMSE (lower is better)
"""

from __future__ import annotations

# Allow running by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import pandas as pd

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


CLASSIFICATION_METRIC = "auc_roc"
REGRESSION_METRIC = "rmse"


def _best_per_target_matrix(
    df: pd.DataFrame, eval_set: str = "oof_global",
) -> pd.DataFrame:
    """Pick the best primary-tier (matrix, family) cell per target on the headline metric."""
    if df.empty:
        return df
    primary = df[df["tier"] == "primary"]
    rows: list[dict] = []
    for target in primary["target"].unique():
        sub = primary[(primary["target"] == target) & (primary["eval_set"] == eval_set)]
        task = sub["task"].iloc[0]
        metric = CLASSIFICATION_METRIC if task == "classification" else REGRESSION_METRIC
        sub_m = sub[sub["metric"] == metric].copy()
        if sub_m.empty:
            continue
        ascending = (task == "regression")  # for RMSE, lower is better
        sub_m = sub_m.sort_values("value", ascending=ascending)
        for matrix in sub_m["matrix"].unique():
            best = sub_m[sub_m["matrix"] == matrix].iloc[0]
            rows.append({
                "target": target,
                "task": task,
                "matrix": matrix,
                "metric": metric,
                "best_family": best["family"],
                "value": best["value"],
                "ci_lo": best["ci_lo"],
                "ci_hi": best["ci_hi"],
            })
    return pd.DataFrame(rows)


def _load_corpus_size(processed_dir: Path, parquet_name: str) -> int:
    p = processed_dir / parquet_name
    if not p.is_file():
        return -1
    return len(pd.read_parquet(p))


def main() -> None:
    paths.ensure_dirs()
    tables = paths.REPORTS_TABLES_DIR

    summaries: list[pd.DataFrame] = []
    for v_label, csv_minilm, csv_mpnet in [
        ("v1", "phase4_benchmark.csv", "phase4_benchmark_mpnet.csv"),
        ("v2", "phase4_benchmark_v2.csv", "phase4_benchmark_mpnet_v2.csv"),
    ]:
        for encoder, csv_path in [("minilm", tables / csv_minilm),
                                   ("mpnet", tables / csv_mpnet)]:
            if not csv_path.is_file():
                logger.warning("missing: %s — skipping", csv_path)
                continue
            df = pd.read_csv(csv_path)
            best = _best_per_target_matrix(df)
            best["corpus"] = v_label
            best["encoder"] = encoder
            summaries.append(best)

    if not summaries:
        logger.error("No benchmark CSVs found; run benchmarks first")
        return
    full = pd.concat(summaries, ignore_index=True)

    # Pivot: one row per (target, matrix, encoder), columns for v1 vs v2.
    pivot = (
        full.pivot_table(
            index=["target", "task", "encoder", "matrix", "metric"],
            columns="corpus",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    if "v1" in pivot.columns and "v2" in pivot.columns:
        pivot["delta_v2_minus_v1"] = pivot["v2"] - pivot["v1"]
        pivot["pct_change"] = 100.0 * pivot["delta_v2_minus_v1"] / pivot["v1"]
    pivot = pivot.sort_values(["target", "encoder", "matrix"]).reset_index(drop=True)

    out = tables / "phase4_v1_vs_v2_comparison.csv"
    pivot.to_csv(out, index=False)

    # Corpus sizes + headline.
    sz_v1 = _load_corpus_size(paths.DATA_PROCESSED_DIR, "films_joined.parquet")
    sz_v2 = _load_corpus_size(paths.DATA_PROCESSED_DIR / "v2", "films_joined_v2.parquet")
    print("\n=== v1 vs v2 headline comparison ===")
    print(f"Corpus size: v1={sz_v1}, v2={sz_v2} "
          f"(Δ +{sz_v2 - sz_v1}, +{100*(sz_v2-sz_v1)/sz_v1:.1f}%)" if sz_v1 > 0 else "")
    print()
    if "v1" in pivot.columns and "v2" in pivot.columns:
        # Headline: best per target across all (encoder, matrix) cells
        best_per_target_v1 = (
            full[full["corpus"] == "v1"]
            .sort_values("value", ascending=full[full["corpus"] == "v1"]["task"].iloc[0] == "regression")
            .groupby("target", as_index=False)
        )
        # Simpler: print the comparison dataframe
        cols = ["target", "encoder", "matrix", "metric", "v1", "v2", "delta_v2_minus_v1", "pct_change"]
        print(pivot[cols].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
