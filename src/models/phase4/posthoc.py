"""Post-hoc Phase 4 utilities: paired tests + figures recomputed from disk.

The benchmark orchestrator writes everything needed to recompute paired
Bayesian comparisons and figures into per-cell ``metrics.json`` files
under ``runs/phase_4/<timestamp>_<cell>/``. This module reads those
files and produces:

* ``reports/tables/phase4_paired_tests.csv`` (overwriting the
  benchmark's in-process version, which is necessary if the paired-test
  code was patched mid-run).
* ``reports/figures/phase4_train_oof_gap.png``.
* ``reports/figures/phase4_calibration_pre.png``.

CLI::

    python -m src.models.phase4.posthoc

Idempotent. Safe to re-run after every benchmark invocation; pulls the
latest run per (matrix, family) by directory mtime.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.phase4.benchmark import (
    CLASSIFICATION_TEST_METRICS,
    REGRESSION_TEST_METRICS,
    comparisons_to_rows,
    task_for_target,
)
from src.models.phase4.figures import (
    ALL_TARGETS,
    CellMetrics,
    PRIMARY_FAMILY_NAMES,
    load_all_cells,
    plot_calibration_pre,
    plot_train_oof_gap,
)
from src.models.phase4.paired_test import all_pairwise_comparisons
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


def latest_per_cell(cells: list[CellMetrics]) -> list[CellMetrics]:
    """If multiple runs exist for the same (matrix, family), keep the latest.

    The benchmark may be relaunched; older run directories should not
    contaminate the post-hoc summary. We use the file order (already
    sorted by name = timestamp) and overwrite earlier entries.
    """
    seen: dict[tuple[str, str], CellMetrics] = {}
    for c in cells:
        seen[(c.matrix, c.family)] = c  # later entries overwrite earlier
    return list(seen.values())


def recompute_paired_tests(cells: list[CellMetrics]) -> pd.DataFrame:
    """Run baycomp paired comparisons across primary-tier families per cell.

    The per-fold metric arrays come from ``metrics.json``'s
    ``per_target.<target>.per_fold_metrics``. Returns the DataFrame
    written to ``phase4_paired_tests.csv``.
    """
    matrices = sorted({c.matrix for c in cells})
    rows: list[dict] = []
    for matrix in matrices:
        for target in ALL_TARGETS:
            task = task_for_target(target)
            metrics = (
                CLASSIFICATION_TEST_METRICS if task == "classification"
                else REGRESSION_TEST_METRICS
            )
            per_family: dict[str, dict[str, np.ndarray]] = {}
            for c in cells:
                if c.matrix != matrix or c.family not in PRIMARY_FAMILY_NAMES:
                    continue
                if target not in c.per_target:
                    continue
                per_fold = c.per_target[target]["per_fold_metrics"]
                per_family[c.family] = {
                    k: np.asarray(v, dtype=float) for k, v in per_fold.items()
                }
            if len(per_family) < 2:
                continue
            cmps = all_pairwise_comparisons(per_family, metrics)
            rows.extend(comparisons_to_rows(matrix, target, cmps))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, default=None,
                        help="Override runs/phase_4/ source directory.")
    parser.add_argument(
        "--paired-out", type=Path,
        default=paths.REPORTS_TABLES_DIR / "phase4_paired_tests.csv",
    )
    parser.add_argument(
        "--gap-figure", type=Path,
        default=paths.REPORTS_FIGURES_DIR / "phase4_train_oof_gap.png",
    )
    parser.add_argument(
        "--calibration-figure", type=Path,
        default=paths.REPORTS_FIGURES_DIR / "phase4_calibration_pre.png",
    )
    args = parser.parse_args()

    paths.ensure_dirs()
    cells = latest_per_cell(load_all_cells(args.phase_dir))
    if not cells:
        raise RuntimeError("No Phase 4 cells found; run the benchmark first.")
    logger.info("Loaded %d unique (matrix, family) cells", len(cells))

    paired = recompute_paired_tests(cells)
    paired.to_csv(args.paired_out, index=False)
    logger.info("Wrote %s (%d rows)", args.paired_out, len(paired))

    plot_train_oof_gap(cells, args.gap_figure)
    plot_calibration_pre(cells, args.calibration_figure)


if __name__ == "__main__":
    main()
