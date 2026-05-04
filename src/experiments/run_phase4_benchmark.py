"""CLI entry point for the Phase 4 benchmark.

Modes::

    # Smoke test: 1 family, 1 target, 1 matrix, tiny grid.
    python -m src.experiments.run_phase4_benchmark --mode smoke

    # Full primary tier on both matrices (the headline benchmark).
    python -m src.experiments.run_phase4_benchmark --mode primary

    # Secondary tier (Lasso, Linear-SVM) on all_five only.
    python -m src.experiments.run_phase4_benchmark --mode secondary

    # Both tiers in one invocation.
    python -m src.experiments.run_phase4_benchmark --mode full

The full-mode invocation is the canonical Phase 4 benchmark per
``docs/proposals/phase4_preregistration.md``. Smoke is for harness
verification before launching long runs and is not the deliverable.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

# sklearn 1.8 deprecated the explicit ``penalty`` kwarg on
# LogisticRegression in favor of ``l1_ratio``. We still pass
# ``penalty="l1"`` explicitly for the Lasso family because liblinear
# requires it; the warning is noisy without changing behavior, so we
# silence it for the duration of the benchmark.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.models.phase4.benchmark import (
    BenchmarkConfig,
    run_full_benchmark,
)
from src.models.phase4.families import FAMILIES, FamilySpec
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def _smoke_config() -> BenchmarkConfig:
    """A 5-minute end-to-end harness check.

    One primary family (linear), one secondary family disabled,
    ``all_five`` matrix only, ``roi_gt_2`` only, with the grid
    pruned to 3 alpha cells. Output to a separate ``smoke`` CSV
    so it does not collide with the deliverable benchmark.
    """
    # Smaller-grid clones of FamilySpec to keep the smoke test under a minute.
    smoke_linear = replace(
        FAMILIES["linear"],
        regression_grid={"alpha": [0.1, 1.0, 10.0]},
        classification_grid={"C": [0.1, 1.0, 10.0]},
    )
    # Patch the registry so the benchmark orchestrator picks up the override.
    FAMILIES[smoke_linear.name] = smoke_linear

    return BenchmarkConfig(
        mode="smoke",
        families=("linear",),
        secondary_families=(),
        matrices=("all_five",),
        secondary_matrices=(),
        targets=("roi_gt_2",),
        save_models=False,
        out_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_benchmark.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_paired.csv",
    )


def _smoke_histgb_config() -> BenchmarkConfig:
    """Smoke test for HistGB sample_weight routing.

    One primary family (histgb) on roi_gt_2 with a 4-cell grid.
    Verifies that ``model__sample_weight`` flows through
    ``GridSearchCV.fit`` and that scoring on held-out folds is not
    optimistic.
    """
    smoke_histgb = replace(
        FAMILIES["histgb"],
        regression_grid={
            "max_depth": [2, 3], "learning_rate": [0.05],
            "min_samples_leaf": [20],
        },
        classification_grid={
            "max_depth": [2, 3], "learning_rate": [0.05],
            "min_samples_leaf": [20],
        },
    )
    FAMILIES[smoke_histgb.name] = smoke_histgb

    return BenchmarkConfig(
        mode="smoke",
        families=("histgb",),
        secondary_families=(),
        matrices=("all_five",),
        secondary_matrices=(),
        targets=("roi_gt_2",),
        save_models=False,
        out_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_histgb.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_histgb_paired.csv",
    )


def _primary_config() -> BenchmarkConfig:
    """Pre-registered primary-tier benchmark on both matrices."""
    return BenchmarkConfig(
        mode="full",
        families=tuple(f.name for f in FAMILIES.values() if f.tier == "primary"),
        secondary_families=(),
        matrices=("all_five", "standalone_positive_union"),
        secondary_matrices=(),
    )


def _mpnet_config(save_models: bool = False) -> BenchmarkConfig:
    """Phase 4 Tier-A3 follow-up: primary tier on mpnet-replaced matrices.

    Outputs to a separate CSV so the canonical benchmark CSV (under the
    pre-registered MiniLM matrices) is unaffected. Per-cell save_run
    directories preserve the audit trail. ``save_models=True`` also
    overwrites the canonical winner artifacts in
    ``data/processed/phase4_primary_model_<target>.joblib`` based on
    the highest-scoring (matrix x family) cell across both the MiniLM
    and mpnet benchmarks.
    """
    return BenchmarkConfig(
        mode="full",
        families=tuple(f.name for f in FAMILIES.values() if f.tier == "primary"),
        secondary_families=(),
        matrices=("all_five_mpnet", "standalone_positive_union_mpnet"),
        secondary_matrices=(),
        out_table=paths.REPORTS_TABLES_DIR / "phase4_benchmark_mpnet.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_paired_tests_mpnet.csv",
        save_models=save_models,
    )


def _mpnet_save_config() -> BenchmarkConfig:
    """Mpnet matrices with model artifacts saved (overwrites canonical winner files)."""
    return _mpnet_config(save_models=True)


def _secondary_config() -> BenchmarkConfig:
    """Pre-registered secondary-tier benchmark on all_five only."""
    return BenchmarkConfig(
        mode="full",
        families=(),
        secondary_families=tuple(
            f.name for f in FAMILIES.values() if f.tier == "secondary"
        ),
        matrices=(),
        secondary_matrices=("all_five",),
        out_table=paths.REPORTS_TABLES_DIR / "phase4_secondary_benchmark.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_secondary_paired.csv",
        save_models=False,
    )


def _full_config() -> BenchmarkConfig:
    """Both tiers in one invocation."""
    return BenchmarkConfig(
        mode="full",
        families=tuple(f.name for f in FAMILIES.values() if f.tier == "primary"),
        secondary_families=tuple(
            f.name for f in FAMILIES.values() if f.tier == "secondary"
        ),
        matrices=("all_five", "standalone_positive_union"),
        secondary_matrices=("all_five",),
    )


CONFIG_BUILDERS = {
    "smoke": _smoke_config,
    "smoke_histgb": _smoke_histgb_config,
    "primary": _primary_config,
    "secondary": _secondary_config,
    "full": _full_config,
    "mpnet": _mpnet_config,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(CONFIG_BUILDERS),
        default="full",
        help="Which benchmark configuration to run.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="GridSearchCV n_jobs; -1 uses all cores.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level. DEBUG shows per-fold scores.",
    )
    args = parser.parse_args()

    set_log_level(args.log_level)
    cfg = CONFIG_BUILDERS[args.mode]()
    cfg = replace(cfg, n_jobs=args.n_jobs)

    logger.info("Phase 4 benchmark starting: mode=%s", args.mode)
    logger.info("Config: families=%s matrices=%s targets=%s",
                cfg.families, cfg.matrices, cfg.targets)
    if cfg.secondary_families:
        logger.info("Secondary: families=%s matrices=%s",
                    cfg.secondary_families, cfg.secondary_matrices)

    result = run_full_benchmark(cfg)

    logger.info("Phase 4 benchmark complete.")
    if result["winners"]:
        logger.info("Winners per target:")
        for target, winfo in result["winners"].items():
            logger.info(
                "  %s: %s on %s (OOF %s)",
                target, winfo["family"], winfo["matrix"],
                _format_metrics(winfo["oof_metrics_global"]),
            )


def _format_metrics(d: dict[str, float]) -> str:
    return ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(d.items())
        if isinstance(v, float) and not np.isnan(v)
    )


if __name__ == "__main__":
    main()
