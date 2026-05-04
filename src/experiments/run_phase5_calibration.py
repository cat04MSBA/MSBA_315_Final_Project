"""CLI entry point for the Phase 5 calibration pipeline.

Modes::

    # Smoke test on one target.
    python -m src.experiments.run_phase5_calibration --targets roi_gt_2

    # Full Phase 5 (all three targets).
    python -m src.experiments.run_phase5_calibration

The full mode is the canonical Phase 5 invocation per
``docs/proposals/phase5_preregistration.md``.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import warnings

# sklearn 1.8 deprecated the explicit ``penalty`` kwarg on
# LogisticRegression; CalibratedClassifierCV's internal Platt fit
# does not need to surface this in our run logs.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.calibration.pipeline import ALL_TARGETS, run_phase5
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+", default=list(ALL_TARGETS),
        help="Subset of targets to calibrate (default: all three).",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_log_level(args.log_level)
    logger.info("Phase 5 calibration starting | targets=%s", args.targets)
    results = run_phase5(tuple(args.targets))
    logger.info("Phase 5 calibration complete | %d targets calibrated", len(results))


if __name__ == "__main__":
    main()
