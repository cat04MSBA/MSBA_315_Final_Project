"""CLI entry point for the Phase 6 cost-decision pipeline.

Run from project root::

    python -m src.experiments.run_phase6_decision
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from src.decision.pipeline import run_phase6
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_log_level(args.log_level)
    logger.info("Phase 6 cost-decision pipeline starting")
    run_phase6()
    logger.info("Phase 6 complete")


if __name__ == "__main__":
    main()
