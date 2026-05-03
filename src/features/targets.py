"""Phase 3 prediction targets.

Three targets are derived from ``budget`` and ``revenue`` on the master
processed corpus and modelled in parallel through Phases 3-4:

* ``log_roi`` (regression) — ``ln(revenue) - ln(budget)``. Natural log.
  Decomposable into the two component log columns already on the master
  table, and approximately symmetric about zero on this corpus.
* ``roi_gt_1`` (classification) — boolean ``revenue / budget > 1``.
  Approximately 80% positive on the working corpus (gross-profitable
  films). Imbalanced.
* ``roi_gt_2`` (classification) — boolean ``revenue / budget > 2``.
  Industry rule-of-thumb threshold for "net profitable after marketing
  and distribution overhead." Approximately balanced.

The three targets share threshold structure:
``roi_gt_1 == (log_roi > 0)`` and ``roi_gt_2 == (log_roi > ln 2)``. A
regression model on ``log_roi`` reproduces both classifiers by
thresholding, which makes Phase 4 cross-target comparisons direct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG_ROI_COL: str = "log_roi"
ROI_GT_1_COL: str = "roi_gt_1"
ROI_GT_2_COL: str = "roi_gt_2"

REGRESSION_TARGETS: tuple[str, ...] = (LOG_ROI_COL,)
CLASSIFICATION_TARGETS: tuple[str, ...] = (ROI_GT_1_COL, ROI_GT_2_COL)
ALL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Append the three target columns to ``df`` and return a new frame.

    Parameters
    ----------
    df
        Master film DataFrame; must contain ``budget`` and ``revenue``
        columns, both strictly positive (the master corpus filter
        already guarantees this).

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the three target columns appended.

    Raises
    ------
    ValueError
        If any film has a non-positive budget or revenue.
    """
    if not (df["budget"] > 0).all() or not (df["revenue"] > 0).all():
        raise ValueError("budget and revenue must be > 0 for every film")

    log_roi = np.log(df["revenue"].astype(float)) - np.log(df["budget"].astype(float))
    out = df.copy()
    out[LOG_ROI_COL] = log_roi
    out[ROI_GT_1_COL] = log_roi > 0
    out[ROI_GT_2_COL] = log_roi > np.log(2.0)
    return out
