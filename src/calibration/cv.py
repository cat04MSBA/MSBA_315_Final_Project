"""5-fold cross-validation helpers within the calibration set.

The 257-film calibration set is used both to fit the calibration
procedure and to evaluate empirical coverage. Honest evaluation
requires cross-validation within the set: for each fold, fit on
the other 4 folds and evaluate on the held-out fold.

The deployed artifact at the end of Phase 5 uses **all 257 films**
to fit (not the per-fold reduced set); the per-fold metrics are
the honest estimates of how well the deployed artifact will
generalize.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.utils import paths


SEED: int = 42
N_FOLDS: int = 5


def load_cal_assignments() -> pd.DataFrame:
    """Return the 257 calibration-set imdb_ids."""
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    return splits.loc[splits["split"] == "cal"].reset_index(drop=True)


def _decade_bucket(year: int) -> str:
    """Match the bucketing convention used in Phase 3 split assignments."""
    if year < 1980:
        return "pre_1980s"
    if year < 1990:
        return "1980s"
    if year < 2000:
        return "1990s"
    if year < 2010:
        return "2000s"
    return "2010s_2020s"


def stratified_cal_folds(
    y: np.ndarray,
    task: str,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (fit_idx, eval_idx) tuples for each of n_folds.

    For classification, stratification is on the binary target.
    For regression, stratification is on a 4-quantile binning of
    ``y`` (since ``decade_bucket`` is not directly available here
    and quantile binning is a reasonable proxy).
    """
    y = np.asarray(y)
    if task == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fit_idx, eval_idx in cv.split(np.zeros(len(y)), y.astype(int)):
            yield fit_idx, eval_idx
    else:
        # Regression: quantile-bin y for stratification stability.
        bins = np.quantile(y, np.linspace(0, 1, 5))
        bins[0] = -np.inf
        bins[-1] = np.inf
        bin_labels = np.digitize(y, bins[1:-1])
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fit_idx, eval_idx in cv.split(np.zeros(len(y)), bin_labels):
            yield fit_idx, eval_idx
