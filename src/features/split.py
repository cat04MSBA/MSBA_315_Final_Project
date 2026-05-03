"""Train / calibration / test split for Phase 3.

The split happens once, before any feature engineering that depends on
the data distribution (LDA, scalers, embedders trained on the corpus,
etc.). Per ``PROJECT_CONTEXT.md`` Section 6, the test set is touched
only at final evaluation in Phase 8; the calibration set is reserved
for Phase 5 conformal prediction.

Stratification balances the two largest known confounds in the corpus:
``primary_genre_bucketed`` and a coarse ``decade_bucket`` derived from
``release_year_parsed``. Pre-1980 decades collapse into one stratum
(per ``DATA_NOTES.md`` Section 2 — pre-1980s decades each have <30
films), and 2010s + 2020s collapse into one stratum because the corpus
2020s coverage is only 2020-2023.

Composite (genre, decade) cells with fewer than ``rare_cell_threshold``
films are pooled into a single ``"rare|rare"`` stratum so that
``StratifiedShuffleSplit`` is well-defined for every named stratum.

All knobs live on :class:`SplitConfig`. The default reproduces the
planning conversation's reference split (70 / 15 / 15, seed 42).

Run from the project root:

    python -m src.features.split

Idempotent.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


SPLIT_NAMES: tuple[str, ...] = ("train", "cal", "test")
RARE_LABEL: str = "rare"


@dataclass(frozen=True)
class SplitConfig:
    """Knobs for the train/cal/test split.

    Override any field via :func:`dataclasses.replace` to test
    alternatives.
    """
    # Target proportions; must sum to 1.0.
    train_frac: float = 0.70
    cal_frac: float = 0.15
    test_frac: float = 0.15

    # Decade-bucket labels (used to build the stratification key).
    decade_pre_1980_label: str = "pre_1980s"
    decade_post_2010_label: str = "2010s_2020s"

    # Composite (genre, decade) cells with fewer than this many films
    # are pooled into a single rare stratum so the stratified split is
    # well-defined for every named stratum.
    rare_cell_threshold: int = 5

    seed: int = 42

    in_path: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "films_joined.parquet"
    )
    out_path: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    )
    diagnostic_path: Path = field(
        default_factory=lambda: paths.REPORTS_TABLES_DIR / "phase3_split_diagnostics.csv"
    )


def assign_decade_bucket(year: int, cfg: SplitConfig) -> str:
    """Bucket a film's release year into a stratification-friendly decade.

    Pre-1980 collapses into one stratum (``DATA_NOTES.md`` §2: each
    pre-1980 decade has <30 films, too thin for independent
    stratification). 2010s and 2020s collapse because the corpus
    2020s coverage is only 2020-2023.
    """
    if year < 1980:
        return cfg.decade_pre_1980_label
    if year < 1990:
        return "1980s"
    if year < 2000:
        return "1990s"
    if year < 2010:
        return "2000s"
    return cfg.decade_post_2010_label


def build_strata(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Build a composite stratification label for each film.

    Returns a Series indexed by ``df.index`` with values like
    ``"Drama|2000s"``. Cells with fewer than ``cfg.rare_cell_threshold``
    films are re-bucketed into ``"rare|rare"``.
    """
    decade = (
        df["release_year_parsed"]
        .astype(int)
        .map(lambda y: assign_decade_bucket(y, cfg))
    )
    composite = df["primary_genre_bucketed"].astype(str) + "|" + decade
    counts = composite.value_counts()
    sparse_cells = set(counts.index[counts < cfg.rare_cell_threshold])
    rare_label = f"{RARE_LABEL}|{RARE_LABEL}"
    return composite.where(~composite.isin(sparse_cells), other=rare_label)


def make_splits(
    df: pd.DataFrame,
    cfg: SplitConfig | None = None,
) -> pd.DataFrame:
    """Carve the corpus into train / calibration / test splits.

    Two-step stratified shuffle split: first peel off the test fraction
    from the full corpus, then split the remainder into train and
    calibration in proportions that re-normalize to the original
    train_frac : cal_frac ratio.

    Parameters
    ----------
    df
        Master film DataFrame. Must contain ``imdb_id``,
        ``release_year_parsed``, ``primary_genre_bucketed``.
    cfg
        Split configuration; ``None`` uses defaults.

    Returns
    -------
    pd.DataFrame
        One row per film (same length as ``df``), with columns
        ``imdb_id``, ``stratum``, and ``split`` (one of ``"train"``,
        ``"cal"``, ``"test"``).

    Raises
    ------
    ValueError
        If the fractions don't sum to 1, or if a required column is
        missing.
    """
    cfg = cfg or SplitConfig()
    fracs = (cfg.train_frac, cfg.cal_frac, cfg.test_frac)
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(
            f"train+cal+test fractions must sum to 1.0; got {fracs}"
        )
    for required in ("imdb_id", "release_year_parsed", "primary_genre_bucketed"):
        if required not in df.columns:
            raise ValueError(f"input DataFrame missing required column {required!r}")

    df = df.reset_index(drop=True)
    strata = build_strata(df, cfg)

    # Peel off test fraction first.
    test_split = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.test_frac, random_state=cfg.seed
    )
    train_cal_idx, test_idx = next(test_split.split(df, strata))

    # Split the remainder into train and cal at the renormalized ratio.
    cal_relative = cfg.cal_frac / (cfg.train_frac + cfg.cal_frac)
    cal_split = StratifiedShuffleSplit(
        n_splits=1, test_size=cal_relative, random_state=cfg.seed
    )
    sub_strata = strata.iloc[train_cal_idx]
    train_rel, cal_rel = next(cal_split.split(np.zeros(len(train_cal_idx)), sub_strata))
    train_idx = train_cal_idx[train_rel]
    cal_idx = train_cal_idx[cal_rel]

    split_arr = np.empty(len(df), dtype=object)
    split_arr[train_idx] = "train"
    split_arr[cal_idx] = "cal"
    split_arr[test_idx] = "test"

    return pd.DataFrame(
        {
            "imdb_id": df["imdb_id"].values,
            "stratum": strata.values,
            "split": split_arr,
        }
    )


def split_diagnostics(splits: pd.DataFrame) -> pd.DataFrame:
    """One row per stratum with total + per-split counts.

    Useful sanity check: every stratum should have non-zero counts in
    each split (or be the rare-pool stratum where small counts are
    expected).
    """
    pivoted = (
        splits.groupby(["stratum", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SPLIT_NAMES), fill_value=0)
    )
    pivoted["total"] = pivoted.sum(axis=1)
    pivoted = pivoted.sort_values("total", ascending=False)
    return pivoted.reset_index()


def main() -> None:
    """CLI entrypoint: build split, save, log diagnostic counts."""
    cfg = SplitConfig()
    paths.ensure_dirs()

    logger.info("Loading master corpus")
    df = pd.read_parquet(cfg.in_path)
    logger.info("Loaded %d films", len(df))

    logger.info("Building train/cal/test split (seed=%d)", cfg.seed)
    splits = make_splits(df, cfg)

    counts = splits["split"].value_counts().reindex(SPLIT_NAMES)
    logger.info(
        "Split sizes: train=%d cal=%d test=%d",
        int(counts["train"]),
        int(counts["cal"]),
        int(counts["test"]),
    )

    diagnostics = split_diagnostics(splits)
    n_strata = len(diagnostics)
    n_rare = int(
        diagnostics.loc[diagnostics["stratum"] == f"{RARE_LABEL}|{RARE_LABEL}", "total"].sum()
    )
    logger.info("Stratification: %d strata, %d films pooled into rare", n_strata, n_rare)

    splits.to_parquet(cfg.out_path)
    diagnostics.to_csv(cfg.diagnostic_path, index=False)
    logger.info("Saved split assignments and diagnostic table")


if __name__ == "__main__":
    main()
