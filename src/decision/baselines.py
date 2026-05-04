"""Naive baseline decision strategies for Phase 6 comparison.

Five baselines per ``phase6_preregistration.md`` Section 5:
Always-Greenlight, Always-Pass, Read-Everything, Random, Genre-prior.
Each returns a per-film action sequence which is then scored under
the same cost matrix as the system's decision rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.cost_matrix import Action

SEED: int = 42

# Per-genre positive-class base rates from the train split's
# ``roi_gt_2`` column. Computed once on a 1199-film train fold
# (``primary_genre_bucketed`` -> P(roi_gt_2 = 1)). Used by the
# Genre-prior baseline. Numbers below are computed at first call;
# see :func:`_compute_genre_priors`.
_GENRE_PRIORS: dict[str, float] | None = None


def _compute_genre_priors() -> dict[str, float]:
    """Compute per-genre positive base rate on the train split."""
    global _GENRE_PRIORS
    if _GENRE_PRIORS is not None:
        return _GENRE_PRIORS

    from src.features.targets import add_targets
    from src.utils import paths

    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df = add_targets(df)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df[df["imdb_id"].isin(train_ids)].reset_index(drop=True)
    priors = (
        df_train.groupby("primary_genre_bucketed")["roi_gt_2"]
        .mean()
        .to_dict()
    )
    _GENRE_PRIORS = {str(k): float(v) for k, v in priors.items()}
    return _GENRE_PRIORS


def always_greenlight(imdb_ids: list[str], **kwargs) -> list[Action]:
    return ["Greenlight"] * len(imdb_ids)


def always_pass(imdb_ids: list[str], **kwargs) -> list[Action]:
    return ["Pass"] * len(imdb_ids)


def read_everything(imdb_ids: list[str], **kwargs) -> list[Action]:
    return ["Refer"] * len(imdb_ids)


def random_baseline(imdb_ids: list[str], **kwargs) -> list[Action]:
    """Equal-probability random choice per film, seeded."""
    rng = np.random.default_rng(SEED)
    choices = rng.choice(["Greenlight", "Pass", "Refer"], size=len(imdb_ids))
    return list(choices)


def genre_prior(
    imdb_ids: list[str], genres: list[str], **kwargs,
) -> list[Action]:
    """Greenlight if the genre's train-split positive rate >= 0.5; else Pass."""
    priors = _compute_genre_priors()
    actions: list[Action] = []
    for genre in genres:
        rate = priors.get(genre, 0.5)
        actions.append("Greenlight" if rate >= 0.5 else "Pass")
    return actions


BASELINES = {
    "always_greenlight": always_greenlight,
    "always_pass": always_pass,
    "read_everything": read_everything,
    "random": random_baseline,
    "genre_prior": genre_prior,
}
