"""Phase 3a baseline feature matrix builder.

Phase 3a establishes a performance floor on existing master-Parquet
features only — no new feature engineering yet. Phase 3b adds feature
groups one at a time and measures the lift each contributes against
this baseline.

Four feature sets are produced (two original, two with the planning-
conversation-mandated revision applied 2026-05-03):

* ``dialogue_only`` (original) — deployable. Uses only features that
  exist at pre-greenlight inference time: parser-derived screenplay
  structure, primary genre (one-hot), release year. Raw counts
  z-scored without log transform; ``log_runtime`` not included.
* ``with_budget`` (original) — sanity-check / ceiling. Adds
  ``log_budget`` to ``dialogue_only``. Not deployable.
* ``dialogue_only_logged`` (revised) — deployable. Same as
  ``dialogue_only`` but applies ``log1p`` to the heavy-tailed
  structural counts before z-scoring, and adds ``log_runtime``.
  Runtime is leak-free pre-greenlight (page count → minutes is the
  industry convention, ~1 page per minute) so it belongs in the
  deployable baseline.
* ``with_budget_logged`` (revised) — sanity-check. Same revision
  applied with ``log_budget`` added.

Returned matrices use ``imdb_id`` as the row index so downstream code
can align rows against the split assignments by id rather than position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths


# Numeric structural features computed at Phase 2 corpus-build time.
# Always present on ``films_joined.parquet``.
STRUCTURAL_FEATURES: tuple[str, ...] = (
    "n_scenes",
    "n_unique_characters",
    "n_dialogue_lines",
    "total_dialogue_chars",
    "total_action_chars",
    "dialogue_to_total_text_ratio",
    "parse_warning_count",
)

# Heavy-tailed structural counts that benefit from a log transform
# before z-scoring. ``dialogue_to_total_text_ratio`` is excluded
# (already a bounded ratio in [0, 1]). ``log1p`` is used because
# ``parse_warning_count`` is zero for the majority of films.
LOG_TRANSFORMABLE: frozenset[str] = frozenset(
    {
        "n_scenes",
        "n_unique_characters",
        "n_dialogue_lines",
        "total_dialogue_chars",
        "total_action_chars",
        "parse_warning_count",
    }
)

ERA_FEATURE: str = "release_year_parsed"
GENRE_FEATURE: str = "primary_genre_bucketed"
BUDGET_FEATURE: str = "log_budget"
RUNTIME_FEATURE: str = "runtime"
LOG_RUNTIME_NAME: str = "log_runtime"

GENRE_PREFIX: str = "genre_"


@dataclass(frozen=True)
class BaselineFeatureConfig:
    """Knobs for the baseline feature matrix.

    Override via :func:`dataclasses.replace`. Reference configurations:

    * ``dialogue_only`` (Phase 3a original):
      ``include_log_budget=False``, ``log_transform_structural=False``,
      ``include_log_runtime=False``.
    * ``with_budget`` (Phase 3a original): same as above but
      ``include_log_budget=True``.
    * ``dialogue_only_logged`` (Phase 3a revised):
      ``log_transform_structural=True``, ``include_log_runtime=True``.
    * ``with_budget_logged`` (Phase 3a revised): same as
      ``dialogue_only_logged`` with ``include_log_budget=True``.
    * ``dialogue_only_logged_lexical`` (Phase 3b lexical):
      ``log_transform_structural=True``, ``include_log_runtime=True``,
      ``include_lexical=True``.
    """
    structural: tuple[str, ...] = STRUCTURAL_FEATURES
    include_era: bool = True
    include_genre_one_hot: bool = True
    include_log_budget: bool = False
    # Apply ``log1p`` to the heavy-tailed structural counts in
    # :data:`LOG_TRANSFORMABLE` before z-scoring. The transformed
    # columns are renamed ``log_<original>`` for traceability.
    log_transform_structural: bool = False
    # Add ``log_runtime`` (computed inline as ``log1p(runtime)``; the
    # raw ``runtime`` column is on the master parquet, no derived
    # ``log_runtime`` column is stored there).
    include_log_runtime: bool = False
    # Phase 3b: include the precomputed lexical feature matrix from
    # ``data/processed/features_lexical.parquet`` (or the path given
    # by ``lexical_features_path``). The diagnostic-only column
    # ``_oov_rate_dialogue`` is excluded from the model-input matrix.
    include_lexical: bool = False
    lexical_features_path: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "features_lexical.parquet"
    )


def build_baseline_features(
    df: pd.DataFrame,
    cfg: BaselineFeatureConfig | None = None,
) -> pd.DataFrame:
    """Construct the Phase 3a baseline feature matrix.

    All numeric features are returned as float64. The genre column is
    one-hot encoded with the ``genre_<value>`` column convention (if
    ``include_genre_one_hot`` is True). No scaling is applied here —
    the modelling pipeline applies scaling inside cross-validation
    folds to avoid leakage.

    Parameters
    ----------
    df
        Master film DataFrame; must contain ``imdb_id`` plus every
        column referenced by the active config.
    cfg
        Feature configuration; ``None`` uses defaults (original
        dialogue-only, no log transforms, no runtime).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by ``imdb_id`` (one row per film, same
        length as ``df``).
    """
    cfg = cfg or BaselineFeatureConfig()
    pieces: list[pd.DataFrame] = []

    structural = df[list(cfg.structural)].astype(float).copy()
    if cfg.log_transform_structural:
        rename_map: dict[str, str] = {}
        for col in cfg.structural:
            if col in LOG_TRANSFORMABLE:
                structural[col] = np.log1p(structural[col])
                rename_map[col] = f"log_{col}"
        if rename_map:
            structural = structural.rename(columns=rename_map)
    pieces.append(structural)

    if cfg.include_era:
        pieces.append(df[[ERA_FEATURE]].astype(float))

    if cfg.include_genre_one_hot:
        dummies = pd.get_dummies(df[GENRE_FEATURE], prefix=GENRE_PREFIX.rstrip("_"), dtype=float)
        pieces.append(dummies)

    if cfg.include_log_budget:
        pieces.append(df[[BUDGET_FEATURE]].astype(float))

    if cfg.include_log_runtime:
        log_runtime = np.log1p(df[RUNTIME_FEATURE].astype(float).values)
        pieces.append(pd.DataFrame({LOG_RUNTIME_NAME: log_runtime}, index=df.index))

    out = pd.concat(pieces, axis=1)
    out.index = pd.Index(df["imdb_id"].values, name="imdb_id")

    if cfg.include_lexical:
        lex = pd.read_parquet(cfg.lexical_features_path)
        # Drop the diagnostic-only column from the model-input matrix.
        diag_cols = [c for c in lex.columns if c.startswith("_")]
        lex = lex.drop(columns=diag_cols)
        # Align by imdb_id; films missing from the lexical parquet end
        # up as all-NaN rows that the modelling pipeline's imputer
        # handles.
        out = out.join(lex, how="left")

    return out
