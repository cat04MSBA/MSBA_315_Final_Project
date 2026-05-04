"""Phase 4 input matrix builders.

Two matrices are pre-registered in
``docs/proposals/phase4_preregistration.md`` Section 8:

* ``all_five``: the full Phase 3 feature matrix (structural baseline +
  lexical + sentiment + topic + character_network + embedding). 127
  feature columns. Maximum-information matrix; the Phase 3c evidence
  showed SVM-RBF extracts substantial signal from it.

* ``standalone_positive_union``: structural baseline + topic +
  character_network + embedding only (drops the two Phase 3b standalone
  null groups). Approximately 92 feature columns. Honors the
  pre-registration discipline from Phase 3b.

Both matrices are constructed via :class:`BaselineFeatureConfig` from
``src.features.baseline_features`` so the column conventions match the
Phase 3 trainer exactly. The structural baseline always uses the
revised parameterization (``log_transform_structural=True``,
``include_log_runtime=True``) and never includes ``log_budget``
(budget is not available pre-greenlight; the deployable framing
applies through Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.features.baseline_features import (
    BaselineFeatureConfig,
    build_baseline_features,
)
from src.utils import paths

MatrixName = Literal[
    "all_five",
    "standalone_positive_union",
    "all_five_mpnet",
    "standalone_positive_union_mpnet",
]


_MPNET_EMBEDDING_PATH = paths.DATA_PROCESSED_DIR / "features_embedding_mpnet.parquet"


@dataclass(frozen=True)
class MatrixSpec:
    """Identifier + feature-config recipe for a Phase 4 input matrix."""

    name: MatrixName
    feature_config: BaselineFeatureConfig
    description: str


MATRICES: dict[str, MatrixSpec] = {
    "all_five": MatrixSpec(
        name="all_five",
        feature_config=BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
            include_lexical=True,
            include_sentiment=True,
            include_topic=True,
            include_character_network=True,
            include_embedding=True,
        ),
        description=(
            "Structural baseline (revised) + all five Phase 3b feature "
            "groups. The Phase 3c maximum-information matrix; 127 "
            "feature columns."
        ),
    ),
    "standalone_positive_union": MatrixSpec(
        name="standalone_positive_union",
        feature_config=BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
            include_lexical=False,
            include_sentiment=False,
            include_topic=True,
            include_character_network=True,
            include_embedding=True,
        ),
        description=(
            "Structural baseline (revised) + the three Phase 3b groups "
            "that landed standalone partial-positive (topic, "
            "character_network, embedding). Drops lexical and sentiment "
            "(the two standalone nulls). Approximately 92 feature "
            "columns."
        ),
    ),
    # Phase 4 Tier-A3 additions: mpnet-base sentence-transformer
    # embeddings replacing the Phase 3 MiniLM embeddings. Same column
    # naming convention so the benchmark + stacking + diagnostic code
    # picks up the new features without modification.
    "all_five_mpnet": MatrixSpec(
        name="all_five_mpnet",
        feature_config=BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
            include_lexical=True,
            include_sentiment=True,
            include_topic=True,
            include_character_network=True,
            include_embedding=True,
            embedding_features_path=_MPNET_EMBEDDING_PATH,
        ),
        description=(
            "Same structure as `all_five` but with the embedding group "
            "replaced by mpnet-base (768-dim source, 32 PCA components, "
            "cumulative variance 0.7434). Phase 4 Tier-A escalation "
            "experiment to test whether a stronger sentence encoder "
            "lifts the corpus ceiling."
        ),
    ),
    "standalone_positive_union_mpnet": MatrixSpec(
        name="standalone_positive_union_mpnet",
        feature_config=BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
            include_lexical=False,
            include_sentiment=False,
            include_topic=True,
            include_character_network=True,
            include_embedding=True,
            embedding_features_path=_MPNET_EMBEDDING_PATH,
        ),
        description=(
            "Same structure as `standalone_positive_union` but with the "
            "mpnet-base embedding group. Pairs with `all_five_mpnet` "
            "for the Tier-A3 ablation."
        ),
    ),
}


def build_matrix(spec: MatrixSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Construct the feature matrix for one specification.

    Parameters
    ----------
    spec
        Matrix specification (name + feature config).
    df
        The master film DataFrame, typically loaded from
        ``data/processed/films_joined.parquet`` and filtered to the
        train split before benchmark training.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by ``imdb_id``. Numeric features only;
        scaling is applied inside cross-validation folds by the model
        pipeline (not here) to avoid leakage.
    """
    return build_baseline_features(df, spec.feature_config)


def get_matrix(name: str) -> MatrixSpec:
    """Look up a matrix specification by name."""
    if name not in MATRICES:
        raise KeyError(
            f"Unknown matrix {name!r}; known: {sorted(MATRICES)!r}"
        )
    return MATRICES[name]
