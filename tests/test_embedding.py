"""Tests for ``src.features.embedding``.

The encoder forward pass is expensive, so unit tests use a fixture
of pre-computed mock pooled embeddings rather than loading MiniLM.
The smoke test that exercises the full pipeline runs only when
the real corpus is available and the cached pooled embeddings
exist on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.embedding import (
    DIAGNOSTIC_ONLY_COLUMNS,
    EmbeddingFeatureConfig,
    compute_embedding_features,
    embedding_feature_columns,
    fit_embedding_pca,
)
from src.utils import paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pooled_embeddings() -> pd.DataFrame:
    """Build a (50, 384) pooled-embedding fixture with a structured signal.

    Films 0-24 have one mean direction; films 25-49 have a different
    mean direction. PCA on a 40-row subset should recover at least
    one component aligned with this group structure.
    """
    rng = np.random.default_rng(42)
    n_films = 50
    embed_dim = 384
    base = rng.normal(size=(n_films, embed_dim)).astype(np.float32)
    # Shift the second group along the first 16 dimensions.
    base[25:, :16] += 2.0
    cols = [f"emb_{j:03d}" for j in range(embed_dim)]
    ids = [f"tt_synth_{i:02d}" for i in range(n_films)]
    return pd.DataFrame(base, columns=cols, index=pd.Index(ids, name="imdb_id"))


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_embedding_feature_columns_have_expected_count():
    cols = embedding_feature_columns(n_pca_components=32)
    assert len(cols) == 32
    assert cols[0] == "embed_pc_00"
    assert cols[31] == "embed_pc_31"
    assert len(DIAGNOSTIC_ONLY_COLUMNS) == 0


def test_embedding_feature_columns_change_with_k():
    cols = embedding_feature_columns(n_pca_components=8)
    assert len(cols) == 8


# ---------------------------------------------------------------------------
# PCA fit and project
# ---------------------------------------------------------------------------


def test_pca_fit_returns_correct_n_components(mock_pooled_embeddings):
    """K = 4 fit produces a PCA with 4 components."""
    cfg = EmbeddingFeatureConfig(n_pca_components=4)
    train_ids = list(mock_pooled_embeddings.index[:40])
    fitted = fit_embedding_pca(mock_pooled_embeddings, train_ids, cfg)
    assert fitted.pca.n_components_ == 4
    assert len(fitted.train_ids) == 40


def test_compute_embedding_features_returns_expected_shape(mock_pooled_embeddings):
    """Output shape = (n_films, n_pca_components)."""
    cfg = EmbeddingFeatureConfig(n_pca_components=4)
    train_ids = list(mock_pooled_embeddings.index[:40])
    fitted = fit_embedding_pca(mock_pooled_embeddings, train_ids, cfg)
    df = compute_embedding_features(mock_pooled_embeddings, fitted)
    assert df.shape == (50, 4)
    assert df.index.name == "imdb_id"
    assert list(df.columns) == ["embed_pc_00", "embed_pc_01", "embed_pc_02", "embed_pc_03"]


def test_pca_recovers_group_structure(mock_pooled_embeddings):
    """PC1 separates the two embedded groups."""
    cfg = EmbeddingFeatureConfig(n_pca_components=4)
    # Fit on a balanced subset of train films from both groups.
    train_ids = (
        list(mock_pooled_embeddings.index[:20])
        + list(mock_pooled_embeddings.index[25:45])
    )
    fitted = fit_embedding_pca(mock_pooled_embeddings, train_ids, cfg)
    df = compute_embedding_features(mock_pooled_embeddings, fitted)
    # PC1 should have substantially different mean for the two groups.
    group_a_pc1_mean = df.iloc[:25]["embed_pc_00"].mean()
    group_b_pc1_mean = df.iloc[25:]["embed_pc_00"].mean()
    assert abs(group_a_pc1_mean - group_b_pc1_mean) > 1.0


# ---------------------------------------------------------------------------
# No-leakage test
# ---------------------------------------------------------------------------


def test_no_leakage_train_inference_unchanged(mock_pooled_embeddings):
    """Adding cal/test films at PCA-fit time does not change train-film projections.

    Strict statement: PCA components are determined by the training
    fold only, so the projection of any train film must be the same
    regardless of the inference batch composition.
    """
    cfg = EmbeddingFeatureConfig(n_pca_components=4)
    train_ids = list(mock_pooled_embeddings.index[:30])

    fitted = fit_embedding_pca(mock_pooled_embeddings, train_ids, cfg)

    one_train_film = mock_pooled_embeddings.loc[[train_ids[0]]]
    df_solo = compute_embedding_features(one_train_film, fitted)

    # Add a non-train film to the inference batch.
    extra = mock_pooled_embeddings.iloc[[31]]
    one_plus_extra = pd.concat([one_train_film, extra])
    df_with_extra = compute_embedding_features(one_plus_extra, fitted)

    np.testing.assert_allclose(
        df_solo.loc[train_ids[0]].values,
        df_with_extra.loc[train_ids[0]].values,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Real-corpus smoke (only runs if cached pooled embeddings exist)
# ---------------------------------------------------------------------------


def test_smoke_pca_pipeline_on_cached_pooled_embeddings():
    """If pooled embeddings are cached, exercise the full PCA pipeline.

    Skipped on a fresh checkout where the encoder has not been run.
    """
    cache_path = paths.DATA_PROCESSED_DIR / "embeddings_minilm_pooled.parquet"
    splits_path = paths.DATA_PROCESSED_DIR / "split_assignments.parquet"
    if not cache_path.is_file():
        pytest.skip(
            f"Cached pooled embeddings missing at {cache_path}; "
            "run `python3 -m src.experiments.run_embedding_ablation` first"
        )
    if not splits_path.is_file():
        pytest.skip(f"Split assignments missing at {splits_path}")

    pooled = pd.read_parquet(cache_path)
    splits = pd.read_parquet(splits_path)
    train_ids = list(splits.loc[splits["split"] == "train", "imdb_id"])

    cfg = EmbeddingFeatureConfig(n_pca_components=32)
    fitted = fit_embedding_pca(pooled, train_ids, cfg)
    df = compute_embedding_features(pooled, fitted)

    assert df.shape == (len(pooled), 32)
    assert df.index.name == "imdb_id"
    assert not np.isnan(df.values).any()
    assert not np.isinf(df.values).any()
    # Cumulative variance explained should be sensible (at least 0.5).
    cum_var = fitted.pca.explained_variance_ratio_.cumsum()[-1]
    assert cum_var > 0.5
