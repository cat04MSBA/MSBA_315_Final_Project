"""Tests for the Phase 4 input matrix builders."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.phase4.matrices import (
    MATRICES,
    build_matrix,
    get_matrix,
)
from src.utils import paths


@pytest.fixture(scope="module")
def df_train():
    """Load the train split of the master corpus once per test module."""
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    return df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)


def test_pre_registered_matrices_present():
    """The two pre-registered matrices must always be available.

    Tier-A3 added two mpnet variants; the test allows additional
    matrices to be registered but requires the original pair.
    """
    assert {"all_five", "standalone_positive_union"}.issubset(set(MATRICES))


def test_mpnet_matrices_present_after_a3():
    """The Tier-A3 mpnet variants land alongside the originals."""
    assert {"all_five_mpnet", "standalone_positive_union_mpnet"}.issubset(set(MATRICES))


def test_get_matrix_unknown_raises():
    with pytest.raises(KeyError):
        get_matrix("ridge_only")


def test_all_five_has_all_phase3_groups(df_train):
    spec = MATRICES["all_five"]
    X = build_matrix(spec, df_train)
    cols = set(X.columns)
    # At least one column from each Phase 3 group should appear.
    assert any(c.startswith("log_") for c in cols), "structural log-counts missing"
    assert "mtld_dialogue" in cols, "lexical missing"
    assert any(c.startswith("vader_") for c in cols), "sentiment missing"
    assert any(c.startswith("topic_") for c in cols), "topic missing"
    assert any(c.startswith("network_") for c in cols), "character_network missing"
    assert any(c.startswith("embed_pc_") for c in cols), "embedding missing"


def test_standalone_positive_union_drops_lexical_and_sentiment(df_train):
    spec = MATRICES["standalone_positive_union"]
    X = build_matrix(spec, df_train)
    cols = set(X.columns)
    # Lexical and sentiment dropped.
    assert "mtld_dialogue" not in cols
    assert not any(c.startswith("vader_") for c in cols)
    assert not any(c.startswith("nrc_") for c in cols)
    # Topic, character_network, embedding present.
    assert any(c.startswith("topic_") for c in cols)
    assert any(c.startswith("network_") for c in cols)
    assert any(c.startswith("embed_pc_") for c in cols)


def test_all_five_is_strict_superset_of_union(df_train):
    a = build_matrix(MATRICES["all_five"], df_train)
    b = build_matrix(MATRICES["standalone_positive_union"], df_train)
    assert set(b.columns).issubset(set(a.columns))
    assert len(a.columns) > len(b.columns)


def test_no_log_budget_in_either_matrix(df_train):
    """Budget is post-greenlight; deployable matrices exclude it."""
    for name in MATRICES:
        X = build_matrix(MATRICES[name], df_train)
        assert "log_budget" not in X.columns, f"{name} leaked log_budget"


def test_log_runtime_present_in_both(df_train):
    """Runtime is leak-free pre-greenlight (page-count convention)."""
    for name in MATRICES:
        X = build_matrix(MATRICES[name], df_train)
        assert "log_runtime" in X.columns, f"{name} missing log_runtime"


def test_row_count_matches_train_split(df_train):
    for name in MATRICES:
        X = build_matrix(MATRICES[name], df_train)
        assert len(X) == len(df_train)
