"""Tests for the Phase 7 explanation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explanation.global_importance import (
    global_shap_ranking,
    spearman_rank_correlation,
)
from src.explanation.per_film import (
    _build_rationale,
    _readable_feature,
    per_film_attributions,
)


# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------


def test_global_shap_ranking_orders_by_mean_abs():
    """Higher mean |SHAP| ranks first."""
    sv = np.array([
        [0.1, 0.2, -0.3],
        [-0.1, 0.4, -0.5],
        [0.0, 0.3, 0.6],
    ])
    feature_names = ["a", "b", "c"]
    df = global_shap_ranking(sv, feature_names)
    assert list(df["feature"]) == ["c", "b", "a"]
    assert df["rank"].tolist() == [1, 2, 3]


def test_global_shap_ranking_validates_shape():
    sv = np.zeros((10, 3))
    with pytest.raises(ValueError):
        global_shap_ranking(sv, ["a", "b"])


def test_spearman_rank_correlation_perfect_agreement():
    df = pd.DataFrame({
        "feature": list("abcdefghij"),
        "shap_rank": list(range(1, 11)),
        "native_rank": list(range(1, 11)),
    })
    rho = spearman_rank_correlation(df)
    assert rho == pytest.approx(1.0)


def test_spearman_rank_correlation_perfect_inverse():
    df = pd.DataFrame({
        "feature": list("abcdefghij"),
        "shap_rank": list(range(1, 11)),
        "native_rank": list(range(10, 0, -1)),
    })
    rho = spearman_rank_correlation(df)
    assert rho == pytest.approx(-1.0)


def test_spearman_rank_correlation_handles_nan():
    df = pd.DataFrame({
        "feature": list("abc"),
        "shap_rank": [1, np.nan, 3],
        "native_rank": [1, 2, 3],
    })
    # Two valid observations -> length under threshold -> NaN
    rho = spearman_rank_correlation(df)
    assert np.isnan(rho)


# ---------------------------------------------------------------------------
# per_film
# ---------------------------------------------------------------------------


def test_readable_feature_genre():
    assert _readable_feature("genre_Comedy") == "Genre=Comedy"


def test_readable_feature_topic():
    assert _readable_feature("topic_05_proportion") == "Topic 05 proportion"


def test_readable_feature_embedding():
    assert _readable_feature("embed_pc_12") == "Embedding PC 12"


def test_readable_feature_log_transform():
    assert _readable_feature("log_n_scenes") == "log(n_scenes)"


def test_per_film_attributions_returns_one_row_per_film():
    sv = np.array([
        [0.5, -0.2, 0.1],
        [-0.4, 0.3, 0.1],
    ])
    feature_names = ["log_n_scenes", "genre_Drama", "embed_pc_01"]
    imdb_ids = ["tt0000001", "tt0000002"]
    df = per_film_attributions(imdb_ids, sv, feature_names, base_value=0.0, top_k=2)
    assert len(df) == 2
    assert "top_pos_features" in df.columns
    assert "top_neg_features" in df.columns
    assert "rationale_features" in df.columns
    assert df.iloc[0]["imdb_id"] == "tt0000001"


def test_build_rationale_returns_non_empty():
    pos = [("genre_Action", 0.123)]
    neg = [("genre_Romance", -0.087)]
    s = _build_rationale(0.5, pos, neg)
    assert "pushing" in s.lower()
    assert "pulling" in s.lower()
    assert len(s) > 50


def test_build_rationale_handles_empty():
    s = _build_rationale(0.5, [], [])
    assert "no notable" in s.lower()
