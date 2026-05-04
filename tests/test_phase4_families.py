"""Tests for the Phase 4 family registry."""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator

from src.models.phase4 import families
from src.models.phase4.families import FAMILIES, FamilySpec, grid_size, primary_tier, secondary_tier


def test_eight_families_registered():
    """Six pre-registered + LightGBM + XGBoost (post-Tier-A follow-up)."""
    assert set(FAMILIES.keys()) == {
        "linear", "histgb", "random_forest", "svm_rbf",
        "lightgbm", "xgboost",
        "lasso", "linear_svm",
    }


def test_primary_and_secondary_tier_split():
    primary = {f.name for f in primary_tier()}
    secondary = {f.name for f in secondary_tier()}
    assert primary == {
        "linear", "histgb", "random_forest", "svm_rbf",
        "lightgbm", "xgboost",
    }
    assert secondary == {"lasso", "linear_svm"}
    assert primary.isdisjoint(secondary)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_factories_produce_estimator(name: str):
    spec = FAMILIES[name]
    reg = spec.regressor_factory()
    clf = spec.classifier_factory()
    assert isinstance(reg, BaseEstimator)
    assert isinstance(clf, BaseEstimator)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_grids_non_empty(name: str):
    spec = FAMILIES[name]
    assert spec.regression_grid, f"{name} has empty regression grid"
    assert spec.classification_grid, f"{name} has empty classification grid"


def test_grid_size_known_values():
    """Pre-registration Section 4 specifies grid sizes per family."""
    assert grid_size(FAMILIES["linear"], "regression") == 13
    assert grid_size(FAMILIES["linear"], "classification") == 13
    assert grid_size(FAMILIES["histgb"], "classification") == 36
    assert grid_size(FAMILIES["random_forest"], "classification") == 18
    assert grid_size(FAMILIES["svm_rbf"], "classification") == 36
    assert grid_size(FAMILIES["lasso"], "classification") == 9
    assert grid_size(FAMILIES["linear_svm"], "classification") == 4


def test_balancing_modes_valid():
    """Each family declares one of the three valid balancing modes."""
    valid = {"class_weight", "sample_weight", "none"}
    for name, spec in FAMILIES.items():
        assert spec.balancing in valid, f"{name}: invalid balancing {spec.balancing!r}"


def test_score_methods_for_classifiers():
    """SVM families use decision_function; the others use predict_proba."""
    assert FAMILIES["svm_rbf"].score_method == "decision_function"
    assert FAMILIES["linear_svm"].score_method == "decision_function"
    for name in ("linear", "histgb", "random_forest", "lasso"):
        assert FAMILIES[name].score_method == "predict_proba"


def test_get_family_unknown_raises():
    with pytest.raises(KeyError):
        families.get_family("nonexistent_family")
