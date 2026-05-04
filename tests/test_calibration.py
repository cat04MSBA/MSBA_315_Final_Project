"""Tests for the Phase 5 calibration modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_perfectly_calibrated_returns_low_ece():
    """If predicted probs match empirical positive rate, ECE -> 0."""
    rng = np.random.default_rng(0)
    n = 5000
    probs = rng.uniform(0, 1, n)
    # y_true ~ Bernoulli(probs); on a large sample, ECE should be small.
    y_true = (rng.uniform(0, 1, n) < probs).astype(int)
    ece = expected_calibration_error(y_true, probs, n_bins=10)
    assert ece < 0.05, f"Perfectly calibrated should yield ECE near 0; got {ece}"


def test_constant_overconfidence_high_ece():
    """If probs say 0.9 but y_true is 50% positive, ECE should be ~0.4."""
    n = 1000
    probs = np.full(n, 0.9)
    y_true = np.zeros(n, dtype=int)
    y_true[: n // 2] = 1
    ece = expected_calibration_error(y_true, probs, n_bins=10)
    assert 0.35 < ece < 0.45, f"Expected ~0.4, got {ece}"


def test_brier_score_bounds():
    """Brier is in [0, 1] with 0 = perfect, 1 = worst."""
    n = 100
    y_true = np.zeros(n)
    y_prob_perfect = np.zeros(n)
    y_prob_worst = np.ones(n)
    assert brier_score(y_true, y_prob_perfect) == pytest.approx(0.0)
    assert brier_score(y_true, y_prob_worst) == pytest.approx(1.0)


def test_negative_log_likelihood_finite_on_extremes():
    """log_loss should be finite even when probs are 0 or 1 (clipped internally)."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 1.0, 0.0])
    nll = negative_log_likelihood(y_true, y_prob)
    assert np.isfinite(nll)
    assert nll > 0  # bad predictions -> high loss


def test_reliability_curve_shape():
    """reliability_curve returns three arrays of length n_bins."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    mean_pred, accuracy, counts = reliability_curve(y_true, y_prob, n_bins=10)
    assert mean_pred.shape == (10,)
    assert accuracy.shape == (10,)
    assert counts.shape == (10,)
    assert counts.sum() == 200


def test_mce_is_at_least_ece():
    """Maximum bin error is always >= mean bin error."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 500)
    y_prob = rng.uniform(0, 1, 500)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
    assert mce >= ece


# ---------------------------------------------------------------------------
# conformal: hand-rolled LAC
# ---------------------------------------------------------------------------


def test_lac_quantiles_monotone_in_confidence():
    """Quantile must be non-decreasing in the confidence level."""
    from src.calibration.conformal import _fit_lac_quantiles

    rng = np.random.default_rng(7)
    n = 200
    # Fake estimator that returns random probabilities.
    class FakeEst:
        def predict_proba(self, X):
            return rng.uniform(0, 1, (len(X), 2)) ** 0.5  # arbitrary

    X = np.zeros((n, 1))
    y = rng.integers(0, 2, n)
    levels = [0.5, 0.8, 0.9, 0.95]
    quantiles = _fit_lac_quantiles(FakeEst(), X, y, levels)
    values = [quantiles[lvl] for lvl in levels]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1)), values


def test_lac_split_conformal_marginal_coverage():
    """On synthetic data with proper split, empirical coverage matches nominal."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from src.calibration.conformal import _fit_lac_quantiles, LACConformalClassifier

    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    X_train, X_cal, X_test = X[:200], X[200:400], X[400:]
    y_train, y_cal, y_test = y[:200], y[200:400], y[400:]

    clf = LogisticRegression().fit(X_train, y_train)
    levels = [0.5, 0.8, 0.9, 0.95]
    quantiles = _fit_lac_quantiles(clf, X_cal, y_cal, levels)
    wrapper = LACConformalClassifier(
        estimator=clf,
        confidence_levels=levels,
        quantiles=quantiles,
        n_cal=len(X_cal),
    )
    _, y_pss = wrapper.predict_set(X_test)
    for li, lvl in enumerate(levels):
        in_set = y_pss[np.arange(len(y_test)), y_test, li]
        coverage = float(in_set.mean())
        # Allow ±10pp wiggle on n=200 test set; standard finite-sample noise.
        assert abs(coverage - lvl) < 0.1, f"At {lvl}: coverage {coverage}"


# ---------------------------------------------------------------------------
# cv folds
# ---------------------------------------------------------------------------


def test_stratified_cal_folds_classification_disjoint():
    """5-fold yields 5 disjoint, exhaustive eval splits over the binary target."""
    from src.calibration.cv import stratified_cal_folds

    y = np.repeat([0, 1], 50)  # 100 samples, balanced
    folds = list(stratified_cal_folds(y, task="classification", n_folds=5))
    assert len(folds) == 5
    eval_sets = [set(eval_idx.tolist()) for _, eval_idx in folds]
    union = set().union(*eval_sets)
    assert union == set(range(100))
    # Disjoint
    for i in range(5):
        for j in range(i + 1, 5):
            assert eval_sets[i].isdisjoint(eval_sets[j])
