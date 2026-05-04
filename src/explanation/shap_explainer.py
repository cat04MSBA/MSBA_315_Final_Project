"""TreeSHAP wrappers for the supported tree-based winners.

Two flavors:

* :func:`build_tree_explainer_xgboost` — wraps an XGBoost classifier,
  returning a callable that produces SHAP values in (n_samples,
  n_features) shape.
* :func:`build_tree_explainer_random_forest` — wraps a sklearn
  RandomForest, returning the (n_samples, n_features) array
  (positive class for classification; the regression target's
  values for regression).

Both expect the **fitted final estimator** (not a sklearn Pipeline);
the caller is responsible for transforming features through the
pipeline's preprocessing branch first. The phase4 winner bundles
contain both the full pipeline and the fitted final ``model``
(the booster); we extract the ``model`` for SHAP and apply the
``pre`` ColumnTransformer manually.

The ``feature_perturbation`` parameter controls TreeSHAP's
conditioning assumption:

* ``"tree_path_dependent"`` (default): conditions on the tree's
  decision path; fast and the standard.
* ``"interventional"``: requires a background dataset; matches the
  unbiased estimator from Lundberg et al. 2020. Used in stability
  validation per Section 7 of the pre-registration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


FeaturePerturbation = Literal["tree_path_dependent", "interventional"]


@dataclass(frozen=True)
class TreeExplainerBundle:
    """Fitted TreeExplainer + the metadata needed to use it."""
    explainer: object  # shap.TreeExplainer
    feature_perturbation: FeaturePerturbation
    feature_names: list[str]
    target: str
    family: str  # "xgboost" or "random_forest"
    base_value: float  # explainer.expected_value scalar (binary class 1)
    is_classification: bool


def _extract_final_estimator(pipeline: Pipeline) -> BaseEstimator:
    """Pull out the last (model) step from a sklearn Pipeline."""
    if not isinstance(pipeline, Pipeline):
        return pipeline
    return pipeline.named_steps["model"]


def _transform_through_preprocessing(
    pipeline: Pipeline, X: pd.DataFrame,
) -> np.ndarray:
    """Apply the pipeline's pre-step ColumnTransformer to X."""
    if not isinstance(pipeline, Pipeline):
        return X.values
    pre = pipeline.named_steps.get("pre")
    if pre is None:
        return X.values
    return np.asarray(pre.transform(X))


def build_tree_explainer(
    pipeline: Pipeline,
    family: str,
    target: str,
    X_for_background: pd.DataFrame | None = None,
    feature_perturbation: FeaturePerturbation = "tree_path_dependent",
) -> TreeExplainerBundle:
    """Construct a TreeExplainer for the family's pipeline.

    Parameters
    ----------
    pipeline
        The sklearn Pipeline from the Phase 4 winner bundle. Must
        have a ``model`` step (XGBoost or RandomForest).
    family
        ``"xgboost"`` or ``"random_forest"``.
    target
        Target name (for tagging the bundle).
    X_for_background
        Required when ``feature_perturbation="interventional"``. A
        DataFrame whose preprocessed values become the background
        dataset for the unbiased estimator. Pass the calibration
        set's feature matrix.
    feature_perturbation
        ``"tree_path_dependent"`` (default, fast) or
        ``"interventional"`` (slower, uses the background dataset).
    """
    if family not in ("xgboost", "random_forest"):
        raise ValueError(
            f"TreeSHAP supports xgboost and random_forest only, not {family!r}",
        )
    final = _extract_final_estimator(pipeline)
    is_classification = hasattr(final, "predict_proba")

    explainer_kwargs: dict = {}
    if feature_perturbation == "interventional":
        if X_for_background is None:
            raise ValueError(
                "feature_perturbation='interventional' requires X_for_background",
            )
        bg = _transform_through_preprocessing(pipeline, X_for_background)
        explainer_kwargs["data"] = bg
        explainer_kwargs["feature_perturbation"] = "interventional"
    else:
        explainer_kwargs["feature_perturbation"] = "tree_path_dependent"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(final, **explainer_kwargs)

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected_arr = np.atleast_1d(np.asarray(expected))
        base_value = float(expected_arr[-1])  # positive class for binary
    else:
        base_value = float(expected)

    feature_names = list(X_for_background.columns) if X_for_background is not None else []

    return TreeExplainerBundle(
        explainer=explainer,
        feature_perturbation=feature_perturbation,
        feature_names=feature_names,
        target=target,
        family=family,
        base_value=base_value,
        is_classification=is_classification,
    )


def shap_values(
    bundle: TreeExplainerBundle,
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values; return shape ``(n_samples, n_features)``.

    For RandomForestClassifier the raw SHAP output is
    ``(n_samples, n_features, n_classes)``; we slice the positive
    class. XGBoost returns ``(n_samples, n_features)`` directly.
    """
    X_pre = _transform_through_preprocessing(pipeline, X)
    raw = bundle.explainer.shap_values(X_pre)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # RandomForestClassifier: (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr
