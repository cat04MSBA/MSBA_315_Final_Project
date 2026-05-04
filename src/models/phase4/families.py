"""Phase 4 model family registry.

Each family is a :class:`FamilySpec` that holds the per-task estimator
factory, the hyperparameter grid for ``GridSearchCV``, the score method
used for classification probability extraction, and the class-balancing
policy.

The registry is the locked roster from
``docs/proposals/phase4_preregistration.md`` Section 3 with the grids
from Section 4. Six families total, partitioned into a primary tier
(four families that participate in the headline benchmark and the
paired Bayesian comparison) and a secondary tier (two families run on
``all_five`` only for breadth and the linear-kernel sanity check).

Three deviations from the brief are baked in here:

* HistGB is the gradient-boosting representative (not LightGBM /
  XGBoost) for Phase 3 longitudinal continuity on the train-OOF gap
  finding.
* Random Forest occupies the slot the brief allocated to XGBoost,
  because RF's bagging-with-deep-trees inductive bias is genuinely
  different from any boosting variant.
* All classifiers use ``class_weight="balanced"`` (Logistic, RF, SVMs)
  or equivalent ``sample_weight`` (HistGB) per Section 7.

The registry exposes ``primary_tier()`` and ``secondary_tier()`` so the
benchmark orchestrator iterates without hard-coding family names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

# LightGBM and XGBoost are added in the user-requested follow-up to
# the original four-family roster. Both are alternative gradient-
# boosting frameworks chosen for cross-framework diversity per the
# Phase 4 brief Section 2's original recommendation; HistGB stays in
# the registry alongside them so the Phase 3 longitudinal comparison
# on the train-OOF gap is preserved.
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


SEED: int = 42

ModelTask = Literal["regression", "classification"]
BalancingMode = Literal["class_weight", "sample_weight", "none"]
ScoreMethod = Literal["predict_proba", "decision_function"]
TierName = Literal["primary", "secondary"]


@dataclass(frozen=True)
class FamilySpec:
    """Specification for one model family in the Phase 4 benchmark.

    Attributes
    ----------
    name
        Stable identifier used in CSV outputs and run-directory naming.
    tier
        ``"primary"`` (full benchmark + paired tests) or ``"secondary"``
        (smaller search, ``all_five`` matrix only, no paired tests).
    needs_scaling
        ``True`` if the numeric branch needs imputation + standard
        scaling (linear, SVM); ``False`` for tree models (HistGB, RF).
    balancing
        How class imbalance is handled. ``"class_weight"`` for families
        that accept the kwarg directly; ``"sample_weight"`` for HistGB
        (passed via ``fit_params`` through ``GridSearchCV``);
        ``"none"`` for regression-only contexts.
    regressor_factory, classifier_factory
        Zero-arg callables returning a fresh estimator for each task.
        All non-grid hyperparameters are set on the returned estimator;
        grid hyperparameters are tuned via ``GridSearchCV``.
    regression_grid, classification_grid
        ``GridSearchCV`` parameter grids. Keys are estimator parameter
        names without the pipeline prefix; the search wrapper prefixes
        with ``"model__"`` when the estimator is wrapped in a Pipeline.
    score_method
        The sklearn method name used to obtain probability-style scores
        for classification. ``"predict_proba"`` for probabilistic
        families; ``"decision_function"`` for SVM families that emit
        signed margins. Hard predictions always come from ``predict``.
    """

    name: str
    tier: TierName
    needs_scaling: bool
    balancing: BalancingMode
    regressor_factory: Callable[[], BaseEstimator]
    classifier_factory: Callable[[], BaseEstimator]
    regression_grid: dict[str, list]
    classification_grid: dict[str, list]
    score_method: ScoreMethod


FAMILIES: dict[str, FamilySpec] = {
    "linear": FamilySpec(
        name="linear",
        tier="primary",
        needs_scaling=True,
        balancing="class_weight",
        regressor_factory=lambda: Ridge(random_state=SEED),
        # sklearn 1.8 deprecated explicit ``penalty="l2"`` in favor of
        # leaving the default. Default is still L2; behavior unchanged.
        classifier_factory=lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        regression_grid={"alpha": list(np.logspace(-3, 3, 13))},
        classification_grid={"C": list(np.logspace(-3, 3, 13))},
        score_method="predict_proba",
    ),
    "histgb": FamilySpec(
        name="histgb",
        tier="primary",
        needs_scaling=False,
        balancing="sample_weight",
        regressor_factory=lambda: HistGradientBoostingRegressor(
            max_iter=200,
            early_stopping=True,
            random_state=SEED,
        ),
        classifier_factory=lambda: HistGradientBoostingClassifier(
            max_iter=200,
            early_stopping=True,
            random_state=SEED,
        ),
        regression_grid={
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.02, 0.05],
            "min_samples_leaf": [10, 20, 40, 80],
        },
        classification_grid={
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.02, 0.05],
            "min_samples_leaf": [10, 20, 40, 80],
        },
        score_method="predict_proba",
    ),
    "random_forest": FamilySpec(
        name="random_forest",
        tier="primary",
        needs_scaling=False,
        balancing="class_weight",
        regressor_factory=lambda: RandomForestRegressor(
            random_state=SEED,
            n_jobs=1,
        ),
        classifier_factory=lambda: RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        ),
        regression_grid={
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 5, 20],
        },
        classification_grid={
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 5, 20],
        },
        score_method="predict_proba",
    ),
    "svm_rbf": FamilySpec(
        name="svm_rbf",
        tier="primary",
        needs_scaling=True,
        balancing="class_weight",
        regressor_factory=lambda: SVR(kernel="rbf"),
        classifier_factory=lambda: SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=SEED,
        ),
        regression_grid={
            "C": [0.1, 0.3, 1, 3, 10, 30],
            "gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
        },
        classification_grid={
            "C": [0.1, 0.3, 1, 3, 10, 30],
            "gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
        },
        score_method="decision_function",
    ),
    # Phase 4 follow-up: LightGBM as an alternative gradient-boosting
    # framework. Same conservative regularization corner as HistGB
    # plus LightGBM's native L1/L2 reg knobs.
    "lightgbm": FamilySpec(
        name="lightgbm",
        tier="primary",
        needs_scaling=False,
        balancing="class_weight",
        regressor_factory=lambda: LGBMRegressor(
            n_estimators=300,
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ),
        classifier_factory=lambda: LGBMClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ),
        regression_grid={
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [15, 31],
            "min_child_samples": [10, 30],
        },
        classification_grid={
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [15, 31],
            "min_child_samples": [10, 30],
        },
        score_method="predict_proba",
    ),
    # Phase 4 follow-up: XGBoost. Different regularization formulation
    # than HistGB / LightGBM (per-leaf reg_alpha + reg_lambda); often
    # diverges on small corpora.
    "xgboost": FamilySpec(
        name="xgboost",
        tier="primary",
        needs_scaling=False,
        balancing="sample_weight",
        regressor_factory=lambda: XGBRegressor(
            n_estimators=300,
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        classifier_factory=lambda: XGBClassifier(
            n_estimators=300,
            tree_method="hist",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        regression_grid={
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_lambda": [1.0, 10.0],
            "min_child_weight": [1, 5],
        },
        classification_grid={
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_lambda": [1.0, 10.0],
            "min_child_weight": [1, 5],
        },
        score_method="predict_proba",
    ),
    "lasso": FamilySpec(
        name="lasso",
        tier="secondary",
        needs_scaling=True,
        balancing="class_weight",
        regressor_factory=lambda: Lasso(random_state=SEED, max_iter=10000),
        classifier_factory=lambda: LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        regression_grid={"alpha": list(np.logspace(-3, 1, 9))},
        classification_grid={"C": list(np.logspace(-2, 2, 9))},
        score_method="predict_proba",
    ),
    "linear_svm": FamilySpec(
        name="linear_svm",
        tier="secondary",
        needs_scaling=True,
        balancing="class_weight",
        # ``LinearSVR`` does not accept ``random_state``; the dual-coordinate
        # descent solver is deterministic for a given input ordering.
        regressor_factory=lambda: LinearSVR(max_iter=10000),
        classifier_factory=lambda: LinearSVC(
            class_weight="balanced",
            random_state=SEED,
            max_iter=10000,
        ),
        regression_grid={"C": [0.01, 0.1, 1, 10]},
        classification_grid={"C": [0.01, 0.1, 1, 10]},
        score_method="decision_function",
    ),
}


def primary_tier() -> list[FamilySpec]:
    """Return the four families in the primary tier (paired-test scope)."""
    return [f for f in FAMILIES.values() if f.tier == "primary"]


def secondary_tier() -> list[FamilySpec]:
    """Return the two families in the secondary tier."""
    return [f for f in FAMILIES.values() if f.tier == "secondary"]


def get_family(name: str) -> FamilySpec:
    """Look up a family by name. Raises ``KeyError`` on miss."""
    if name not in FAMILIES:
        raise KeyError(
            f"Unknown family {name!r}; known: {sorted(FAMILIES)!r}"
        )
    return FAMILIES[name]


def grid_size(spec: FamilySpec, task: ModelTask) -> int:
    """Number of hyperparameter cells in a family's grid for a task."""
    grid = spec.regression_grid if task == "regression" else spec.classification_grid
    n = 1
    for v in grid.values():
        n *= len(v)
    return n
