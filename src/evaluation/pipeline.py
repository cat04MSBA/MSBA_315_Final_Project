"""Phase 8 end-to-end pipeline.

This module assembles the four-layer triage system into a single
deterministic function. The function takes one film's already-
extracted feature row (plus the parsed screenplay for scene-level
demos) and returns a structured ``TriageReport`` that combines all
four layer outputs.

Locked methodology lives in
``docs/proposals/phase8_preregistration.md`` Section 3.

Two execution modes:

* **Batch mode** — :func:`run_batch` iterates over the 257-film
  test set, loading the four-layer artifacts once and producing a
  per-film table.
* **Single-film mode** — :func:`triage_report` is the per-film
  entry point used by both batch mode and the smoke-test demos.

Determinism: identical inputs produce identical outputs. No global
mutable state.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# sklearn 1.8 deprecation noise from CalibratedClassifierCV's Platt fit.
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.decision.cost_matrix import (
    Action,
    CostMatrix,
    DEFAULT_COST_MATRIX,
)
from src.decision.rule import DecisionResult, decide_one
from src.explanation.shap_explainer import (
    TreeExplainerBundle,
    build_tree_explainer,
    shap_values,
)
from src.features.targets import add_targets
from src.models.phase4.matrices import MATRICES, build_matrix
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


HEADLINE_TARGET: str = "roi_gt_2"
SECONDARY_REGRESSION_TARGET: str = "log_roi"
ALL_CLASSIFICATION_TARGETS: tuple[str, ...] = ("roi_gt_1", "roi_gt_2")
DEFAULT_CONFIDENCE_LEVELS: list[float] = [0.50, 0.80, 0.90, 0.95]
DEFAULT_TOP_K_SHAP: int = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageReport:
    """Single-film end-to-end triage output.

    Combines Phase 4 (prediction), Phase 5 (calibrated probability +
    conformal interval), Phase 6 (cost-asymmetric action), and
    Phase 7 (SHAP attribution + per-film rationale).
    """

    imdb_id: str
    movie_name: str

    # Layer 1 — point predictions
    log_roi_point_prediction: float
    roi_gt_1_uncalibrated_probability: float
    roi_gt_2_uncalibrated_probability: float

    # Layer 2 — calibrated probability and conformal intervals
    calibrated_probability_roi_gt_2: float
    calibrated_probability_roi_gt_1: float
    log_roi_interval_at_each_level: dict[float, tuple[float, float]]
    conformal_set_roi_gt_2_at_each_level: dict[float, tuple[bool, bool]]

    # Layer 3 — decision
    recommended_action: Action
    expected_cost_per_action: dict[str, float]
    decision_rationale: str

    # Layer 4 — SHAP attribution
    top_positive_shap_contributors: list[tuple[str, float]]
    top_negative_shap_contributors: list[tuple[str, float]]
    shap_rationale: str

    # Final composed natural-language rationale: Phase 6 + Phase 7
    full_rationale: str


@dataclass
class FourLayerArtifacts:
    """Bundle of all four pre-fit layer artifacts loaded from disk.

    Loaded once at the start of a batch run; passed by reference
    to per-film calls to avoid repeated joblib loads.
    """

    # Phase 4 winners (estimators inside sklearn Pipelines)
    phase4_log_roi: dict
    phase4_roi_gt_1: dict
    phase4_roi_gt_2: dict

    # Phase 5 calibrated wrappers
    phase5_log_roi: dict
    phase5_roi_gt_1: dict
    phase5_roi_gt_2: dict

    # Phase 6 decision pipeline
    phase6_decision: dict

    # Phase 7 SHAP explainer bundle (roi_gt_2 only, per pre-reg)
    phase7_explainer_roi_gt_2: dict

    # Master corpus row (for movie name lookup, genre, etc.)
    master_corpus: pd.DataFrame

    # The cached preprocessed feature matrix for the active set
    # (test split or otherwise). Indexed by imdb_id.
    feature_matrix: pd.DataFrame

    # Pre-computed batch outputs to avoid redundant per-film calls.
    cached_shap_values: np.ndarray | None = None
    cached_shap_feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_four_layer_artifacts(
    feature_matrix: pd.DataFrame,
) -> FourLayerArtifacts:
    """Load every Phase 4-7 artifact from ``data/processed/``.

    Parameters
    ----------
    feature_matrix
        Pre-extracted feature DataFrame for whatever set will be
        evaluated (e.g. the 257-film test split). Must be indexed
        by imdb_id (string) and carry the 92 columns matching the
        ``standalone_positive_union_mpnet`` Phase 4 matrix spec.
    """
    base = paths.DATA_PROCESSED_DIR
    artifacts = FourLayerArtifacts(
        phase4_log_roi=joblib.load(base / "phase4_primary_model_log_roi.joblib"),
        phase4_roi_gt_1=joblib.load(base / "phase4_primary_model_roi_gt_1.joblib"),
        phase4_roi_gt_2=joblib.load(base / "phase4_primary_model_roi_gt_2.joblib"),
        phase5_log_roi=joblib.load(base / "phase5_calibrated_model_log_roi.joblib"),
        phase5_roi_gt_1=joblib.load(base / "phase5_calibrated_model_roi_gt_1.joblib"),
        phase5_roi_gt_2=joblib.load(base / "phase5_calibrated_model_roi_gt_2.joblib"),
        phase6_decision=joblib.load(base / "phase6_decision_pipeline_roi_gt_2.joblib"),
        phase7_explainer_roi_gt_2=joblib.load(
            base / "phase7_shap_explainer_roi_gt_2.joblib"
        ),
        master_corpus=pd.read_parquet(base / "films_joined.parquet"),
        feature_matrix=feature_matrix,
    )
    return artifacts


def build_test_feature_matrix(
    matrix_name: str = "standalone_positive_union_mpnet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the test-set feature matrix.

    Returns
    -------
    (X_test, df_test_meta)
        X_test indexed by imdb_id with 92 feature columns, plus the
        per-film meta DataFrame (with targets, genre, release year,
        budget, revenue, etc).
    """
    df_full = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    df_full = add_targets(df_full)

    test_ids = splits.loc[splits["split"] == "test", "imdb_id"].tolist()
    df_test = df_full[df_full["imdb_id"].isin(test_ids)].reset_index(drop=True)

    matrix_spec = MATRICES[matrix_name]
    X_test = build_matrix(matrix_spec, df_test)
    X_test.index = df_test["imdb_id"].astype(str).values
    return X_test, df_test


def assert_test_set_isolation() -> None:
    """Check programmatically that the test set has not been touched.

    Verifies that the test-split imdb_ids form a disjoint set from
    train + cal. This is cheap and re-runnable; it serves as a
    self-check at the head of every Phase 8 batch run.
    """
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train = set(splits.loc[splits["split"] == "train", "imdb_id"])
    cal = set(splits.loc[splits["split"] == "cal", "imdb_id"])
    test = set(splits.loc[splits["split"] == "test", "imdb_id"])
    if train & test:
        raise RuntimeError(
            f"Test-set leakage: {len(train & test)} imdb_ids in both "
            f"train and test splits.",
        )
    if cal & test:
        raise RuntimeError(
            f"Test-set leakage: {len(cal & test)} imdb_ids in both "
            f"cal and test splits.",
        )
    if not (len(train) == 1199 and len(cal) == 257 and len(test) == 257):
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train)}, cal={len(cal)}, "
            f"test={len(test)}; expected 1199/257/257.",
        )
    logger.info(
        "Test-set isolation verified: 1199 train + 257 cal + 257 test, "
        "all disjoint.",
    )


# ---------------------------------------------------------------------------
# Per-film prediction helpers
# ---------------------------------------------------------------------------


def _calibrated_probability(
    bundle: dict, X_row: pd.DataFrame,
) -> float:
    """Apply the Phase 5 deployed probability calibrator.

    For classification targets the deployed ``probability_calibrator``
    is a CalibratedClassifierCV wrapping FrozenEstimator(phase4
    pipeline). For SVM-RBF (no native predict_proba) the
    calibrator is what makes probability access possible at all.
    """
    cal = bundle["probability_calibrator"]
    if cal is None:
        # Fall back to underlying estimator if for some reason the
        # calibrator wasn't deployed (regression target arrives here).
        phase4 = joblib.load(bundle["phase4_winner_path"])
        return float(phase4["estimator"].predict_proba(X_row)[:, 1][0])
    return float(cal.predict_proba(X_row)[:, 1][0])


def _conformal_classification_set(
    bundle: dict, X_row: pd.DataFrame,
    levels: list[float],
) -> dict[float, tuple[bool, bool]]:
    """Run the LAC conformal wrapper to produce per-level prediction sets.

    Returns ``{level: (label_0_in_set, label_1_in_set)}`` per the
    pre-registration's set-based output schema. Both False means
    the empty set (very rare); both True means Refer.
    """
    wrapper = bundle["conformal_wrapper"]
    # The deployed wrapper carries its own ``confidence_levels`` list;
    # we re-key the output to whatever ``levels`` the caller asked for,
    # filtered to the deployed levels.
    deployed_levels = wrapper.confidence_levels
    _, y_pss = wrapper.predict_set(X_row)  # (1, 2, n_levels)
    out: dict[float, tuple[bool, bool]] = {}
    for level in levels:
        if level in deployed_levels:
            li = deployed_levels.index(level)
            out[level] = (bool(y_pss[0, 0, li]), bool(y_pss[0, 1, li]))
    return out


def _conformal_regression_interval(
    bundle: dict, X_row: pd.DataFrame,
    levels: list[float],
) -> tuple[float, dict[float, tuple[float, float]]]:
    """Apply MAPIE SplitConformalRegressor to produce per-level intervals.

    Returns (point_prediction, {level: (lower, upper)}). The deployed
    levels come from the parent bundle's ``deployed_confidence_levels``
    field; MAPIE's ``SplitConformalRegressor`` does not expose them as
    a public attribute.
    """
    wrapper = bundle["conformal_wrapper"]
    deployed_levels = bundle["deployed_confidence_levels"]
    point, y_pis = wrapper.predict_interval(X_row)  # (1,), (1, 2, n_levels)
    intervals: dict[float, tuple[float, float]] = {}
    for level in levels:
        if level in deployed_levels:
            li = deployed_levels.index(level)
            intervals[level] = (float(y_pis[0, 0, li]), float(y_pis[0, 1, li]))
    return float(point[0]), intervals


def _shap_for_row(
    artifacts: FourLayerArtifacts,
    X_row: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Compute or look up SHAP values for one row on the headline target."""
    p7 = artifacts.phase7_explainer_roi_gt_2
    explainer = p7["explainer"]
    feature_names = p7["feature_names"]

    # Apply the Phase 4 pipeline's preprocessing manually, then call
    # the explainer on the preprocessed array (same convention as
    # ``shap_explainer.shap_values``).
    pipeline = artifacts.phase4_roi_gt_2["estimator"]
    pre = pipeline.named_steps.get("pre")
    X_pre = np.asarray(pre.transform(X_row)) if pre is not None else X_row.values

    raw = explainer.shap_values(X_pre)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # RandomForestClassifier slice; XGBoost is 2-D already
        arr = arr[:, :, 1]
    return arr[0], feature_names


def _format_shap_effect(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.3f}"


def _readable_feature(name: str) -> str:
    """Cosmetic improvement on raw feature names (mirrors per_film._readable_feature)."""
    if name.startswith("genre_"):
        return f"Genre={name.removeprefix('genre_')}"
    if name.startswith("topic_") and name.endswith("_proportion"):
        return f"Topic {name.removeprefix('topic_').removesuffix('_proportion')} proportion"
    if name.startswith("network_"):
        return name.replace("network_", "Network: ")
    if name.startswith("embed_pc_"):
        return f"Embedding PC {name.removeprefix('embed_pc_')}"
    if name.startswith("log_"):
        return f"log({name.removeprefix('log_')})"
    return name


def _build_shap_rationale(
    positives: list[tuple[str, float]],
    negatives: list[tuple[str, float]],
) -> str:
    """Compose the Phase 7-style rationale fragment for the report."""
    parts: list[str] = []
    if positives:
        s = ", ".join(
            f"{_readable_feature(n)} ({_format_shap_effect(v)} log-odds)"
            for n, v in positives[:3]
        )
        parts.append(f"Top features pushing probability up: {s}")
    if negatives:
        s = ", ".join(
            f"{_readable_feature(n)} ({_format_shap_effect(v)} log-odds)"
            for n, v in negatives[:3]
        )
        parts.append(f"Top features pulling probability down: {s}")
    return ". ".join(parts) + "." if parts else "No notable feature contributions."


def _movie_name(master: pd.DataFrame, imdb_id: str) -> str:
    sub = master.loc[master["imdb_id"] == imdb_id]
    if sub.empty:
        return imdb_id
    name = sub["movie_name"].iloc[0]
    if not isinstance(name, str) or not name:
        return imdb_id
    return name


# ---------------------------------------------------------------------------
# The end-to-end function
# ---------------------------------------------------------------------------


def triage_report(
    *,
    imdb_id: str,
    artifacts: FourLayerArtifacts,
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
    confidence_levels: list[float] | None = None,
    top_k_shap: int = DEFAULT_TOP_K_SHAP,
) -> TriageReport:
    """Apply all four layers to one film and return a ``TriageReport``.

    The film must be present in ``artifacts.feature_matrix.index``;
    the function does not re-extract features from raw screenplays.
    Doing the extraction here would couple the function to the
    Phase 3 feature pipeline; the project's deployable surface is
    "given a feature row, return a triage report."
    """
    confidence_levels = confidence_levels or DEFAULT_CONFIDENCE_LEVELS
    if imdb_id not in artifacts.feature_matrix.index:
        raise KeyError(f"imdb_id {imdb_id!r} not in feature matrix")

    X_row = artifacts.feature_matrix.loc[[imdb_id]]

    # ------------------------------------------------------------------
    # Layer 1 — point predictions (uncalibrated)
    # ------------------------------------------------------------------
    log_roi_point = float(
        artifacts.phase4_log_roi["estimator"].predict(X_row)[0],
    )
    # roi_gt_1 / roi_gt_2 uncalibrated probabilities. SVM-RBF (roi_gt_1
    # winner) lacks native predict_proba; use the Phase 5 calibrator as
    # the only path to probability there. Document the column as
    # "calibrated" implicitly for that target.
    p4_roi_gt_2 = artifacts.phase4_roi_gt_2["estimator"]
    if hasattr(p4_roi_gt_2, "predict_proba"):
        roi_gt_2_uncal = float(p4_roi_gt_2.predict_proba(X_row)[:, 1][0])
    else:
        roi_gt_2_uncal = float("nan")

    p4_roi_gt_1 = artifacts.phase4_roi_gt_1["estimator"]
    if hasattr(p4_roi_gt_1, "predict_proba"):
        roi_gt_1_uncal = float(p4_roi_gt_1.predict_proba(X_row)[:, 1][0])
    else:
        # SVM-RBF: no native predict_proba.
        roi_gt_1_uncal = float("nan")

    # ------------------------------------------------------------------
    # Layer 2 — calibrated probability + conformal intervals
    # ------------------------------------------------------------------
    p_roi_gt_2 = _calibrated_probability(artifacts.phase5_roi_gt_2, X_row)
    p_roi_gt_1 = _calibrated_probability(artifacts.phase5_roi_gt_1, X_row)

    conformal_set_roi_gt_2 = _conformal_classification_set(
        artifacts.phase5_roi_gt_2, X_row, confidence_levels,
    )
    log_roi_point_from_conf, log_roi_intervals = _conformal_regression_interval(
        artifacts.phase5_log_roi, X_row, confidence_levels,
    )

    # ------------------------------------------------------------------
    # Layer 3 — cost-asymmetric decision
    # ------------------------------------------------------------------
    decision = decide_one(imdb_id, p_roi_gt_2, cost_matrix)

    # ------------------------------------------------------------------
    # Layer 4 — SHAP attribution
    # ------------------------------------------------------------------
    sv, feat_names = _shap_for_row(artifacts, X_row)
    order = np.argsort(-np.abs(sv))
    positives: list[tuple[str, float]] = []
    negatives: list[tuple[str, float]] = []
    for idx in order:
        v = float(sv[idx])
        if v > 0 and len(positives) < top_k_shap:
            positives.append((feat_names[idx], v))
        elif v < 0 and len(negatives) < top_k_shap:
            negatives.append((feat_names[idx], v))
        if len(positives) >= top_k_shap and len(negatives) >= top_k_shap:
            break

    shap_rat = _build_shap_rationale(positives, negatives)

    # Compose the full natural-language rationale (Phase 6 + Phase 7).
    full_rat = decision.rationale + " " + shap_rat

    return TriageReport(
        imdb_id=imdb_id,
        movie_name=_movie_name(artifacts.master_corpus, imdb_id),
        log_roi_point_prediction=log_roi_point,
        roi_gt_1_uncalibrated_probability=roi_gt_1_uncal,
        roi_gt_2_uncalibrated_probability=roi_gt_2_uncal,
        calibrated_probability_roi_gt_2=p_roi_gt_2,
        calibrated_probability_roi_gt_1=p_roi_gt_1,
        log_roi_interval_at_each_level=log_roi_intervals,
        conformal_set_roi_gt_2_at_each_level=conformal_set_roi_gt_2,
        recommended_action=decision.recommended_action,
        expected_cost_per_action={
            "Greenlight": decision.expected_cost_greenlight,
            "Pass": decision.expected_cost_pass,
            "Refer": decision.expected_cost_refer,
        },
        decision_rationale=decision.rationale,
        top_positive_shap_contributors=positives,
        top_negative_shap_contributors=negatives,
        shap_rationale=shap_rat,
        full_rationale=full_rat,
    )


def run_batch(
    artifacts: FourLayerArtifacts,
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
    confidence_levels: list[float] | None = None,
    top_k_shap: int = DEFAULT_TOP_K_SHAP,
) -> pd.DataFrame:
    """Apply the end-to-end pipeline to every film in ``artifacts.feature_matrix``.

    Returns a DataFrame with one row per film and columns covering
    every layer's output.
    """
    confidence_levels = confidence_levels or DEFAULT_CONFIDENCE_LEVELS
    rows: list[dict] = []
    n = len(artifacts.feature_matrix)
    logger.info("Running end-to-end pipeline on %d films", n)
    for i, imdb_id in enumerate(artifacts.feature_matrix.index, start=1):
        report = triage_report(
            imdb_id=imdb_id,
            artifacts=artifacts,
            cost_matrix=cost_matrix,
            confidence_levels=confidence_levels,
            top_k_shap=top_k_shap,
        )
        row = _flatten_report(report)
        rows.append(row)
        if i % 50 == 0 or i == n:
            logger.info("  triage report progress: %d/%d", i, n)
    return pd.DataFrame(rows)


def _flatten_report(report: TriageReport) -> dict:
    """Flatten a TriageReport into a single row of strings/floats for CSV."""
    out: dict[str, Any] = {
        "imdb_id": report.imdb_id,
        "movie_name": report.movie_name,
        "log_roi_point_prediction": report.log_roi_point_prediction,
        "roi_gt_1_uncalibrated_probability": report.roi_gt_1_uncalibrated_probability,
        "roi_gt_2_uncalibrated_probability": report.roi_gt_2_uncalibrated_probability,
        "calibrated_probability_roi_gt_1": report.calibrated_probability_roi_gt_1,
        "calibrated_probability_roi_gt_2": report.calibrated_probability_roi_gt_2,
        "recommended_action": report.recommended_action,
        "expected_cost_greenlight": report.expected_cost_per_action["Greenlight"],
        "expected_cost_pass": report.expected_cost_per_action["Pass"],
        "expected_cost_refer": report.expected_cost_per_action["Refer"],
        "decision_rationale": report.decision_rationale,
        "shap_rationale": report.shap_rationale,
        "full_rationale": report.full_rationale,
    }
    for level, (lo, hi) in report.log_roi_interval_at_each_level.items():
        out[f"log_roi_lower_{level}"] = lo
        out[f"log_roi_upper_{level}"] = hi
    for level, (l0, l1) in report.conformal_set_roi_gt_2_at_each_level.items():
        out[f"conf_roi_gt_2_class0_in_set_{level}"] = l0
        out[f"conf_roi_gt_2_class1_in_set_{level}"] = l1
        # set_size derived for convenience
        out[f"conf_roi_gt_2_set_size_{level}"] = int(l0) + int(l1)
    out["top_positive_shap"] = "; ".join(
        f"{n}={v:+.3f}" for n, v in report.top_positive_shap_contributors
    )
    out["top_negative_shap"] = "; ".join(
        f"{n}={v:+.3f}" for n, v in report.top_negative_shap_contributors
    )
    return out


# ---------------------------------------------------------------------------
# Smoke test for end-to-end consistency
# ---------------------------------------------------------------------------


def smoke_test_consistency(
    artifacts: FourLayerArtifacts, n_films: int = 3,
) -> dict[str, Any]:
    """Re-run the per-film triage on a small subset and confirm
    determinism (identical outputs across two calls).

    The pre-registration's escalation trigger #5 ("end-to-end
    smoke-test mismatch") is implemented here. Returns a dict with
    pass/fail and the per-film comparison.
    """
    sample = list(artifacts.feature_matrix.index[:n_films])
    first_pass = [triage_report(imdb_id=iid, artifacts=artifacts) for iid in sample]
    second_pass = [triage_report(imdb_id=iid, artifacts=artifacts) for iid in sample]

    mismatches: list[str] = []
    for r1, r2 in zip(first_pass, second_pass):
        if (
            abs(r1.calibrated_probability_roi_gt_2 - r2.calibrated_probability_roi_gt_2)
            > 1e-9
            or r1.recommended_action != r2.recommended_action
        ):
            mismatches.append(r1.imdb_id)
    return {
        "n_films_checked": n_films,
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }
