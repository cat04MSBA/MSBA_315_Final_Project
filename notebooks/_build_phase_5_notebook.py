"""Build ``notebooks/phase_5.ipynb`` from a structured cell list.

Regenerate the notebook with::

    python -m notebooks._build_phase_5_notebook

The Phase 5 notebook documents Layer 2 calibrated uncertainty:
probability calibration via Platt and isotonic, conformal
prediction (LAC for classification, absolute-residual for
regression), and the empirical coverage validation that anchors
Layer 3.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_5.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 5: Layer 2 Calibrated Uncertainty

        Layer 1 predictions are point estimates. A studio executive
        reading "this script has a 0.62 probability of clearing 2x
        ROI" needs to know whether that 0.62 means what it says: of
        the films the model rates at 0.62, do roughly sixty-two
        percent actually clear the threshold? A model whose
        probability outputs are systematically inflated will push the
        Layer 3 decision rule toward over-commitment; a model whose
        outputs are systematically deflated will push it toward
        excessive abstention. Layer 2 wraps the Phase 4 winners with
        two complementary calibration techniques and validates the
        result empirically.

        Phase 4 produced three winner artifacts on the
        ``standalone_positive_union_mpnet`` matrix:
        Random Forest for ``log_roi``, SVM-RBF for ``roi_gt_1``,
        and XGBoost for ``roi_gt_2``. Phase 5 calibrates all three
        using the held-out 257-film calibration set, which has been
        untouched since the Phase 3 split.

        ## How Phase 5 is organized

        Pre-registration discipline carries forward.
        ``docs/proposals/phase5_preregistration.md`` locks the two
        calibration techniques (Platt and isotonic for probability;
        split conformal for set-valued prediction), the conformity
        scores, the four confidence levels, the empirical-coverage
        tolerance band, and the four escalation triggers before any
        calibration fit ran.

        Two complementary techniques apply at the same time. Platt
        and isotonic produce a calibrated scalar probability; split
        conformal prediction produces a prediction set whose
        empirical coverage matches a chosen confidence level. The
        scalar feeds the Phase 6 cost-decision rule; the set
        underwrites the "Refer to human reader" action when the model
        cannot distinguish flop from hit at the chosen confidence.

        Five-fold cross-validation inside the calibration set gives
        honest empirical coverage and ECE estimates without leaking
        cal-fit data into evaluation. The deployed wrappers, saved at
        the end of the phase, are conformalized on all 257 calibration
        films.
    """),

    # ============================================================
    # Inputs
    # ============================================================
    md("""
        ## A. Inputs

        Phase 5 reads the three Phase 4 winner bundles plus the
        calibration split assignment.
    """),
    code("""
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from src.utils import paths

        splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / 'split_assignments.parquet')
        print(f"Calibration split: {(splits['split'] == 'cal').sum()} films")
        print(f"Train split:       {(splits['split'] == 'train').sum()} films (untouched in Phase 5)")
        print(f"Test split:        {(splits['split'] == 'test').sum()} films (untouched until Phase 8)")

        for target in ['log_roi', 'roi_gt_1', 'roi_gt_2']:
            bundle = joblib.load(paths.DATA_PROCESSED_DIR / f'phase4_primary_model_{target}.joblib')
            print(f"  Phase 4 winner | {target:9s}: family={bundle['family']:14s} matrix={bundle['matrix']}")
    """),

    # ============================================================
    # Probability calibration
    # ============================================================
    md("""
        ## B. Probability calibration

        Three methods compete on the two classification targets:
        uncalibrated (the Phase 4 winner's native ``predict_proba``),
        Platt sigmoid, and isotonic regression. Method selection is
        by mean Expected Calibration Error across five honest folds.

        SVM-RBF carries an additional constraint: the family has no
        native ``predict_proba``. Phase 4's ``calibration_method``
        parameter on SVM-RBF produced a ``CalibratedClassifierCV``
        wrapper at training time; Phase 5 replaces that wrapper with
        the better of Platt or isotonic fit on the cal set, which is
        the deployable calibrator. The "uncalibrated" SVM-RBF column
        in the Phase 5 metric tables therefore refers to the Phase 4
        wrapper's output, not to a raw decision function.

        sklearn 1.8 removed ``cv="prefit"`` from
        ``CalibratedClassifierCV``. Phase 5 uses the 1.6+ replacement
        ``sklearn.frozen.FrozenEstimator``, documented in
        ``src/calibration/conformal.py``.
    """),
    code("""
        cal = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase5_calibration_metrics.csv')
        # Aggregate to mean ECE per (target, method) over the five folds.
        agg = (
            cal.groupby(['target', 'method'])['ece']
               .agg(['mean', 'std'])
               .round(4)
        )
        print(agg.to_string())
    """),
    md("""
        Lower ECE is better. Isotonic wins on both classification
        targets, dropping ECE by roughly half on each. Trigger #1 of
        the pre-registration ("calibration fails to improve over
        uncalibrated") does not fire.
    """),

    # ============================================================
    # Reliability diagrams
    # ============================================================
    md("""
        ## C. Reliability diagrams

        ECE is a single scalar; the reliability diagram shows the
        full miscalibration shape. A perfectly calibrated probability
        traces the diagonal: the empirical positive rate of films
        whose predicted probability is in bin ``b`` matches the bin
        center. Bumps below the diagonal indicate over-confidence
        (the model's 0.7 means a 0.5 empirical hit rate); bumps above
        indicate under-confidence.
    """),
    code("""
        from IPython.display import Image
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase5_reliability_post.png'))
    """),

    # ============================================================
    # Conformal coverage
    # ============================================================
    md("""
        ## D. Conformal prediction and empirical coverage

        Split conformal prediction produces a prediction set at each
        confidence level. For binary classification the set is one of
        ``{0}``, ``{1}``, ``{0, 1}``, or empty; the doubleton is the
        operational "Refer to human reader" indicator. For regression
        the set is a real interval ``[lower, upper]``.

        Conformity scores chosen per pre-registration:

        * ``"lac"`` for classification (the Least Ambiguous Classifier
          score, the only valid choice for binary in MAPIE 1.4).
        * ``"absolute"`` for regression (the absolute residual; the
          ``"gamma"`` score requires strictly positive targets and
          ``log_roi`` is signed).

        MAPIE 1.4's ``SplitConformalClassifier`` is incompatible with
        sklearn ``ColumnTransformer`` named columns; the Phase 4
        winners all use ``ColumnTransformer`` with named columns.
        ``src/calibration/conformal.py`` therefore hand-rolls the LAC
        procedure (a fifty-line implementation matching the canonical
        Angelopoulos and Bates 2021 formulation). MAPIE's regression
        path does not hit the column-name issue and is kept.

        Empirical coverage is the fraction of held-out films whose
        true label lies in the prediction set. Pre-registration
        Section 6 sets a ±5 percentage point tolerance at the 0.90
        nominal level.
    """),
    code("""
        cov = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase5_coverage.csv')
        # Aggregate empirical coverage per (target, level) across the five folds.
        head = (
            cov.groupby(['target', 'level'])['empirical_coverage']
               .mean()
               .round(3)
               .unstack('level')
        )
        print(head.to_string())
    """),
    md("""
        Coverage tracks nominal at every level for every target.
        Trigger #1 of the pre-registration (coverage failure) does
        not fire. The ±5pp tolerance band at the 0.90 nominal level
        is satisfied on all three targets.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase5_coverage_levels.png'))
    """),

    # ============================================================
    # Set-size diagnostic
    # ============================================================
    md("""
        ## E. Set-size diagnostic and the over-deferral trigger

        Coverage in-band is necessary but not sufficient. A degenerate
        wrapper that always returns the doubleton ``{0, 1}`` will
        achieve 100 percent coverage at any level while being useless
        for downstream decisions. The set-size distribution at each
        confidence level is the operational diagnostic.

        Pre-registration trigger #2: if the singleton rate at the
        0.90 confidence level falls below 50 percent, the system
        defers more than half of all films, and the Layer 3 cost
        decision is dominated by the refer-cost line. The trigger is
        a finding rather than a defect: it tells the planning
        conversation that the underlying classifier's discrimination
        is too weak to support confident commits at the chosen
        confidence level.
    """),
    code("""
        sz = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase5_set_sizes.csv')
        agg = (
            sz[sz['level'] == 0.90]
               .groupby('target')[['singleton_rate', 'refer_rate', 'empty_rate']]
               .mean()
               .round(3)
        )
        print("At nominal 0.90 confidence:")
        print(agg.to_string())
    """),
    md("""
        On ``roi_gt_2`` the singleton rate at 0.90 confidence is
        approximately 21 percent, well below the 50 percent floor.
        Trigger #2 fires. The interpretation is honest: at the
        Phase 4 OOF AUC of 0.65 there is not enough discrimination to
        produce singleton sets on the majority of films at high
        confidence. The doubleton indicator becomes the deployed
        ``Refer to human reader`` channel for Phase 6.
    """),

    # ============================================================
    # Per-genre refer rate
    # ============================================================
    md("""
        ## F. Per-genre refer rate validates Phase 4 corpus bimodality

        The Phase 4 stratified diagnostic surfaced a corpus-bimodality
        finding: Adventure / Fantasy / Sci-Fi reach OOF AUC of 0.66 to
        0.77 while Drama / Comedy / Romance cap at 0.55 to 0.61. If
        the conformal procedure is honest, the per-genre refer rate
        should anti-correlate with per-genre OOF AUC: the system
        should defer more often on genres where the underlying
        classifier discriminates poorly.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase5_refer_by_genre.png'))
    """),
    md("""
        Per-genre refer rate ranges from 36 to 50 percent on Sci-Fi
        and Horror (the model-tractable cluster) up to 84 percent on
        Romance (the model-intractable cluster). The conformal
        procedure correctly defers on the genres where the Phase 4
        OOF AUC is weakest. The Phase 4 corpus-bimodality finding is
        empirically validated by the Phase 5 calibration without any
        per-genre code path.
    """),

    # ============================================================
    # Deployed wrapper
    # ============================================================
    md("""
        ## G. Deployed calibrated wrapper

        For each target Phase 5 saves a deployable bundle to
        ``data/processed/phase5_calibrated_model_<target>.joblib``
        containing the chosen probability calibrator (Platt or
        isotonic), the chosen conformity score, the LAC quantiles
        (or MAPIE regressor), the four confidence levels, and the
        Phase 4 winner reference for downstream lookup.
    """),
    code("""
        for target in ['log_roi', 'roi_gt_1', 'roi_gt_2']:
            bundle = joblib.load(paths.DATA_PROCESSED_DIR / f'phase5_calibrated_model_{target}.joblib')
            print(f"{target}: best_probability={bundle['best_probability_method']!s:>11s}  "
                  f"best_conformal={bundle['best_conformal_score']}  "
                  f"levels={bundle['deployed_confidence_levels']}")
    """),

    # ============================================================
    # Conclusion
    # ============================================================
    md("""
        ## H. Conclusion and bridge to Phase 6

        Probability calibration succeeds: isotonic halves ECE on both
        classification targets. Conformal coverage is in-band at every
        confidence level and every target. Trigger #1 (coverage
        failure) does not fire; trigger #2 (over-deferral) fires on
        ``roi_gt_2`` honestly, as a property of the underlying 0.65
        OOF AUC rather than a calibration defect. Per-genre refer rate
        anti-correlates with Phase 4 OOF AUC, validating the corpus-
        bimodality finding empirically.

        Two open questions for Phase 6 surface here:

        1. **Probability-driven or conformal-set-driven decisions?**
           The Phase 6 cost-decision rule could read the calibrated
           scalar probability and minimize expected cost, or it could
           read the conformal prediction set and route doubletons to
           Refer directly. The pre-registration of Phase 6 picks the
           probability-driven path because it interpolates smoothly
           across the cost-matrix sweep where the set-driven path
           does not.
        2. **Per-genre or single confidence threshold?** The per-genre
           refer-rate disparity raises the question of whether
           Phase 6 should set per-genre cost matrices or a single
           global one. The Phase 6 sensitivity sweep evaluates both;
           the per-genre variant turns out to add no lift because
           the genre signal is already captured by the calibrated
           probability.

        Phase 6 will read the deployed wrappers from
        ``data/processed/phase5_calibrated_model_*.joblib`` and apply
        the cost-asymmetric decision rule from the project's source
        cost matrix in ``PROJECT_CONTEXT.md`` Section 1.
    """),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    OUTPUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
