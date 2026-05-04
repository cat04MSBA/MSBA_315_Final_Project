"""Build ``notebooks/phase_7.ipynb`` from a structured cell list.

Regenerate the notebook with::

    python -m notebooks._build_phase_7_notebook

The Phase 7 notebook documents Layer 4 SHAP explanations:
TreeSHAP attribution on the Phase 4 winners, the SHAP-vs-native
and stability validation, the per-film rationale that extends
Phase 6, and the proof-of-concept scene-level attribution on five
example films.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_7.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 7: Layer 4 SHAP Explanations

        A "Pass" recommendation without explanation is useless to a
        writer. The point of the four-layer architecture is not just
        to triage scripts but to produce actionable feedback: the
        system should be able to say which features pushed the
        prediction up and which pulled it down, ideally at scene
        level so the writer knows which scenes to revise. Layer 4
        wraps the Phase 4 winner with TreeSHAP attribution and
        produces a per-film rationale plus a proof-of-concept
        scene-level breakdown.

        TreeSHAP is the standard explanation method for tree-based
        models. It is exact (no Monte Carlo), fast (path-dependent
        conditioning runs in time linear in tree depth), and additive
        (per-feature contributions sum to the model's output minus
        the base rate). The Phase 4 ``roi_gt_2`` winner is XGBoost,
        and the ``log_roi`` winner is RandomForest; both are
        TreeSHAP-supported. The ``roi_gt_1`` winner is SVM-RBF, and
        KernelSHAP for SVM-RBF would take 4-8 hours on a 257-film
        calibration set with 92 features. The pre-registration
        substitutes the Phase 4 permutation importance for SVM SHAP
        global ranking; per-film attribution for ``roi_gt_1`` is
        omitted from Phase 7.

        ## How Phase 7 is organized

        Pre-registration carries forward.
        ``docs/proposals/phase7_preregistration.md`` locks the
        explainer family per target, the stability validation
        protocol, the per-film rationale schema, the scene-level
        approach, and the four escalation triggers before any SHAP
        value was computed.
    """),

    # ============================================================
    # Inputs
    # ============================================================
    md("""
        ## A. Inputs

        Phase 7 reads the Phase 4 winner bundles and the Phase 5
        calibrated wrapper. SHAP runs on the calibration set (the
        held-out test set is reserved for Phase 8 per the no-test-set
        rule).
    """),
    code("""
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from src.utils import paths

        for target in ['log_roi', 'roi_gt_1', 'roi_gt_2']:
            p4 = joblib.load(paths.DATA_PROCESSED_DIR / f'phase4_primary_model_{target}.joblib')
            print(f"Phase 4 winner | {target:9s}: {p4['family']}")

        explainer = joblib.load(paths.DATA_PROCESSED_DIR / 'phase7_shap_explainer_roi_gt_2.joblib')
        print(f"\\nDeployed Phase 7 explainer (roi_gt_2): {explainer['family']}, "
              f"{len(explainer['feature_names'])} features, base value = {explainer['base_value']:.3f}")
    """),

    # ============================================================
    # Global ranking
    # ============================================================
    md("""
        ## B. Global SHAP feature ranking

        Mean absolute SHAP value across the 257 calibration films is
        the global ranking metric. Higher mean |SHAP| means the
        feature contributes more to model output across the corpus.
        Mean signed SHAP carries the direction: positive means the
        feature pushes probability up on average; negative means it
        pulls down.
    """),
    code("""
        ranking = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase7_global_shap_roi_gt_2.csv')
        head = ranking.head(15)[['rank', 'feature', 'mean_abs_shap', 'mean_signed_shap']]
        print(head.round(4).to_string(index=False))
    """),
    md("""
        ``release_year_parsed`` is the strongest feature: the model
        treats the era as a meaningful predictor, consistent with
        Phase 4's pre-1985 unreliability finding. ``genre_Horror``
        ranks second with mean signed SHAP +0.082, meaning Horror
        genre pushes probability up; ``genre_Romance`` carries
        mean signed SHAP −0.065, the SHAP version of the Phase 4
        finding that Romance is the lowest-AUC genre. Network and
        embedding features fill the mid-pack; topic features appear
        but lower.
    """),
    code("""
        from IPython.display import Image
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase7_global_shap.png'))
    """),

    # ============================================================
    # SHAP vs native
    # ============================================================
    md("""
        ## C. SHAP-vs-native and stability validation

        Two pre-registered triggers protect against degenerate SHAP
        outputs.

        **Trigger #1: SHAP-vs-native rank correlation.** TreeSHAP
        ranks features by per-film attribution. The Phase 4 native
        importance ranks them by something else (gain or split count
        for tree models). If the two rankings disagree completely
        (Spearman ρ near zero) the SHAP attribution is unrelated to
        what the model is actually doing. The pre-registration sets
        a 0.5 floor.

        **Trigger #2: SHAP stability across conditioning modes.**
        TreeSHAP has two conditioning modes: tree-path-dependent
        (default, fast) and interventional (slower, requires a
        background dataset, the unbiased estimator from Lundberg et
        al. 2020). If they agree on global ranking (Spearman ρ >
        0.8), the model's interactions are well-behaved enough that
        either method's output is trustworthy.
    """),
    code("""
        stab = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase7_stability.csv')
        print(stab.round(3).to_string(index=False))
    """),
    md("""
        Both triggers pass with margin: SHAP-vs-native ρ = 0.745 on
        the headline target (above the 0.5 floor) and 0.886 on the
        regression target (above the 0.8 strong-agreement
        threshold). Stability ρ is 0.967 / 0.979, both well above
        the 0.8 floor.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase7_shap_vs_native.png'))
    """),

    # ============================================================
    # Per-film rationale
    # ============================================================
    md("""
        ## D. Per-film rationale

        For each of the 257 calibration films, Phase 7 produces a
        natural-language rationale of the form "Top features pushing
        probability up: A (+v log-odds), B (+v), C (+v). Top features
        pulling down: X (−v), Y (−v), Z (−v)." The rationale extends
        the Phase 6 decision rationale, giving the writer a feature-
        level account of which dialogue properties drove the
        recommendation.
    """),
    code("""
        per_film = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase7_per_film_rationale.csv')
        # Show two illustrative films: one Greenlight and one Refer.
        cols = ['imdb_id', 'recommended_action', 'calibrated_probability',
                'top_pos_features', 'top_neg_features']
        gl = per_film[per_film['recommended_action'] == 'Greenlight'].head(1)
        refer = per_film[per_film['recommended_action'] == 'Refer'].head(1)
        print("--- Greenlight example ---")
        for _, r in gl.iterrows():
            for c in cols:
                print(f"  {c}: {r[c]}")
        print("\\n--- Refer example ---")
        for _, r in refer.iterrows():
            for c in cols:
                print(f"  {c}: {r[c]}")
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase7_per_film_examples.png'))
    """),

    # ============================================================
    # Scene-level
    # ============================================================
    md("""
        ## E. Scene-level attribution: proof of concept

        The most ambitious part of Phase 7. For one example film,
        the per-scene removal counterfactual asks: "if scene 47 had
        not been in the script, how would the model's predicted
        probability change?" The implementation re-extracts only the
        features that depend on scene-level structure (structural
        counts, embedding mean-pool) and re-uses the rest. The
        approximation captures relative ranking but not exact
        magnitudes, which is why the pre-registration recommends
        scene-level as a proof-of-concept rather than a deployable
        surface.

        Five example films were selected by category (one
        high-confidence Greenlight, one Drama Refer, one Adventure
        true positive, one most-wrong from the Phase 4 gallery, one
        sleeper-hit pattern). One was deduped on overlap, leaving
        four films with per-scene contributions saved to
        ``data/processed/phase7_scene_level_examples.json``.
    """),
    code("""
        scene = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase7_scene_level.csv')
        # Headline statistics: how many scenes per example, and the range of contributions.
        agg = scene.groupby('imdb_id')['contribution'].agg(['count', 'min', 'max']).round(4)
        print("Per-example summary (scenes per film, min/max scene contribution):")
        print(agg.to_string())
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase7_scene_level_example.png'))
    """),
    md("""
        Per-scene contributions are small in absolute terms (0.5 to
        1.5 percentage points per scene) but distinguishable from
        the median by 5-10× — the system's prediction is structural
        and distributed across many scenes rather than driven by one
        decisive scene. The recommendation in the Phase 7 summary
        is to ship feature-level attribution as the deployable and
        document scene-level as proof-of-concept for the report.
    """),

    # ============================================================
    # Conclusion
    # ============================================================
    md("""
        ## F. Conclusion and bridge to Phase 8

        All four pre-registered triggers pass. SHAP-vs-native rank
        correlation clears the 0.5 floor on both targets; stability
        across conditioning modes clears the 0.8 floor by margin;
        scene-level attribution is feasible at typical screenplay
        scene counts; total compute is well under the 90-minute
        budget.

        The deployable artifact at
        ``data/processed/phase7_shap_explainer_roi_gt_2.joblib`` lets
        Phase 8 compute SHAP on the 257 test films in under a minute.
        The 257 per-film rationales at
        ``reports/tables/phase7_per_film_rationale.csv`` can be
        embedded into the Phase 9 report's example-output gallery.

        Phase 8 will:

        * Open the held-out 257-film test set for the first time
          across the project.
        * Run the four-layer pipeline end-to-end: Phase 4 prediction
          → Phase 5 calibrated probability + conformal interval →
          Phase 6 cost-asymmetric action → Phase 7 SHAP attribution.
        * Compute final test-set metrics and the per-genre /
          per-decade / per-budget-tier / per-length-tier breakdowns.
        * Curate the example-output gallery for the Phase 9
          presentation.
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
