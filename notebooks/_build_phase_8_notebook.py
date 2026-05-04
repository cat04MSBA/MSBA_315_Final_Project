"""Build ``notebooks/phase_8.ipynb`` from a structured cell list.

Regenerate the notebook with::

    python -m notebooks._build_phase_8_notebook

The Phase 8 notebook documents the end-to-end integration and the
final test-set evaluation. The held-out 257-film test set is
touched for the first time in this phase; the Phase 8 numbers are
the project's headline test-set figures.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_8.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 8: End-to-End Integration & Final Test-Set Evaluation

        Phases 4 through 7 each built one layer of the four-layer
        triage system in isolation. Phase 8 assembles them into a
        single end-to-end pipeline, evaluates the assembled system on
        the held-out 257-film test set for the first time across the
        project, and produces the example-output gallery that
        anchors the Phase 9 report and presentation.

        The held-out test set has been untouched since the Phase 3
        stratified split. Phase 8 is the first and only phase to
        evaluate on it. After Phase 8 the test-set numbers are
        frozen; nothing in Phase 9 revisits the test set with a new
        model or a new threshold. Whatever the test set produces is
        reported honestly.

        ## How Phase 8 is organized

        Pre-registration discipline applies to the most consequential
        phase in the project.
        ``docs/proposals/phase8_preregistration.md`` locks the
        end-to-end pipeline contract, the test-set metric set with
        bootstrap confidence intervals, the four error-analysis cuts,
        the five example-film selection rules, and five escalation
        triggers before any test-set load.
    """),

    # ============================================================
    # Inputs and isolation
    # ============================================================
    md("""
        ## A. Test-set isolation check

        Before any test-set load, the pipeline verifies
        programmatically that the test split is disjoint from train
        and cal. Running this check at the head of every Phase 8
        invocation forces a hard failure if the splits ever drift.
    """),
    code("""
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from src.evaluation.pipeline import assert_test_set_isolation
        from src.utils import paths

        assert_test_set_isolation()
    """),

    # ============================================================
    # End-to-end pipeline contract
    # ============================================================
    md("""
        ## B. The end-to-end pipeline

        ``src.evaluation.pipeline.triage_report`` is the single-film
        entry point. Given an imdb_id and the four-layer artifact
        bundle, it returns a structured ``TriageReport`` with
        Layer 1 point predictions, Layer 2 calibrated probability
        and conformal intervals, Layer 3 recommended action with
        rationale, and Layer 4 SHAP contributors with the composed
        natural-language rationale. ``run_batch`` applies it to
        every film in the test feature matrix.

        Two execution modes exercise the pipeline. Batch mode reads
        feature rows from ``data/processed/features.parquet`` and is
        what produces the headline test-set numbers. Single-film
        smoke-test mode runs three test films through the per-film
        function twice and verifies that identical inputs produce
        identical outputs (the determinism guarantee).
    """),

    # ============================================================
    # Layer 1
    # ============================================================
    md("""
        ## C. Layer 1: predictive performance with bootstrap CIs

        Each metric carries a 95 percent percentile bootstrap CI
        from 2,000 resamples of the 257 films at seed 42. Bootstrap
        is over films, not folds; there is one test-set point
        estimate per metric.
    """),
    code("""
        m = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_test_metrics.csv')
        # Pivot for readable display.
        m['ci'] = m.apply(lambda r: f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]", axis=1)
        m['point'] = m['point'].round(3)
        out = m.pivot_table(index='target', columns='metric', values='point', aggfunc='first')
        ci_table = m.pivot_table(index='target', columns='metric', values='ci', aggfunc='first')
        print("Point estimates:")
        print(out.to_string())
        print("\\n95% bootstrap CIs:")
        print(ci_table.to_string())
    """),
    md("""
        The headline classification target ``roi_gt_2`` lands at
        AUC 0.507 [0.437, 0.584] on the test set, essentially chance
        performance, well below the Phase 4 OOF 0.652. Trigger #1 of
        the pre-registration fires accordingly. The other targets
        fare differently: ``roi_gt_1`` AUC 0.596 [0.514, 0.677] CI
        clears 0.5; ``log_roi`` RMSE 1.217 [1.074, 1.357] is actually
        slightly better than the OOF RMSE of 1.310.

        The gap is real, not a code defect. The smoke-test passes
        (deterministic per-film triage); the test-set isolation
        check passes; the artifacts loaded match the records on
        disk. The result is what the assembled system actually
        produces on a held-out 257-film slice of the corpus.
    """),

    # ============================================================
    # Layer 2
    # ============================================================
    md("""
        ## D. Layer 2: calibration and conformal coverage

        Calibration on the test set is reported by ECE, Brier
        score, and log-loss. Conformal coverage at the four
        confidence levels is the empirical fraction of test films
        whose true label lies in the prediction set; the pre-
        registration sets a ±5 percentage point tolerance band at
        the 0.90 nominal level.
    """),
    code("""
        cal = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_calibration_test.csv')
        print("Calibration metrics on the test set:")
        print(cal.round(3).to_string(index=False))

        cov = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_coverage_test.csv')
        print("\\nConformal coverage at four confidence levels:")
        cov_pivot = cov.pivot_table(
            index='target', columns='level', values='empirical_coverage'
        ).round(3)
        print(cov_pivot.to_string())
    """),
    md("""
        Calibration on the test set is **better** than on cal: ECE
        drops to 0.054 (``roi_gt_1``) and 0.085 (``roi_gt_2``) from
        the cal-set values of 0.095 and 0.108. The isotonic
        calibrator generalizes; the test set is a mild
        in-distribution sample for the monotone fit.

        Conformal coverage at 0.90 nominal is 0.864 (``roi_gt_2``)
        and 0.891 (``log_roi``), both within the ±5pp tolerance
        band. Trigger #2 of the pre-registration does not fire.
    """),
    code("""
        from IPython.display import Image
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_calibration_test.png'))
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_coverage_test.png'))
    """),

    # ============================================================
    # Layer 3
    # ============================================================
    md("""
        ## E. Layer 3: decision quality on the test set

        Under the default cost matrix, the system commits 2
        Greenlights + 0 Pass + 255 Refer on the test set. The
        action mix tracks the cal-set distribution closely; the
        difference is in the realized cost.
    """),
    code("""
        dec = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_decision_evaluation_test.csv')
        dec = dec.sort_values('total_cost', ascending=True)
        print(dec[['strategy', 'total_cost', 'p_greenlight', 'p_pass', 'p_refer', 'cost_per_film_M']]
              .round(3).to_string(index=False))
    """),
    md("""
        System total cost on the test set is **$51.275M**. Trigger
        #3 of the pre-registration fires: $51.3M is above the $2.6M
        ceiling (twice the cal-set value). The cause is one
        Greenlight on a 1983 Disney Fantasy flop with calibrated
        probability 1.0; the other Greenlight (a 1981 Animation
        film) was correct.

        The system still beats four of five baselines by 2-4 orders
        of magnitude. Read-Everything beats the system by a factor
        of about 40, entirely because of the single mis-confident
        commit. The system's value proposition is intact: it ties
        Read-Everything when calibrated probabilities are uncertain
        (the 99 percent refer behavior) and only diverges when
        probability hits the isotonic upper plateau for a small
        handful of films.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_decision_costs_test.png'))
    """),
    md("""
        The refer-cost sweep on the test set reproduces the Phase 6
        finding: at any realistic refer cost the action distribution
        is essentially constant; the transition to commit-mostly
        behavior happens between $1M and $25M refer cost.
    """),
    code("""
        sw = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_decision_sensitivity_test.csv')
        print(sw[['cost_matrix_name', 'cost_refer_flop', 'p_greenlight', 'p_refer', 'total_cost_system']]
              .round(3).to_string(index=False))
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_decision_sensitivity_test.png'))
    """),

    # ============================================================
    # Layer 4
    # ============================================================
    md("""
        ## F. Layer 4: SHAP attribution generalizes to the test set

        Two diagnostics establish that the system reads the same
        things on test that it read on cal. SHAP-vs-native rank
        correlation on test reproduces the Phase 7 cal-set value;
        the test-vs-cal top-15 Jaccard overlap measures whether the
        ranking shape itself is preserved.
    """),
    code("""
        rank = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_top_shap_test.csv').head(15)
        print("Top-15 SHAP features on the test set:")
        print(rank[['rank', 'feature', 'mean_abs_shap', 'mean_signed_shap']].round(4).to_string(index=False))
    """),
    code("""
        overlap = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_shap_test_vs_cal.csv')
        jaccard = float(overlap['jaccard_top15'].iloc[0])
        print(f"Test-vs-cal top-15 Jaccard overlap: {jaccard:.3f}")
    """),
    md("""
        The test-set top-15 SHAP features overlap the cal-set
        top-15 by Jaccard 0.875: 14 of 16 features in the union
        appear in both. The two swaps are minor genre / topic
        substitutions. SHAP-vs-native ρ on test is 0.750, matching
        the Phase 7 cal-set value of 0.745. Trigger #4 of the
        pre-registration does not fire. The headline takeaway:
        the system reads the same structural / genre / network /
        embedding signals on test that it read on cal. The
        instability on test is in the AUC, not in the attribution.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_top_shap_test.png'))
    """),

    # ============================================================
    # Error analysis
    # ============================================================
    md("""
        ## G. Error-analysis cuts

        Four pre-registered cuts: by primary genre, by release
        decade, by budget tier, and by screenplay-length tier.
        Cells with one class present have NaN AUC by convention.
    """),
    code("""
        for fname, label in [
            ('phase8_error_by_genre.csv',         'PER GENRE'),
            ('phase8_error_by_decade.csv',        'PER DECADE'),
            ('phase8_error_by_budget_tier.csv',   'PER BUDGET TIER'),
            ('phase8_error_by_length_tier.csv',   'PER LENGTH TIER'),
        ]:
            df = pd.read_csv(paths.REPORTS_TABLES_DIR / fname)
            print(f"--- {label} ---")
            cols = ['cut_value', 'n', 'n_pos', 'auc', 'p_refer', 'cost_per_film_M']
            print(df[cols].round(3).to_string(index=False))
            print()
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase8_per_genre_metrics_test.png'))
    """),
    md("""
        Three findings stand out. **Big-budget tentpoles
        (over-$150M) reach AUC 0.846**: these are the films where
        the corpus is heavily survivorship-biased and the
        structural / network features correlate strongly with the
        budget tier. **Long scripts (over 200 scenes) reach AUC
        0.71**: the structural / embedding features have more
        material to work with on long-form scripts. **Mid-budget
        ($50M-$150M) AUC is 0.42**: the system fails on the
        mid-budget segment, which is also the segment most
        important to studio triage.

        Per-genre, the Phase 4 corpus-bimodality finding (Adventure
        / Fantasy / Sci-Fi tractable; Drama / Comedy / Romance not)
        does not replicate cleanly per-cell at n=8-17 per cell on
        the test set. Drama is the strongest cell on test (n=66,
        AUC 0.58); Fantasy is the weakest (n=8, AUC 0.25).
    """),

    # ============================================================
    # Galleries
    # ============================================================
    md("""
        ## H. Most-correct and most-wrong galleries

        For ``roi_gt_2``, top 15 films by lowest log-loss among
        true positives (most-correct) and top 15 by highest
        log-loss in either direction (most-wrong).
    """),
    code("""
        for fname, label in [
            ('phase8_top_correct_roi_gt_2.csv', 'MOST-CORRECT'),
            ('phase8_top_wrong_roi_gt_2.csv',   'MOST-WRONG'),
        ]:
            df = pd.read_csv(paths.REPORTS_TABLES_DIR / fname)
            print(f"--- {label} ---")
            cols = ['movie_name', 'primary_genre_bucketed', 'release_year_parsed',
                    'true_label', 'calibrated_probability_roi_gt_2', 'recommended_action']
            print(df[cols].head(8).to_string(index=False))
            print()
    """),
    md("""
        The most-wrong headline is *Something Wicked This Way Comes*
        (1983, Fantasy): per-film log-loss 16.1, the maximum
        possible value (calibrated probability 1.0 with a flop true
        label). The most-correct headline is *Heavy Metal* (1981,
        Animation): the system correctly greenlit it. Both top
        films are pre-1985, which is consistent with the Phase 4
        Tier-A finding that pre-1985 films are statistically
        unreliable. A deployment guard clamping Greenlight to
        ``release_year >= 1990`` would suppress both commits and
        recover the Read-Everything baseline cost.

        Films at the isotonic plateau (calibrated probability
        ≈ 0.7125) populate both galleries: the plateau is the
        honest output of isotonic regression on a 257-film cal
        set. The system cannot distinguish among films on the
        plateau and correctly defers all of them.
    """),

    # ============================================================
    # Example gallery
    # ============================================================
    md("""
        ## I. Five-film curated example gallery

        Five films selected per the locked rules in pre-registration
        Section 4.7. Full per-example tables, decision rationales,
        and SHAP rationales are in
        ``reports/tables/phase8_example_gallery.md``.

        The selection rules are: (1) highest-probability Greenlight,
        (2) highest-confidence Pass or, if none exist, the
        lowest-probability Refer, (3) Refer closest to 0.50
        calibrated probability, (4) Adventure / Fantasy / Sci-Fi
        true positive correctly identified, (5) Drama / Comedy /
        Romance correctly deferred near 0.50.
    """),
    code("""
        gallery_path = paths.REPORTS_TABLES_DIR / 'phase8_example_gallery.md'
        # Print the first three example sections (the others follow the same template).
        text = gallery_path.read_text()
        sections = text.split('## ')
        # First chunk is the header; sections[1:] are example entries.
        preview = '## '.join(sections[:4])  # header + 3 examples
        print(preview[:3500])
    """),

    # ============================================================
    # Triggers
    # ============================================================
    md("""
        ## J. Pre-registered escalation triggers

        Five triggers were pre-registered. Two fired honestly; three
        passed.
    """),
    code("""
        trig = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase8_escalation_triggers.csv')
        print(trig[['trigger_id', 'metric', 'value', 'threshold', 'fired', 'status_note']].to_string(index=False))
    """),

    # ============================================================
    # Conclusion
    # ============================================================
    md("""
        ## K. Conclusion and bridge to Phase 9

        The headline finding of Phase 8 is the predictive-performance
        gap: ``roi_gt_2`` test AUC 0.507 [0.437, 0.584] vs the Phase
        4 OOF point estimate of 0.652. The gap is real and reflects
        the corpus's small effective size at the per-cell granularity
        the Phase 4 OOF stratified diagnostic worked at. Phase 9
        will lead the report with the 0.51 figure honestly.

        The architecture's contribution is intact. Calibration on
        the test set is excellent (ECE 0.054 / 0.085, the best
        numbers in the project). Conformal coverage is in-band at
        every confidence level. SHAP attribution generalizes
        cleanly (Jaccard 0.875 on top-15 features; Spearman ρ
        0.750 vs native importance). The system still beats four
        of five Phase 6 baselines by 2-4 orders of magnitude.

        Phase 9 will:

        * Draft the report (≤ 10 pages) following the course
          rubric: Abstract, Introduction, Literature, Methodology,
          Results, Conclusion. The headline result is the
          calibration + decision layer; the 0.51 test AUC is
          reported honestly.
        * Draft the presentation slides (10-15) leading with the
          example-film gallery on real test-set films.
        * Merge the per-phase Python scripts and per-phase summary
          documents into a single Jupyter notebook organized by
          topic.
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
