"""Build ``notebooks/phase_6.ipynb`` from a structured cell list.

Regenerate the notebook with::

    python -m notebooks._build_phase_6_notebook

The Phase 6 notebook documents Layer 3 asymmetric-cost decision:
the cost matrix sourced from ``PROJECT_CONTEXT.md`` Section 1, the
expected-cost decision rule, baseline comparisons, sensitivity
across cost-matrix variants, and the per-film deliverable that
feeds Phase 7.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_6.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 6: Layer 3 Asymmetric-Cost Decision

        A calibrated probability is half of a decision; the other
        half is the cost of being wrong. Producing a flop costs
        roughly fifty million dollars in lost budget. Passing on a
        hit can cost two to four times that in foregone revenue.
        Routing a film to a human reader costs about five thousand
        dollars in reviewer time. The three actions Greenlight,
        Pass, and Refer have wildly asymmetric error costs, and a
        decision rule that ignores the asymmetry will commit the
        wrong errors.

        Phase 6 converts the Phase 5 calibrated probability
        ``P(roi_gt_2 | features)`` into one of the three actions by
        minimizing expected cost under the project's source cost
        matrix. The headline diagnostic is system total cost
        compared against five baselines (Always-Greenlight,
        Always-Pass, Read-Everything, Random, Genre-prior).
        Sensitivity sweeps across cost-matrix variants establish
        where the system's behavior is robust and where it changes.

        ## How Phase 6 is organized

        Pre-registration discipline carries forward to Phase 6.
        ``docs/proposals/phase6_preregistration.md`` locks the cost
        matrix structure, the decision rule (expected-cost
        minimization with tie-break to Refer), the baseline list,
        the sensitivity-sweep variants, and the four escalation
        triggers before any decision evaluation ran.
    """),

    # ============================================================
    # Inputs and cost matrix
    # ============================================================
    md("""
        ## A. Inputs and the source cost matrix

        Phase 6 reads the Phase 5 calibrated wrapper for the
        headline target ``roi_gt_2`` from
        ``data/processed/phase5_calibrated_model_roi_gt_2.joblib``.
        The cost matrix defaults are sourced from
        ``PROJECT_CONTEXT.md`` Section 1.
    """),
    code("""
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        from src.decision.cost_matrix import DEFAULT_COST_MATRIX
        from src.utils import paths

        cm = DEFAULT_COST_MATRIX
        print("Default cost matrix (USD):")
        print(f"  Greenlight + flop:  ${cm.cost_greenlight_flop:>13,.0f}  (lost production budget)")
        print(f"  Greenlight + hit:   ${cm.cost_greenlight_hit:>13,.0f}  (correct, no error cost)")
        print(f"  Pass + flop:        ${cm.cost_pass_flop:>13,.0f}  (correct, no error cost)")
        print(f"  Pass + hit:         ${cm.cost_pass_hit:>13,.0f}  (foregone revenue)")
        print(f"  Refer + flop:       ${cm.cost_refer_flop:>13,.0f}  (one human reader pass)")
        print(f"  Refer + hit:        ${cm.cost_refer_hit:>13,.0f}  (one human reader pass)")
    """),

    # ============================================================
    # Decision rule
    # ============================================================
    md("""
        ## B. The decision rule

        The rule is expected-cost minimization with tie-break to
        Refer. For a calibrated probability ``p`` of a hit, expected
        cost per action is

        $$
          \\mathbb{E}[\\text{cost} \\mid \\text{Greenlight}] = (1 - p) \\cdot 50\\text{M} \\\\
          \\mathbb{E}[\\text{cost} \\mid \\text{Pass}]       = p \\cdot 100\\text{M} \\\\
          \\mathbb{E}[\\text{cost} \\mid \\text{Refer}]      = 5\\text{K}
        $$

        At the source default values, Refer beats Pass when ``p <
        0.99995`` and beats Greenlight when ``p > 0.0001``. The
        decision boundary at which Greenlight beats Refer is the
        calibrated probability above which Greenlight's expected
        cost falls below the refer cost: ``p > 1 - 5K / 50M ≈
        0.9999``. With realistic calibrated probabilities the
        Refer action wins almost everywhere, which is exactly the
        result trigger #1 of the pre-registration anticipates.
    """),

    # ============================================================
    # System vs baselines
    # ============================================================
    md("""
        ## C. System vs five baselines under the default cost matrix
    """),
    code("""
        baselines = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase6_baselines.csv')
        baselines['total_cost_M'] = (baselines['total_cost'] / 1_000_000).round(1)
        baselines = baselines.sort_values('total_cost', ascending=True)
        print(baselines[['strategy', 'cost_matrix', 'total_cost_M', 'p_greenlight', 'p_pass', 'p_refer']]
              .to_string(index=False))
    """),
    md("""
        On the calibration set the system ties Read-Everything at
        $1.3M total cost and beats the other four baselines by 3-4
        orders of magnitude. The 1.2 percent Greenlight rate is
        produced by two films whose calibrated probability is
        sufficiently extreme to clear the 0.9999 threshold. The
        zero Pass rate is structural: at the default cost matrix
        Refer always dominates Pass at any probability where the
        model has any uncertainty.
    """),
    code("""
        from IPython.display import Image
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase6_baselines_comparison.png'))
    """),

    # ============================================================
    # Trigger #1
    # ============================================================
    md("""
        ## D. Trigger #1 fires as predicted: cost asymmetry alone
        does not change behavior

        Pre-registration trigger #1 anticipates that the asymmetry
        variants (1:1, 1:2, 1:4, 2:1) and the base-magnitude
        variants (×0.1, ×1, ×10) will produce identical action
        distributions, because Refer at $5K is dramatically cheaper
        than any error at $25M-$200M. The sensitivity sweep
        confirms this empirically.
    """),
    code("""
        sens = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase6_sensitivity.csv')
        # Asymmetry + base-magnitude variants: do they shift the action mix?
        asym_mag = sens[
            sens['cost_matrix_name'].str.startswith(('asymmetry', 'scale', 'default'))
        ].copy()
        print(asym_mag[['cost_matrix_name', 'p_greenlight', 'p_pass', 'p_refer', 'total_cost_system']]
              .round(4).to_string(index=False))
    """),
    md("""
        Action distribution is identical across all asymmetry and
        base-magnitude variants. The cost asymmetry alone does not
        change behavior; what does change behavior is the refer
        cost.
    """),

    # ============================================================
    # Refer-cost sweep
    # ============================================================
    md("""
        ## E. The refer-cost sweep is the operationally meaningful
        sensitivity

        The refer cost represents the per-film opportunity cost of
        a human reader, and it is the parameter the deploying
        studio actually controls (their human-reader budget divided
        by their annual film throughput). The sweep spans six
        orders of magnitude.
    """),
    code("""
        refer = sens[sens['cost_matrix_name'].str.startswith(('refer_', 'default'))].copy()
        print(refer[['cost_matrix_name', 'cost_refer_flop', 'p_greenlight', 'p_pass', 'p_refer', 'total_cost_system']]
              .round(4).to_string(index=False))
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase6_cost_curve.png'))
    """),
    md("""
        At $0K refer cost the system defers 100 percent. At $5K to
        $1M the action distribution is essentially constant at
        roughly 99 percent Refer. The transition happens between
        $1M and $25M refer cost: when refer cost rises to half the
        production budget, abstention is no longer cheap relative
        to taking the model's recommendation, and the system flips
        to roughly 95 percent Greenlight. At any realistic per-film
        human-reader cost (between $1K and $100K) the system
        behaves as a "flag everything for human review" channel
        with rare confident commits.
    """),

    # ============================================================
    # Per-genre breakdown
    # ============================================================
    md("""
        ## F. Per-genre action breakdown

        Phase 5 surfaced an anti-correlation between per-genre
        refer rate and Phase 4 OOF AUC. Phase 6 reproduces it under
        the cost-decision rule. The per-genre cost-matrix variant,
        which uses per-genre median budget and revenue from the
        train split, is reported alongside the global default to
        test whether genre-specific cost matrices change the action
        mix.
    """),
    code("""
        Image(filename=str(paths.REPORTS_FIGURES_DIR / 'phase6_per_genre_actions.png'))
    """),
    code("""
        per_genre = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase6_per_genre_actions.csv')
        print(per_genre.round(3).to_string(index=False))
    """),
    md("""
        Trigger #3 of the pre-registration ("no genre-tuning lift")
        fires: per-genre cost matrices produce the same total cost
        as the global default. The genre signal is already captured
        by the calibrated probability; per-genre cost matrices add
        nothing.
    """),

    # ============================================================
    # Per-film output
    # ============================================================
    md("""
        ## G. Per-film decision output

        For each calibration film, Phase 6 saves the calibrated
        probability, the expected cost per action, the recommended
        action, and a natural-language rationale. This per-film
        table feeds Phase 7 (which extends the rationale with SHAP
        contributors) and Phase 8 (which evaluates the same
        pipeline on the held-out test set).
    """),
    code("""
        decisions = pd.read_csv(paths.REPORTS_TABLES_DIR / 'phase6_decisions.csv')
        # Show the two Greenlight films and four representative Refers.
        gl = decisions[decisions['recommended_action'] == 'Greenlight']
        refer_sample = (
            decisions[decisions['recommended_action'] == 'Refer']
            .sort_values('calibrated_probability', ascending=False)
            .head(4)
        )
        print("Greenlight films on calibration set:")
        for _, r in gl.iterrows():
            print(f"  {r['imdb_id']} ({r['genre']}): P={r['calibrated_probability']:.3f}")
        print("\\nFour highest-probability Refer films:")
        for _, r in refer_sample.iterrows():
            print(f"  {r['imdb_id']} ({r['genre']}): P={r['calibrated_probability']:.3f}")
    """),
    code("""
        # Example rationale (the deployable string for one film).
        sample = decisions.iloc[0]
        print(f"imdb_id: {sample['imdb_id']}")
        print(f"action:  {sample['recommended_action']}")
        print(f"rationale: {sample['rationale']}")
    """),

    # ============================================================
    # Conclusion
    # ============================================================
    md("""
        ## H. Conclusion and bridge to Phase 7

        Phase 6 builds the cost-asymmetric decision rule and the
        per-film deliverable that turns the model's probability
        output into an action and a written rationale. Triggers #1
        and #3 of the pre-registration fire as anticipated; triggers
        #2 (system worse than random) and #4 (cost discontinuity) do
        not fire. The system ties Read-Everything at $1.3M total
        cost on the calibration set and beats the other four
        baselines by 3-4 orders of magnitude.

        The operational interpretation is direct: at any realistic
        refer cost the system functions as a "flag everything for
        human review" channel that supplies the calibrated
        probability and conformal interval to the human reviewer,
        and commits unilaterally only on the 1-2 percent of films
        whose calibrated probability is essentially 1.0 or 0.0.

        Phase 7 will read the Phase 4 winner pipeline and produce
        TreeSHAP attributions to extend the per-film rationale with
        feature contributions. Phase 8 will exercise the assembled
        pipeline on the held-out test set for the first time.
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
