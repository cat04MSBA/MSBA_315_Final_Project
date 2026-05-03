"""Build ``notebooks/phase_3.ipynb`` from a structured cell list.

The notebook content lives here for maintainability. Regenerate the
notebook with:

    python -m notebooks._build_phase_3_notebook

The Phase 3 notebook documents feature extraction with incremental
ablation. The current build covers Phase 3a (the train / calibration /
test split and the baseline floor on existing master-Parquet features,
including the planning-conversation revision adopted 2026-05-03).
Phase 3b feature groups (lexical, sentiment, topic, embedding, character
network) will be appended to this builder as each proposal is approved
and implemented.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_3.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 3: Feature Extraction with Incremental Baselining

        ## Objective

        Phase 2 produced the canonical processed corpus
        (`data/processed/films_joined.parquet`, 1,713 films across 42
        columns; `data/processed/screenplays_parsed.pkl`, the
        per-film `ParsedScreenplay` structures). Phase 3 converts each
        screenplay into a fixed-length numerical feature vector that
        Phase 4 will model on. Methodologically, the phase is
        structured around incremental ablation: a baseline floor is
        established on existing master-Parquet features (Phase 3a),
        then five engineered feature groups (lexical, sentiment,
        topic, embedding, and character network, in that order) are
        added one at a time with the lift each contributes measured
        against the floor (Phase 3b).

        The current build documents Phase 3a: the train / calibration
        / test split, the three prediction targets, and the baseline
        floor. Phase 3b sections will be appended to this notebook as
        each feature group's proposal is reviewed and implemented.

        Three prediction targets are tracked through the phase:

        * `log_roi = ln(revenue) - ln(budget)` (regression).
          Decomposable into the two log columns already on the master
          parquet, and approximately symmetric about zero on this
          corpus.
        * `roi_gt_1 = (revenue / budget > 1)` (classification).
          Approximately 80% positive (gross-profitable films).
        * `roi_gt_2 = (revenue / budget > 2)` (classification).
          Industry rule-of-thumb threshold for "net profitable after
          marketing and distribution overhead." Approximately 64%
          positive.

        The targets share threshold structure: `roi_gt_1` is the same
        as `(log_roi > 0)` and `roi_gt_2` is the same as
        `(log_roi > ln 2)`, so a regression model on `log_roi`
        reproduces both classifiers by thresholding. The choice of
        primary outcome is formally deferred to the end of Phase 4.
    """),

    # ============================================================
    # 0. Setup
    # ============================================================
    md("## 0. Environment setup"),
    md("""
        The bootstrap cell resolves the project root by walking the
        directory tree until `docs/PROJECT_CONTEXT.md` is found, then
        adds it to `sys.path` so the project's package imports
        (`from src... import ...`) succeed regardless of the
        notebook's location. The `%autoreload 2` directive ensures
        edits to any module under `src/` propagate into the notebook
        without requiring a kernel restart, which supports iterative
        development. Inline plotting is enabled.
    """),
    code("""
        import sys
        from pathlib import Path


        def _find_project_root(start: Path) -> Path:
            \"\"\"Walk up the directory tree until docs/PROJECT_CONTEXT is found.\"\"\"
            markers = ("docs/PROJECT_CONTEXT.md", "docs/PROJECT_CONTEXT.txt")
            for candidate in (start.resolve(), *start.resolve().parents):
                if any((candidate / m).is_file() for m in markers):
                    return candidate
            raise RuntimeError(f"Could not find project root from {start!s}.")


        PROJECT_ROOT = _find_project_root(Path.cwd())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        print("Project root:", PROJECT_ROOT)

        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
        get_ipython().run_line_magic("matplotlib", "inline")

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    """),

    md("Imports and path constants used throughout the notebook."),
    code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from src.utils import paths

        paths.ensure_dirs()
        print("Data processed: ", paths.DATA_PROCESSED_DIR)
        print("Reports figures:", paths.REPORTS_FIGURES_DIR)
        print("Reports tables: ", paths.REPORTS_TABLES_DIR)
    """),

    # ============================================================
    # 1. Methodology
    # ============================================================
    md("---\n\n## 1. Methodology"),
    md("""
        ### 1.1 Incremental ablation

        Phase 3 is divided into two sub-phases. Phase 3a fits a
        baseline model on existing master-Parquet features only,
        without any new feature engineering, and documents that
        floor. Phase 3b then adds engineered feature groups one at a
        time, retrains the same baseline models on the expanded
        feature matrix, and computes lift over the Phase 3a numbers.
        Each Phase 3b group is preceded by a written proposal that
        pre-registers expected lift on each of the three targets;
        the predicted-versus-actual comparison is recorded in the
        ablation table.

        The advantage of this structure is twofold. First, every
        engineered feature has to "earn its place" against a
        documented floor rather than against the raw data. Second,
        the resulting ablation table is the methodology section's
        central artifact: each row tells a story about which
        category of dialogue feature contributes which predictive
        signal.

        ### 1.2 Calibration set required for Phase 5

        Phase 5 (calibrated uncertainty via conformal prediction)
        requires a held-out calibration set distinct from training
        and test. The split is therefore three-way (train,
        calibration, test) rather than two-way. The split is
        constructed once in this phase, before any feature fitting
        that depends on the data distribution, to ensure no leakage
        across phases.

        ### 1.3 No-leakage discipline

        Per `PROJECT_CONTEXT.md` Section 6, the test set is touched
        only at final evaluation (Phase 8). The calibration set is
        reserved for Phase 5. All cross-validation in Phase 3 and
        Phase 4 operates on the training split alone. Any feature
        fitting that involves the data distribution (LDA topic
        models in Phase 3b, scalers in the modelling pipelines, any
        future imputation parameters) is fit on training data only
        and applied to the calibration and test splits.
    """),

    # ============================================================
    # 2. Train / calibration / test split
    # ============================================================
    md("---\n\n## 2. Train / calibration / test split"),
    md("""
        ### 2.1 Justification

        The corpus contains 1,713 films. A 70 / 15 / 15 partition
        produces approximately 1,200 training films, 257 calibration
        films, and 257 test films. The 70% training fraction supplies
        sufficient data for cross-validation while the calibration
        and test sets remain large enough to support stable estimates
        in Phases 5 and 8 respectively. Under the conformal-prediction
        formulae used in Phase 5, calibration sets of 200 or more
        observations yield interval widths whose Monte-Carlo
        variability is small relative to the underlying uncertainty,
        which makes 257 a comfortable sample size for that purpose.

        ### 2.2 Stratification

        Two confounds in the corpus are large enough to warrant
        explicit stratification: `primary_genre_bucketed` (per
        `DATA_NOTES.md`, Drama, Comedy, Thriller, and Action lead
        the distribution; thinner genres are bucketed via the Phase 2
        pipeline) and a coarse decade bucket derived from
        `release_year_parsed`.

        The decade bucket pools pre-1980 decades into a single
        `pre_1980s` stratum, since each individual pre-1980 decade
        contains fewer than thirty films (`DATA_NOTES.md`, Section
        2). The 2010s and 2020s are pooled because the 2020s coverage
        in the corpus extends only to 2023 and is thin relative to
        the 2010s.

        Composite (genre, decade) cells with fewer than five films
        are pooled into a single `rare|rare` stratum so that
        `StratifiedShuffleSplit` is well-defined for every named
        stratum. Empirically thirty-eight films land in the rare
        pool, approximately 2.2 percent of the corpus.

        ### 2.3 Implementation

        The split is implemented in `src/features/split.py`. All
        knobs (target proportions, decade-bucket boundaries,
        rare-cell threshold, RNG seed) live on the
        :class:`SplitConfig` dataclass. The default reproduces the
        planning-conversation reference split (70 / 15 / 15, seed
        42).
    """),
    code("""
        from src.features.split import (
            SplitConfig, build_strata, make_splits, split_diagnostics,
        )

        config = SplitConfig()
        print("Split configuration:")
        print(f"  train / cal / test: {config.train_frac} / "
              f"{config.cal_frac} / {config.test_frac}")
        print(f"  rare-cell threshold: {config.rare_cell_threshold}")
        print(f"  seed: {config.seed}")

        df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
        print(f"\\nLoaded master corpus: {len(df):,} films")
    """),
    code("""
        splits = make_splits(df, config)
        counts = splits["split"].value_counts().reindex(["train", "cal", "test"])
        print("Split sizes:")
        for name, n in counts.items():
            print(f"  {name:<6} {n:>5}  ({100 * n / len(df):.1f}%)")
    """),
    md("""
        Reproducibility was verified by running `make_splits` twice
        with the default config; the assignments are byte-identical
        across runs. The split file
        (`data/processed/split_assignments.parquet`) is the
        authoritative source for split membership in every
        downstream phase.
    """),

    md("### 2.4 Stratum diagnostics"),
    code("""
        diagnostics = split_diagnostics(splits)
        print(f"Total strata: {len(diagnostics)}")
        print(f"Rare-pool count: "
              f"{int(diagnostics.loc[diagnostics['stratum'] == 'rare|rare', 'total'].sum())}")
        print()
        print("Top-10 strata by total count:")
        diagnostics.head(10)
    """),
    md("""
        Every named stratum has at least one film in each of the
        three splits, confirming that the stratified shuffle is
        well-defined for the chosen rare-cell threshold. The full
        diagnostic table (fifty-seven strata) is saved to
        `reports/tables/phase3_split_diagnostics.csv`.
    """),

    # ============================================================
    # 3. Targets
    # ============================================================
    md("---\n\n## 3. Prediction targets"),
    md("""
        ### 3.1 Three targets in parallel

        The Phase 3 brief instructs that all three targets be tracked
        through Phases 3 and 4 in parallel; the choice of primary
        outcome is deferred to the end of Phase 4.

        * **Regression target** `log_roi`. The natural-log
          return-on-investment, computed as
          `ln(revenue) - ln(budget)`. Both component log columns
          already exist on the master parquet (`log_revenue`,
          `log_budget`), but the difference form is computed fresh
          to keep the target-construction code self-contained.
        * **Classification target 1** `roi_gt_1`. Boolean indicator
          for `revenue / budget > 1`. Equivalent to `log_roi > 0`.
        * **Classification target 2** `roi_gt_2`. Boolean indicator
          for `revenue / budget > 2`. Equivalent to
          `log_roi > ln 2 ≈ 0.693`.

        ### 3.2 Why log_roi for regression

        Three reasons. First, decomposability: the difference of two
        log columns that already exist permits Phase 4 to model
        either the joint log-ROI directly or to model log_revenue
        and log_budget separately and combine. Second, symmetry: ROI
        is heavily right-skewed (corpus maximum approximately 8500x)
        but log(ROI) is approximately symmetric about the corpus
        median of 1.06, satisfying standard regression assumptions.
        Third, threshold consistency: classification thresholds at
        ROI = 1 and ROI = 2 correspond to log_roi = 0 and
        log_roi = ln 2; a single regression model on log_roi
        reproduces both classifiers by thresholding, which clarifies
        the relationship between the three targets.
    """),
    code("""
        from src.features.targets import (
            ALL_TARGETS, LOG_ROI_COL, ROI_GT_1_COL, ROI_GT_2_COL,
            add_targets,
        )

        df_with_targets = add_targets(df)
        for col in ALL_TARGETS:
            print(f"{col:<12} dtype: {str(df_with_targets[col].dtype):<8} "
                  f"sample: {df_with_targets[col].iloc[0]}")
    """),

    md("### 3.3 Target distributions"),
    code("""
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        log_roi = df_with_targets[LOG_ROI_COL]
        axes[0].hist(log_roi, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
        axes[0].axvline(0, color="black", linewidth=1, linestyle="--", label="ROI = 1")
        axes[0].axvline(np.log(2), color="firebrick", linewidth=1, linestyle="--", label="ROI = 2")
        axes[0].set(title=f"log_roi  (median {log_roi.median():.2f})", xlabel="log_roi", ylabel="Films")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)

        rates = {
            ROI_GT_1_COL: df_with_targets[ROI_GT_1_COL].mean(),
            ROI_GT_2_COL: df_with_targets[ROI_GT_2_COL].mean(),
        }
        bars = axes[1].bar(["roi_gt_1", "roi_gt_2"], list(rates.values()),
                           color=["steelblue", "indianred"], edgecolor="black", alpha=0.85)
        axes[1].set(title="Positive-class rates", ylabel="Fraction positive", ylim=(0, 1))
        for bar, rate in zip(bars, rates.values()):
            axes[1].text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                         f"{rate:.2%}", ha="center", fontsize=10)
        axes[1].axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)

        # Cross-tab so the threshold consistency is visible.
        cross = pd.crosstab(df_with_targets[ROI_GT_1_COL], df_with_targets[ROI_GT_2_COL])
        cross = cross.reindex(index=[False, True], columns=[False, True], fill_value=0)
        im = axes[2].imshow(cross.values, cmap="Blues", aspect="auto")
        axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["roi_gt_2 = F", "roi_gt_2 = T"])
        axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["roi_gt_1 = F", "roi_gt_1 = T"])
        axes[2].set(title="Threshold consistency cross-tab")
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f"{cross.values[i, j]:,}", ha="center", va="center",
                             color="white" if cross.values[i, j] > cross.values.max() / 2 else "black")

        fig.tight_layout()
        fig.savefig(paths.REPORTS_FIGURES_DIR / "phase3_target_distributions.png", dpi=120)
        plt.show()
    """),
    md("""
        The `log_roi` distribution is approximately symmetric about
        its median of 1.06 (corresponding to a roughly 2.9x ROI),
        which justifies its use as a regression target under
        standard squared-error assumptions. The two classification
        rates differ substantially: `roi_gt_1` is approximately 80%
        positive (closely tracking the corpus's gross-profitability
        rate documented in Phase 2) while `roi_gt_2` is
        approximately 64% positive. The cross-tabulation confirms
        the threshold-consistency property: every film with
        `roi_gt_2 = True` also has `roi_gt_1 = True`. The empty
        upper-left cell (`roi_gt_1 = False AND roi_gt_2 = True`)
        verifies this construction.

        The asymmetric base rates of the two classification targets
        have downstream implications. At 80% positive, `roi_gt_1`
        sits the closest to a degenerate "always predict positive"
        model and is the most challenging target on which to
        demonstrate predictive lift; the model must specifically
        identify the thin minority of unprofitable films.
        `roi_gt_2`, with its more balanced 64% positive rate,
        admits more headroom for predictive features to reveal
        themselves.
    """),

    # ============================================================
    # 4. Phase 3a baseline feature set
    # ============================================================
    md("---\n\n## 4. Phase 3a baseline: feature set"),
    md("""
        ### 4.1 Inclusion criteria

        Phase 3a operates exclusively on features already present on
        the master parquet (or trivially derivable from columns
        already there). No screenplay-text feature engineering occurs
        in this sub-phase; the floor is intentionally a "constant
        comparator" that Phase 3b's groups must clear.

        Two feature configurations are evaluated, and each is
        evaluated under two parameterizations (raw versus log-
        transformed structural counts), yielding four runs in total.

        ### 4.2 Feature inventory

        **Structural counts (computed by the Phase 2 parser).**

        * `n_scenes`, `n_unique_characters`, `n_dialogue_lines`,
          `total_dialogue_chars`, `total_action_chars`,
          `parse_warning_count`. Six right-skewed counts whose raw
          values exhibit substantial heavy tails (a single
          large-cast or long-dialogue film sits multiple standard
          deviations above the median in the raw scale). The
          revised feature configuration (Section 5) applies
          `log1p` to these six columns before z-scoring.
        * `dialogue_to_total_text_ratio`. A bounded ratio in
          [0, 1] with mean approximately 0.40. Already
          well-conditioned; no log transform is applied.

        **Era and genre.**

        * `release_year_parsed`. Treated as a numeric feature.
        * `primary_genre_bucketed` one-hot encoded. Thirteen
          dummies after the Phase 2 bucketing step.

        **Runtime (revised configuration only).**

        * `log_runtime = log1p(runtime)`. Runtime in minutes is
          present on every film in the corpus (median 112,
          standard deviation 20). Runtime is leak-free at the
          deployment scenario the project targets, since a
          script's intended runtime is implicit in its page count
          (industry convention, approximately one page per
          minute).

        **Sanity-check addition.**

        * `log_budget`. Already on the master parquet as
          `log1p(budget)`. Included only in the with-budget
          ceiling configuration to document the predictive ceiling
          that budget knowledge alone provides. Not deployable at
          inference time, since at the pre-greenlight moment the
          system is meant to support, the budget is precisely the
          decision the studio is trying to inform. The dialogue-
          only configuration is the headline number.

        ### 4.3 Implementation

        The feature builder lives in
        `src/features/baseline_features.py`. Knobs on
        :class:`BaselineFeatureConfig` control which features are
        included and whether the heavy-tailed structural counts are
        log-transformed.
    """),
    code("""
        from src.features.baseline_features import (
            BaselineFeatureConfig, LOG_TRANSFORMABLE,
            build_baseline_features,
        )

        feature_cfg_original = BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=False,
            include_log_runtime=False,
        )
        X_original = build_baseline_features(df_with_targets, feature_cfg_original)
        print(f"Original feature matrix: {X_original.shape}")
        print(f"  Columns: {list(X_original.columns[:5])}, ..., "
              f"{list(X_original.columns[-3:])}")
    """),
    md("""
        The original feature matrix has twenty-one columns: seven
        structural counts, `release_year_parsed`, and thirteen
        genre dummies. Row order matches the master parquet; the
        index is `imdb_id` for downstream alignment by identifier
        rather than by position.
    """),

    # ============================================================
    # 5. Heavy-tailed structural counts and the log1p transform
    # ============================================================
    md("---\n\n## 5. Heavy-tailed structural counts and the `log1p` transform"),
    md("""
        ### 5.1 Motivation

        Six of the seven structural counts in Section 4.2 exhibit
        substantial right-skew on the corpus. When a linear
        regression or logistic regression is fitted on z-scored
        raw values of such columns, the heavy tail dominates the
        gradient and the bulk of the distribution receives little
        attention from the model. Compressing the tail with a log
        transform before z-scoring produces a more balanced feature
        scale and is standard practice for count-valued predictors.

        The `log1p` form (`log(1 + x)`) is preferred over `log` for
        defensive consistency: `parse_warning_count` is zero on the
        majority of films in the corpus, and a bare `log` would
        propagate `-inf`. The set of columns to which `log1p` is
        applied is fixed in code as the constant
        `LOG_TRANSFORMABLE`; `dialogue_to_total_text_ratio` is
        excluded because it is already a bounded ratio.

        ### 5.2 Empirical effect of the transform
    """),
    code("""
        from src.features.baseline_features import LOG_TRANSFORMABLE

        feature_cfg_revised = BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
        )
        X_revised = build_baseline_features(df_with_targets, feature_cfg_revised)
        print(f"Revised feature matrix: {X_revised.shape}")
        print(f"  Log-transformed columns: {sorted(LOG_TRANSFORMABLE)}")
        print(f"  Added runtime feature:   log_runtime")
    """),
    code("""
        # Visualize the transform's effect on three representative columns.
        cols_to_show = ["n_dialogue_lines", "n_unique_characters", "parse_warning_count"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for j, col in enumerate(cols_to_show):
            raw = df_with_targets[col].astype(float)
            logged = np.log1p(raw)
            axes[0, j].hist(raw, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
            axes[0, j].set(title=f"{col} (raw)", ylabel="Films" if j == 0 else "")
            axes[0, j].grid(axis="y", linestyle=":", alpha=0.5)
            axes[1, j].hist(logged, bins=50, color="seagreen", edgecolor="black", alpha=0.85)
            axes[1, j].set(title=f"log1p({col})", ylabel="Films" if j == 0 else "",
                           xlabel="value")
            axes[1, j].grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle("Heavy-tailed structural counts: raw (top) vs log1p (bottom)")
        fig.tight_layout()
        fig.savefig(paths.REPORTS_FIGURES_DIR / "phase3_log_transform_effect.png", dpi=120)
        plt.show()
    """),
    md("""
        The raw distributions are right-skewed with isolated heavy-
        tail observations. The log-transformed distributions are
        more symmetric and unimodal, which produces a feature scale
        that is more compatible with z-score standardization and
        with the ridge / logistic-regression regularization paths
        used in the baseline. The most pronounced effect is on
        `parse_warning_count`, where the raw distribution is
        dominated by a heavy left spike at zero plus a thin tail of
        warning-heavy films, while the log-transformed distribution
        is closer to a bimodal mixture (zero versus non-zero) which
        the z-score subsequently handles more gracefully.
    """),

    # ============================================================
    # 6. Baseline training
    # ============================================================
    md("---\n\n## 6. Baseline training"),
    md("""
        ### 6.1 Model families

        The two model families are chosen as deliberately simple
        "constant comparators" for Phase 3b to exceed.

        * **Regression on `log_roi`: Ridge regression with
          cross-validated alpha** (`sklearn.linear_model.RidgeCV`,
          alpha grid `np.logspace(-3, 3, 13)`). Alpha is selected
          via leave-one-out generalized cross-validation, which is
          numerically stable and parameter-free at the level of CV
          fold count. Ridge is the standard linear baseline in
          regression contexts and is robust to the modest
          multicollinearity introduced by the genre dummies.
        * **Classification on `roi_gt_1` and `roi_gt_2`: logistic
          regression with L2 penalty and cross-validated C**
          (`sklearn.linear_model.LogisticRegressionCV`, C grid
          `np.logspace(-3, 3, 13)`). C is selected via inner 5-fold
          stratified cross-validation optimizing AUC-ROC. Same
          penalty family as the regression baseline supports a
          clean ablation comparison.

        Both models are wrapped in a `Pipeline` with
        `StandardScaler` applied to the numeric columns and
        passthrough on the genre one-hot dummies. Scaling occurs
        inside the cross-validation folds to avoid leakage.

        ### 6.2 Cross-validation

        Five-fold cross-validation is run on the training split
        only (n = 1,199). `KFold` is used for regression;
        `StratifiedKFold` is used for classification to ensure
        each fold reflects the global positive-class rate.
        Out-of-fold predictions are concatenated across folds and
        used for both the headline metric and the bootstrap CI.

        ### 6.3 Bootstrap confidence intervals

        Each metric is reported with a percentile-bootstrap 95%
        confidence interval (1,000 resamples, seed 42). For
        AUC-ROC, bootstrap samples that contain only a single
        class (rare at this sample size, possible for the
        80%-positive `roi_gt_1` target) are skipped rather than
        propagating `nan` into the CI bounds. This is the standard
        practice for bootstrap CI's on AUC-style metrics.
    """),

    md("### 6.4 Run the baseline trainer"),
    md("""
        The trainer (`src.models.baseline.train`) iterates the
        three targets across each of four feature configurations:
        original dialogue-only, original with-budget,
        revised dialogue-only (log-transformed structural counts +
        `log_runtime`), and revised with-budget. The output table
        has twenty-eight rows
        (4 configurations × {3 + 2 + 2} target-metric pairs) and
        is saved to `reports/tables/phase3a_baseline.csv`.
    """),
    code("""
        from src.models.baseline.train import (
            BaselineTrainConfig, evaluate_feature_set,
        )

        train_cfg = BaselineTrainConfig()
        train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
        df_train = df_with_targets[df_with_targets["imdb_id"].isin(train_ids)].reset_index(drop=True)
        print(f"Training set: {len(df_train):,} films")
    """),
    code("""
        rows: list[dict] = []
        for set_name, feat_cfg in [
            ("dialogue_only", BaselineFeatureConfig(include_log_budget=False)),
            ("with_budget",   BaselineFeatureConfig(include_log_budget=True)),
            ("dialogue_only_logged", BaselineFeatureConfig(
                include_log_budget=False,
                log_transform_structural=True,
                include_log_runtime=True,
            )),
            ("with_budget_logged",   BaselineFeatureConfig(
                include_log_budget=True,
                log_transform_structural=True,
                include_log_runtime=True,
            )),
        ]:
            rows.extend(evaluate_feature_set(df_train, feat_cfg, train_cfg, set_name=set_name))
        baseline = pd.DataFrame(rows)
        print(f"Baseline rows produced: {len(baseline)}")
    """),

    # ============================================================
    # 7. Results
    # ============================================================
    md("---\n\n## 7. Results"),
    md("### 7.1 Original configuration"),
    code("""
        wide_original = (
            baseline[baseline["feature_set"] == "dialogue_only"]
            .pivot_table(index="target", columns="metric", values="value")
            .round(4)
        )
        wide_original
    """),
    md("""
        The original dialogue-only baseline meets the brief's
        escalation thresholds (R² ≥ 0.05, AUC-ROC ≥ 0.55) on every
        target, but only marginally. The 95% CI lower bounds dip
        below the thresholds for the two weaker targets
        (regression R² lower bound 0.024; `roi_gt_1` AUC-ROC lower
        bound 0.521). The brief gates on point estimate, so the
        thresholds are technically cleared, but the borderline
        nature of the floor motivates the configuration revision
        in the next subsection.
    """),

    md("### 7.2 Revised configuration"),
    code("""
        wide_revised = (
            baseline[baseline["feature_set"] == "dialogue_only_logged"]
            .pivot_table(index="target", columns="metric", values="value")
            .round(4)
        )
        wide_revised
    """),
    md("""
        Two changes compose the revision: `log1p` is applied to the
        six heavy-tailed structural counts before z-scoring
        (Section 5), and `log_runtime` is added to the deployable
        feature set. Each change is independently defensible and
        the two together produce the revised numbers above.

        The lift pattern is uneven across targets, which is itself
        informative. `roi_gt_2` AUC-ROC lifts approximately
        twenty basis points to 0.602, with a 95% CI lower bound
        of 0.572 that fully clears the 0.55 floor. This is the
        only target in the baseline whose entire confidence
        interval sits above the floor. `log_roi` MAE drops by
        seven thousandths in log units, consistent with the log
        transforms producing a more honest regression scale.
        `roi_gt_1` is unchanged within CI noise; the
        80%-positive base rate makes this the most signal-thin of
        the three targets and a more honest feature scale does not
        help when the underlying signal is weak.
    """),

    md("### 7.3 Side-by-side comparison"),
    code("""
        comparison_rows = []
        for target in ["log_roi", "roi_gt_1", "roi_gt_2"]:
            for metric in ("r2", "mae", "rmse", "auc_roc", "pr_auc"):
                row_orig = baseline[(baseline["feature_set"] == "dialogue_only")
                                    & (baseline["target"] == target)
                                    & (baseline["metric"] == metric)]
                row_rev = baseline[(baseline["feature_set"] == "dialogue_only_logged")
                                   & (baseline["target"] == target)
                                   & (baseline["metric"] == metric)]
                if row_orig.empty or row_rev.empty:
                    continue
                v_orig = float(row_orig["value"].iloc[0])
                v_rev = float(row_rev["value"].iloc[0])
                comparison_rows.append({
                    "target": target,
                    "metric": metric,
                    "original": round(v_orig, 4),
                    "revised":  round(v_rev, 4),
                    "delta":    round(v_rev - v_orig, 4),
                })
        comparison = pd.DataFrame(comparison_rows)
        comparison
    """),

    md("### 7.4 With-budget sanity ceiling"),
    code("""
        ceiling = (
            baseline[baseline["feature_set"].isin(["with_budget", "with_budget_logged"])]
            .pivot_table(index=["feature_set", "target"], columns="metric", values="value")
            .round(4)
        )
        ceiling
    """),
    md("""
        Adding `log_budget` raises regression R² from 0.099 in the
        revised dialogue-only configuration to a higher value in
        the revised with-budget configuration, but the absolute
        gain remains modest, and AUC-ROC on the classification
        targets is essentially unchanged. The interpretation
        follows directly from the corpus's survivorship structure:
        approximately 80% of the corpus is gross-profitable
        (`DATA_NOTES.md`, Section 2), so films with different
        budgets share the "made it onto a major aggregator"
        selection. Budget alone does not separate hits from misses
        within the survived population.

        For the project framing this is a useful diagnostic.
        Phase 3b's dialogue features are therefore not competing
        against an obvious dominant budget signal; whatever lift
        each engineered group contributes is genuinely incremental
        information from the screenplay text.
    """),

    # ============================================================
    # 8. Threshold check
    # ============================================================
    md("---\n\n## 8. Threshold check"),
    md("""
        The Phase 3 brief defines an escalation point at the end of
        Task 2: if the dialogue-only baseline R² (regression) is
        below 0.05 or the AUC-ROC (classification) is below 0.55
        on cross-validation across all three targets, the executing
        chat must pause and escalate to the planning conversation.
        The check is applied to the revised dialogue-only
        configuration (the new floor adopted by the planning
        conversation 2026-05-03).
    """),
    code("""
        from src.models.baseline.train import _check_thresholds

        triggers = _check_thresholds(rows)
        any_triggered = any(triggers.values())
        print(f"Escalation thresholds tripped: {any_triggered}")
        for key, hit in sorted(triggers.items()):
            mark = "FAIL" if hit else "ok  "
            print(f"  [{mark}] {key}")
    """),
    md("""
        No escalation is triggered. All three deployable point
        estimates clear the brief's floor on the revised
        configuration. The original configuration also passes the
        check by a smaller margin and is retained in
        `phase3a_baseline.csv` for the report's before/after
        comparison.
    """),

    # ============================================================
    # 9. Interpretation
    # ============================================================
    md("---\n\n## 9. Interpretation"),
    md("""
        Five points summarize the Phase 3a findings.

        **The baseline floor lands where prior work suggests it
        should.** Linear baselines on screenplay-structural
        features alone establish a regression R² of approximately
        0.05 on `log_roi` and AUC-ROC values in the 0.56-0.60 band
        across the two classification targets. Published
        screenplay-based predictive work on adjacent tasks
        (audience rating, Oscar nomination probability) reports
        values in this same magnitude band when restricted to
        non-text features, which corroborates the floor as
        non-trivially above chance but well below the headroom
        required for production deployment.

        **`roi_gt_2` is comfortably the most tractable target.**
        Post-revision AUC-ROC is 0.602 with 95% CI lower bound
        0.572. The "net-profitable" distinction tracks observable
        features (Action and Animation lean blockbuster, small
        genres lean sub-2x) more cleanly than the
        gross-profitable distinction does. Phase 3b feature
        groups are expected to benefit `roi_gt_2` and `log_roi`
        more than `roi_gt_1`.

        **Budget knowledge alone barely lifts deployable
        performance.** The with-budget ceiling adds a modest R²
        gain over the dialogue-only floor, with no meaningful
        AUC-ROC change. Survivorship bias accounts for this:
        within the population of films that were both produced
        and recognized enough to appear in major aggregators,
        budget does not differentiate hits from misses. Phase 3b
        screenplay features have meaningful headroom rather than
        competing against a dominant signal.

        **PR-AUC for `roi_gt_1` is misleading on its own.** The
        revised dialogue-only PR-AUC for `roi_gt_1` is
        approximately 0.85, which superficially appears strong.
        However, the 80%-positive base rate sets the
        random-guess PR-AUC near 0.80, so the model's lift over
        random is approximately five PR-AUC points. AUC-ROC is
        the more honest summary for this target.

        **The `parse_warning_count` ↔ `n_unique_characters`
        redundancy is preserved.** Both columns appear in the
        baseline feature set despite their Spearman ρ ≈ +0.39
        documented in Phase 2. The brief's "earned its place"
        criterion does not yet apply (Phase 3b is the
        feature-selection phase); the redundancy is documented
        in `FEATURE_NOTES.md` for Phase 4 to handle if it
        impacts model behaviour.
    """),

    # ============================================================
    # 10. Outputs and next phase
    # ============================================================
    md("---\n\n## 10. Phase 3a outputs and Phase 3b prerequisites"),
    md("""
        ### 10.1 Outputs

        * `data/processed/split_assignments.parquet`: one row per
          film with columns `imdb_id`, `stratum`, `split`. The
          authoritative split definition for every downstream
          phase.
        * `reports/tables/phase3_split_diagnostics.csv`:
          per-stratum split counts; fifty-seven strata, every
          named stratum with at least one film in each split.
        * `reports/tables/phase3a_baseline.csv`: twenty-eight
          rows covering the four feature configurations × three
          targets × {three regression metrics, two classification
          metrics}. Original (pre-revision) rows preserved for
          the report's before/after comparison.
        * `reports/figures/phase3_target_distributions.png`:
          target-distribution diagnostics referenced in Section
          3.3.
        * `reports/figures/phase3_log_transform_effect.png`:
          before/after `log1p` on three representative
          structural counts, referenced in Section 5.2.

        ### 10.2 Prerequisites for Phase 3b

        Phase 3b begins with a written proposal at
        `docs/proposals/phase3_lexical_proposal.md`. The proposal
        pre-registers expected lift on each of the three targets
        and lists the specific lexical features under
        consideration. The planning conversation reviews the
        proposal against a literature-reference document before
        implementation begins. The same proposal-then-implement
        cycle is repeated for the four remaining feature groups
        (sentiment, topic, embedding, character network).

        The `data_quality_flag` films from Phase 2 (thirty films
        with degenerate source-XML scene structure) are kept in
        the train, calibration, and test splits for sample-size
        reasons but are excluded from feature groups whose
        validity depends on scene-level integrity (character
        network features in particular). Per-feature handling
        will be documented in `docs/FEATURE_NOTES.md` as each
        Phase 3b group lands.

        ### 10.3 Forward expectations

        Phase 3b feature groups are anticipated to lift dialogue-
        only `log_roi` R² into the 0.10-0.20 band and `roi_gt_2`
        AUC-ROC into the 0.65-0.72 band, by reference to
        published work on similar screenplay-based prediction
        tasks. Lift on `roi_gt_1` is expected to be small
        regardless of feature group, given the 80%-positive base
        rate. Substantial deviation from these expectations,
        either above or below, is grounds for surfacing to the
        planning conversation before the next group's
        implementation.
    """),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUTPUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
