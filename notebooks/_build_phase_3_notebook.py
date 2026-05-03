"""Build ``notebooks/phase_3.ipynb`` from a structured cell list.

Regenerate the notebook with:

    python -m notebooks._build_phase_3_notebook

The Phase 3 notebook documents feature extraction. The current build
covers the baseline step (Section A): the train / calibration / test
split and a simple linear floor on screenplay-structural features and
metadata. The incremental feature-engineering step (Section B,
covering lexical, sentiment, topic, embedding, and character-network
features) is appended to this same builder as each group lands.
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
        # Phase 3: Feature Extraction

        ## Where this phase fits in the project

        Studios decide which scripts to greenlight under sharp cost
        asymmetry: producing a flop costs roughly fifty million
        dollars in lost budget, while passing on a hit can cost two
        to four times that in foregone revenue. At the moment of
        the greenlight decision, most predictors of success (cast,
        marketing spend, the budget itself) are still unknown. The
        only signal that exists is the script.

        This project trains a triage model that reads a screenplay's
        dialogue and outputs a recommendation (greenlight, pass, or
        refer to a human reader) together with a calibrated
        confidence interval and an explanation of which scenes
        drove the recommendation. The pipeline has four layers
        stacked on a single core predictive model: the model
        itself, calibrated uncertainty around its predictions, an
        asymmetric-cost decision rule on top of those predictions,
        and scene-level explanations.

        Phase 1 verified that we have enough usable data, and
        Phase 2 produced a clean processed corpus of 1,713 films
        with screenplays, IMDb ratings, budgets, and revenues. This
        notebook documents Phase 3, which converts each screenplay
        into a fixed-length feature vector that the modelling phase
        will consume.

        ## How Phase 3 is organized

        Feature engineering carries methodological risk. Adding many
        features at once makes it impossible to tell which ones
        actually help and which ones are noise. To avoid that, we
        split Phase 3 into two parts.

        **Part A (this notebook).** Establish a performance floor
        using only features that already exist on the processed
        corpus, with a deliberately simple linear model. The floor
        answers a basic question: how well can we predict film
        success from screenplay structure alone, before any text
        engineering? Without that number, we have nothing to
        measure subsequent feature engineering against.

        **Part B (added later).** Add five engineered feature
        groups one at a time (lexical, sentiment, topic, embedding,
        character network), retraining the same baseline after each
        group and recording the lift each contributes. Each group
        is preceded by a written prediction of expected lift, then
        compared against the actual lift. This produces a clean
        ablation table for the report and prevents the team from
        retrofitting explanations to whatever happened to work.

        Part A is what this notebook documents. Part B sections will
        be appended once each feature group is implemented.

        ## What we predict

        Three prediction targets are tracked in parallel. The choice
        of which one to feature in the final report is deferred
        until after the modelling phase, when comparative results
        are available.

        * `log_roi`, the natural log of return-on-investment,
          defined as `ln(revenue) - ln(budget)`. A regression
          target.
        * `roi_gt_1`, a boolean indicator for revenue greater than
          budget (gross profitability). A classification target.
        * `roi_gt_2`, a boolean indicator for revenue greater than
          twice budget. The doubling threshold is the industry
          rule of thumb for net profitability after marketing and
          distribution overhead. A classification target.

        The three are constructed so they share threshold
        consistency: `roi_gt_1` is the same as `log_roi > 0`, and
        `roi_gt_2` is the same as `log_roi > ln 2`. A single
        regression on `log_roi` therefore reproduces both
        classifiers by thresholding, which makes downstream
        comparisons clean.
    """),

    # ============================================================
    # 0. Setup
    # ============================================================
    md("## 0. Environment setup"),
    md("""
        The first cell finds the project root, adds it to
        `sys.path` so package imports work regardless of where the
        notebook is opened, and turns on inline plotting and
        module auto-reloading.
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
    md("Common imports for the rest of the notebook."),
    code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from src.utils import paths

        paths.ensure_dirs()
        print("Processed data:  ", paths.DATA_PROCESSED_DIR)
        print("Reports figures: ", paths.REPORTS_FIGURES_DIR)
        print("Reports tables:  ", paths.REPORTS_TABLES_DIR)
    """),

    # ============================================================
    # 1. Three prediction targets
    # ============================================================
    md("---\n\n## 1. Defining the three prediction targets"),
    md("""
        Before we can split the data or train any model, we need
        to be precise about what we are predicting. This section
        constructs the three target columns and inspects their
        distributions.

        ### Why log of ROI for the regression target

        Return-on-investment in raw form is heavily right-skewed.
        On this corpus the median is around 2.9, but the maximum
        approaches 8500, and the distribution has a thick upper
        tail that pulls the mean far above the median. Standard
        regression models trained against squared-error loss are
        dominated by such tails, fitting the few extreme values at
        the expense of the bulk of the data. Taking the logarithm
        compresses the tail and produces a distribution that is
        approximately symmetric around its median, which satisfies
        the standard regression assumptions far better.

        Two further reasons motivate the log form specifically.
        First, it decomposes cleanly: `log(revenue/budget)` equals
        `log(revenue) - log(budget)`, both of which are already
        stored on the processed corpus. The modelling phase can
        therefore choose to model the joint quantity directly or to
        model the two components separately. Second, the
        log-transformed target has direct correspondence with the
        two classification thresholds (zero and `ln 2`), so a
        single regression model produces both classifiers by
        thresholding.

        ### Why two classification thresholds rather than one

        The naive choice of "profitable versus not profitable" uses
        the threshold revenue equals budget, which corresponds to a
        gross-profitable cutoff. On this corpus that threshold is
        positive for roughly 80 percent of films. A target with
        such a skewed positive class is easy to game with a "always
        predict positive" model and gives little headroom for
        features to demonstrate predictive lift.

        The doubling threshold (revenue greater than twice budget)
        captures the industry's rule of thumb for net profitability
        after marketing and distribution overhead. On this corpus
        it produces a more balanced 64 percent positive rate, which
        gives more discriminating power to the AUC and PR metrics.
        Tracking both thresholds in parallel lets us compare which
        notion of success the screenplay carries information about.
    """),
    code("""
        from src.features.targets import (
            ALL_TARGETS, LOG_ROI_COL, ROI_GT_1_COL, ROI_GT_2_COL,
            add_targets,
        )

        df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
        df_with_targets = add_targets(df)
        print(f"Corpus: {len(df_with_targets):,} films")
        for col in ALL_TARGETS:
            print(f"  {col:<12}  dtype: {str(df_with_targets[col].dtype):<8}")
    """),
    code("""
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        log_roi = df_with_targets[LOG_ROI_COL]
        axes[0].hist(log_roi, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
        axes[0].axvline(0, color="black", linewidth=1, linestyle="--", label="ROI = 1 (gross-profitable)")
        axes[0].axvline(np.log(2), color="firebrick", linewidth=1, linestyle="--", label="ROI = 2 (net-profitable)")
        axes[0].set(title=f"log_roi  (median {log_roi.median():.2f})", xlabel="log_roi", ylabel="Films")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)

        rates = {
            "roi_gt_1": df_with_targets[ROI_GT_1_COL].mean(),
            "roi_gt_2": df_with_targets[ROI_GT_2_COL].mean(),
        }
        bars = axes[1].bar(list(rates.keys()), list(rates.values()),
                           color=["steelblue", "indianred"], edgecolor="black", alpha=0.85)
        axes[1].set(title="Positive-class rate", ylabel="Fraction positive", ylim=(0, 1))
        for bar, rate in zip(bars, rates.values()):
            axes[1].text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                         f"{rate:.1%}", ha="center", fontsize=10)
        axes[1].axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)

        cross = pd.crosstab(df_with_targets[ROI_GT_1_COL], df_with_targets[ROI_GT_2_COL])
        cross = cross.reindex(index=[False, True], columns=[False, True], fill_value=0)
        axes[2].imshow(cross.values, cmap="Blues", aspect="auto")
        axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["roi_gt_2 = F", "roi_gt_2 = T"])
        axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["roi_gt_1 = F", "roi_gt_1 = T"])
        axes[2].set(title="Threshold consistency check")
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f"{cross.values[i, j]:,}", ha="center", va="center",
                             color="white" if cross.values[i, j] > cross.values.max() / 2 else "black")

        fig.tight_layout()
        fig.savefig(paths.REPORTS_FIGURES_DIR / "phase3_target_distributions.png", dpi=120)
        plt.show()
    """),
    md("""
        The left panel shows the regression target. Its symmetry
        around the median value of roughly 1.06 (corresponding to a
        revenue of about 2.9 times budget) confirms that the log
        transform produced a distribution suitable for linear
        regression. The two dashed lines mark the locations of the
        classification thresholds.

        The middle panel shows the positive-class rates for the two
        classification targets. The 80 percent positive rate on
        `roi_gt_1` reflects survivorship in the corpus: every film
        in the dataset was both produced and recognized enough to
        appear on a major metadata aggregator, which selects for
        success. We discuss the implications for the cost-decision
        layer of the system in a later phase. The 64 percent
        positive rate on `roi_gt_2` is closer to balanced and gives
        the model more room to discriminate.

        The right panel verifies the threshold-consistency
        construction. Every film with `roi_gt_2 = True` also has
        `roi_gt_1 = True` (the upper-right cell is empty as
        expected), and the proportions across cells match the
        cumulative ROI distribution.
    """),

    # ============================================================
    # 2. The split
    # ============================================================
    md("---\n\n## 2. Splitting the corpus"),
    md("""
        We need to evaluate predictive performance honestly, which
        means setting aside data the model never sees during
        training. Two considerations shape the split design.

        ### Why three-way rather than two-way

        A standard train and test split would suffice for the core
        predictive model. The downstream calibration layer of the
        triage system, however, requires a separate pool of data
        the model has not seen, used to fit conformal-prediction
        intervals. Carving that calibration set out now, before any
        feature fitting touches the data distribution, ensures we
        do not have to redo the split later and avoids leakage
        across the calibration step.

        We use a 70 / 15 / 15 split: about 1,200 films for training,
        257 for calibration, and 257 for held-out testing. The 15
        percent calibration set is large enough to produce stable
        conformal intervals, and the same size for the test set
        gives sufficient power to detect meaningful differences
        between candidate models in the final evaluation.

        ### Why stratified

        Two properties of the corpus would create a noisy split if
        we sampled at random. First, the genre distribution has a
        long tail: Drama, Comedy, Action, and Thriller dominate
        while smaller genres each contain only a few dozen films.
        A random split could end up with all the Animation films
        in the test set and none in the training set. Second, the
        corpus spans nine decades with very uneven density: the
        2000s and 2010s contain hundreds of films while the
        pre-1980s decades each contain fewer than thirty. Both
        factors plausibly affect the relationship between
        screenplay features and outcomes, so we want to balance
        them across splits.

        We therefore stratify by the cross of primary genre and a
        coarse decade bucket. Pre-1980 decades are pooled into a
        single bucket because each is too thin on its own. The
        2010s and 2020s are pooled because the 2020s coverage in
        the corpus extends only to 2023.

        ### Handling rare cells

        The cross of genre and decade bucket produces some cells
        that are too small for stratification to function (for
        example, fewer than five Animation films in the
        pre-1980s). We pool any cell with fewer than five films
        into a single rare bucket so that the stratifier sees a
        well-defined population for every named cell. Roughly
        thirty-eight films land in that pool, around two percent of
        the corpus.

        The split is implemented in `src/features/split.py`. All
        knobs (target proportions, decade boundaries, rare-cell
        threshold, random seed) live on a configuration dataclass
        so alternative split designs can be tried without changing
        any code.
    """),
    code("""
        from src.features.split import (
            SplitConfig, make_splits, split_diagnostics,
        )

        config = SplitConfig()
        print("Split configuration:")
        print(f"  train / calibration / test:  {config.train_frac} / "
              f"{config.cal_frac} / {config.test_frac}")
        print(f"  rare-cell threshold:         {config.rare_cell_threshold}")
        print(f"  random seed:                 {config.seed}")
    """),
    code("""
        splits = make_splits(df, config)
        counts = splits["split"].value_counts().reindex(["train", "cal", "test"])

        print("Resulting split sizes:")
        for name, n in counts.items():
            label = {"train": "Train", "cal": "Calibration", "test": "Test"}[name]
            print(f"  {label:<13} {n:>5,}   ({100 * n / len(df):.1f}% of corpus)")
    """),
    md("""
        We verified reproducibility by running the split twice with
        the default configuration; the assignments are
        byte-identical across runs. The result is saved to
        `data/processed/split_assignments.parquet` so every
        downstream step uses the same definitive partition.
    """),

    md("### Stratum diagnostics"),
    code("""
        diagnostics = split_diagnostics(splits)
        print(f"Number of strata used: {len(diagnostics)}")
        rare_count = int(diagnostics.loc[diagnostics['stratum'] == 'rare|rare', 'total'].sum())
        print(f"Films pooled into the rare bucket: {rare_count}")
        print()
        print("Top 10 strata by size:")
        diagnostics.head(10)
    """),
    md("""
        The full diagnostic table is saved to
        `reports/tables/phase3_split_diagnostics.csv`. Every named
        stratum (genre, decade pair) has at least one film in each
        of the three splits, confirming that the stratifier
        functions as intended. The rare bucket absorbs the long
        tail of small cells without collapsing the rest of the
        corpus into less granular strata.
    """),

    # ============================================================
    # 3. Baseline: features
    # ============================================================
    md("---\n\n## 3. Choosing features for the baseline"),
    md("""
        The point of the baseline is to establish how well a
        simple model performs without any text-derived feature
        engineering. We need to select features that meet two
        criteria: they should already exist on the processed
        corpus (or be trivially derivable), and they should be
        available at the moment the system would actually be used.
        The second criterion is important: the system is meant to
        run before the studio greenlights the film, so any feature
        that depends on the studio's eventual decisions (budget,
        marketing, cast) is not available at the moment of
        prediction.

        ### Features used in the deployable baseline

        **Screenplay-structural features.** Phase 2's parser
        extracted seven aggregate measures of each screenplay's
        structure from its XML form: number of scenes, number of
        unique characters, number of dialogue lines, total
        character counts for dialogue and for action description,
        the dialogue-to-total-text ratio, and a count of structural
        irregularities encountered while parsing. These are all
        properties of the screenplay itself, available at
        pre-greenlight time.

        **Release year.** Year of release encoded as a numeric
        feature. We expect it to carry a small amount of signal
        about era-specific market dynamics. Although the actual
        release year is not known until the film is released, the
        year of submission is a reasonable proxy and is always
        available.

        **Primary genre, one-hot encoded.** Genre is an obvious
        confound and a strong correlate of both budget and
        revenue. Phase 2 grouped genres with fewer than thirty
        films into an "Other" category to keep cell sizes
        well-conditioned, leaving thirteen genre dummies.

        **Runtime.** A film's intended runtime in minutes. The
        screenplay's page count is a tight proxy for runtime
        (industry convention is roughly one page per minute), so
        runtime is leak-free at the pre-greenlight moment even
        though the final theatrical runtime might differ
        slightly. We use the log-transformed form for consistency
        with the other heavy-tailed counts.

        ### A separate ceiling baseline includes budget

        Budget is the central decision the system is meant to
        inform, and is not available at inference time. We
        therefore exclude it from the deployable baseline. It is
        useful, however, to train a parallel model that includes
        log-budget, purely as a diagnostic. If budget alone
        produced a strong baseline, it would mean the deployable
        model is competing against a dominant signal it cannot
        access. If budget contributes little, it suggests the
        deployable features have meaningful headroom. We discuss
        what we found below.
    """),

    # ============================================================
    # 4. Heavy-tailed counts: the log transform
    # ============================================================
    md("---\n\n## 4. Why we apply a log transform to several features"),
    md("""
        Several of the structural counts above are heavily
        right-skewed. A film with an unusually large cast or an
        unusually long screenplay sits multiple standard
        deviations above the median, while most films sit close to
        the median. When a linear regression is fit on z-scored
        raw values of such a column, the heavy-tail observations
        dominate the gradient and the model effectively learns to
        fit them well while ignoring the bulk of the data.

        The standard fix is to compress the tail with a logarithm
        before standardization. We use the form `log(1 + x)`
        rather than `log(x)` because one of the features
        (`parse_warning_count`) takes the value zero on the
        majority of films, and the bare log would propagate
        negative infinity. The `log(1 + x)` form is well-defined
        at zero and is approximately equal to `log(x)` for the
        non-zero values where the compression is needed.

        The log transform is applied to six of the seven
        structural counts: `n_scenes`, `n_unique_characters`,
        `n_dialogue_lines`, `total_dialogue_chars`,
        `total_action_chars`, and `parse_warning_count`. The
        seventh, `dialogue_to_total_text_ratio`, is already a
        bounded proportion in the interval zero to one and does
        not benefit from log transformation.

        The figure below shows the effect of the transform on
        three representative columns.
    """),
    code("""
        from src.features.baseline_features import LOG_TRANSFORMABLE

        cols_to_show = ["n_dialogue_lines", "n_unique_characters", "parse_warning_count"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for j, col in enumerate(cols_to_show):
            raw = df_with_targets[col].astype(float)
            logged = np.log1p(raw)
            axes[0, j].hist(raw, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
            axes[0, j].set(title=f"{col} (raw)", ylabel="Films" if j == 0 else "")
            axes[0, j].grid(axis="y", linestyle=":", alpha=0.5)
            axes[1, j].hist(logged, bins=50, color="seagreen", edgecolor="black", alpha=0.85)
            axes[1, j].set(title=f"log(1 + {col})",
                           ylabel="Films" if j == 0 else "", xlabel="value")
            axes[1, j].grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle("Heavy-tailed structural counts: raw form (top row) versus log form (bottom row)")
        fig.tight_layout()
        fig.savefig(paths.REPORTS_FIGURES_DIR / "phase3_log_transform_effect.png", dpi=120)
        plt.show()
    """),
    md("""
        The raw distributions exhibit pronounced right skew with
        thin tails reaching far above the bulk. The log-transformed
        versions are closer to symmetric and unimodal, which is
        the shape a linear model with z-score standardization is
        designed to handle. The transform is most consequential
        for `parse_warning_count`, where the raw distribution is
        dominated by a heavy spike at zero with a thin tail of
        warning-heavy films, and the log form spreads the non-zero
        observations across a more interpretable range.
    """),

    md("### Building the feature matrix"),
    code("""
        from src.features.baseline_features import (
            BaselineFeatureConfig, build_baseline_features,
        )

        feature_cfg = BaselineFeatureConfig(
            include_log_budget=False,
            log_transform_structural=True,
            include_log_runtime=True,
        )
        X_dialogue_only = build_baseline_features(df_with_targets, feature_cfg)
        print(f"Feature matrix shape: {X_dialogue_only.shape}")
        print(f"First five columns: {list(X_dialogue_only.columns[:5])}")
        print(f"Last three columns: {list(X_dialogue_only.columns[-3:])}")
    """),

    # ============================================================
    # 5. Models, CV, bootstrap
    # ============================================================
    md("---\n\n## 5. Model choice, cross-validation, and confidence intervals"),
    md("""
        ### Why four model families instead of one

        The most common approach to a baseline floor is to pick a
        single simple model family and report its numbers. We
        instead evaluate every feature configuration across four
        model families with different inductive biases. The reason
        is methodological. When we later add an engineered feature
        group and observe whether it lifts performance, we want to
        be able to tell two scenarios apart: (a) the features
        carry no signal, full stop; (b) the features carry signal
        that the chosen model family cannot extract. Without
        running multiple families, those two scenarios produce the
        same observable result and we cannot tell which is true.
        With multiple families, a feature group that lifts none of
        them is genuinely uninformative on this corpus, while a
        group that lifts only some is informative for choosing the
        modelling approach in the next phase.

        We chose four families that span four distinct paradigms:

        * **L2 linear**, the regularized linear baseline. Ridge
          regression with cross-validated alpha for the regression
          target, logistic regression with an L2 penalty and
          cross-validated regularization strength for the two
          classification targets. Captures linear additive signal.
          Robust to the modest multicollinearity that the genre
          dummies introduce.
        * **Histogram-based gradient boosting**, a tree ensemble
          that captures non-linear interactions. Conservative
          defaults: 300 boosting iterations, maximum tree depth of
          four, learning rate of 0.05, with internal early
          stopping. Handles missing values natively and is
          invariant to monotonic transforms of the inputs, so its
          numeric branch needs no preprocessing.
        * **K-nearest-neighbours**, a non-parametric instance-based
          family that captures local-neighbourhood structure.
          Twenty neighbours with distance weighting. Computes its
          scores from the standardized feature space, so it shares
          the impute-and-scale numeric pipeline with the linear
          family.
        * **Support vector machine with an RBF kernel**, a
          kernel-based family that captures non-linear signal via
          the kernel trick. Default regularization strength of one
          and the standard "scale" gamma. For classification we
          use the decision-function score for AUC computation
          rather than calibrated probabilities, since AUC depends
          only on the ordering of scores and avoiding the internal
          probability calibration cuts training time substantially.

        Standardization for the three families that need it
        happens inside each cross-validation fold so that the
        scaling parameters are estimated only on training data and
        applied to the held-out fold, avoiding leakage.

        ### Why these four and not others

        Each cell of the inductive-bias grid (linear vs non-linear,
        global vs local, parametric vs non-parametric) gets exactly
        one representative. We considered alternatives and rejected
        them: random forest sits in the same cell as gradient
        boosting and adds little diagnostic information; multilayer
        perceptrons are noisy on a corpus of 1,200 training rows
        and require seed-sensitive tuning; naive Bayes assumes
        feature independence that the corpus violates. Lasso would
        offer a feature-selection variant of the linear paradigm
        but lives in the same cell as ridge; we kept ridge because
        the threshold check the project committed to in advance is
        defined against the L2 family.

        ### Cross-validation and confidence intervals

        Five-fold cross-validation over the training split only.
        The calibration set and test set are not touched at this
        stage. For the regression target we use plain five-fold;
        for the classification targets we use stratified five-fold
        so each fold reflects the global positive-class rate,
        which matters in particular for `roi_gt_1` where the
        positive class dominates.

        Within each cross-validation iteration we record the
        out-of-fold predictions. Concatenating these across the
        five folds gives one prediction per training-set film,
        which is what the metrics are evaluated on.

        Each headline metric is reported with a 95-percent
        confidence interval computed by percentile bootstrap (1,000
        resamples, seed fixed for reproducibility). This gives a
        sense of how much the metric estimate would vary under
        re-sampling, which matters because the corpus is not
        large. For AUC, bootstrap samples that happen to contain
        only a single class are skipped rather than producing
        missing values.

        Multiple metrics are reported per target rather than a
        single headline number. For regression we report mean
        squared error, root-mean-square error, mean absolute error,
        and the coefficient of variation of the RMSE
        (RMSE divided by the absolute mean of the target, useful
        for comparing across feature configurations on the same
        scale). For classification we report AUC-ROC, PR-AUC, F1
        at the 0.5 decision threshold, and log-loss. Each metric
        captures a different aspect of model quality.

        ### Two evaluation sets reported per metric

        The trainer fits each model twice: once on the entire
        training split (the in-sample fit), and once via the
        five-fold cross-validation described above (the out-of-fold
        validation). We report metrics for both. The gap between
        in-sample and out-of-fold values indicates how much each
        model is overfitting the training data, which is a useful
        diagnostic for the modelling phase that follows.

        The held-out fifteen percent test set and the held-out
        fifteen percent calibration set are not touched by this
        trainer. They are reserved for the final evaluation phase
        and the calibration phase respectively.
    """),

    md("### Running the baseline"),
    md("""
        We train two feature configurations: a dialogue-only
        baseline (the deployable matrix at the pre-greenlight
        moment) and a sanity-check matrix that adds the log of
        budget. Each configuration is evaluated under all four
        model families on both evaluation sets.
    """),
    code("""
        from src.models.baseline.train import (
            BaselineTrainConfig, MODEL_FAMILIES, evaluate_feature_set,
        )

        train_cfg = BaselineTrainConfig()
        train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
        df_train = (
            df_with_targets[df_with_targets["imdb_id"].isin(train_ids)]
            .reset_index(drop=True)
        )
        print(f"Training set: {len(df_train):,} films")
        print(f"Model families: {', '.join(MODEL_FAMILIES)}")
    """),
    code("""
        rows: list[dict] = []
        configurations = [
            ("Dialogue only", BaselineFeatureConfig(
                include_log_budget=False,
                log_transform_structural=True,
                include_log_runtime=True,
            )),
            ("Dialogue + log_budget", BaselineFeatureConfig(
                include_log_budget=True,
                log_transform_structural=True,
                include_log_runtime=True,
            )),
        ]
        for name, fc in configurations:
            print(f"Training: {name}")
            label = "dialogue_only_logged" if not fc.include_log_budget else "with_budget_logged"
            rows.extend(evaluate_feature_set(df_train, fc, train_cfg, set_name=label))
        baseline = pd.DataFrame(rows)
        print(f"\\nBaseline rows produced: {len(baseline)} ({len(MODEL_FAMILIES)} families x 2 configs x 7 metric rows)")
    """),

    # ============================================================
    # 6. Results
    # ============================================================
    md("---\n\n## 6. Results"),
    md("### 6.1 Deployable baseline (out-of-fold) across the four model families"),
    code("""
        deployable_oof = (
            baseline[
                (baseline["feature_set"] == "dialogue_only_logged")
                & (baseline["eval_set"] == "oof")
            ]
            .pivot_table(
                index=["target", "metric"],
                columns="model_family",
                values="value",
            )
            .round(3)
        )
        deployable_oof
    """),
    md("""
        The pattern across families on out-of-fold evaluation is
        informative.

        * **Histogram gradient boosting outperforms the linear
          family on the headline targets.** RMSE on `log_roi`
          drops from 1.339 under linear to 1.327 under gradient
          boosting (lower is better), and `roi_gt_2` AUC rises
          from 0.602 to 0.610. The lift comes from non-linear
          interactions among the structural features that linear
          regression cannot capture. This is itself a finding:
          even on the structural baseline alone, the corpus
          contains interactions that a tree ensemble extracts but
          a linear model cannot.
        * **The linear family is the strongest on `roi_gt_1`,**
          the gross-profitability target, with AUC of 0.558
          versus 0.552 for gradient boosting. The 80-percent
          positive base rate makes the target signal-thin, and
          the linear family's preference for smooth decision
          boundaries appears to be the right inductive bias here.
        * **K-nearest-neighbours and the RBF support vector
          machine underperform on this corpus,** with AUC values
          near or below 0.55 on both classification targets and
          RMSE values higher than the linear and tree families.
          The likely cause is the high feature-to-sample ratio
          (around twenty-six features per film with one thousand
          two hundred training rows), which is unfavourable for
          distance-based methods.

        On the easiest target (`roi_gt_2`), the AUC of 0.610 means
        that for a randomly chosen pair of films, one of which
        doubled its budget and one of which did not, the model
        ranks them in the correct order roughly six times in ten.
        The PR-AUC of about 0.85 on `roi_gt_1` looks impressive
        but is misleading on its own: with 80 percent of films
        already in the positive class, the random-guess PR-AUC is
        around 0.80, so the model's lift over random is only
        about five PR-AUC points. F1 saturates around 0.89 across
        all families because predicting the majority class gives
        high F1 mechanically.
    """),

    md("### 6.2 Train-versus-OOF gap (overfit diagnostic)"),
    code("""
        # Restrict to the headline metrics for readability.
        gap_subset = baseline[
            (baseline["feature_set"] == "dialogue_only_logged")
            & (baseline["metric"].isin(["rmse", "auc_roc"]))
        ]
        gap = (
            gap_subset
            .pivot_table(
                index=["model_family", "target", "metric"],
                columns="eval_set",
                values="value",
            )
            .round(3)
        )
        gap["train_minus_oof"] = (gap["train"] - gap["oof"]).round(3)
        gap
    """),
    md("""
        The gap between in-sample fit and out-of-fold validation
        indicates how much each family is overfitting the training
        data. Two findings worth flagging.

        First, **gradient boosting overfits substantially despite
        conservative defaults**. Its training-fold AUC on
        `roi_gt_2` reaches roughly 0.81 while its OOF AUC is
        0.61, a gap of about 0.20. The gap on `roi_gt_1` is
        similar. This means the model is memorizing
        idiosyncrasies of the training split that do not
        generalize. The conservative defaults used here
        (`max_depth=4`, `learning_rate=0.05`, internal
        early-stopping) are not enough to prevent this on
        n = 1,200 training rows. The modelling phase that follows
        should explore even more conservative regularization
        (lower learning rate, larger minimum samples per leaf).

        Second, **the linear family is the most stable across
        the diagnostic**, with smaller absolute train-OOF gaps on
        every metric. This is consistent with linear regression's
        strong inductive bias toward smooth functions, which
        constrains the in-sample fit to remain close to the
        out-of-fold prediction.

        Reporting both eval sets surfaces dynamics that an OOF-
        only view would have hidden. The OOF numbers are still the
        right comparison point for ablation lift; the train
        numbers provide the diagnostic context that interprets the
        OOF numbers correctly.
    """),

    md("### 6.3 The with-budget sanity check (OOF)"),
    code("""
        ceiling = (
            baseline[
                (baseline["feature_set"] == "with_budget_logged")
                & (baseline["eval_set"] == "oof")
            ]
            .pivot_table(
                index=["target", "metric"],
                columns="model_family",
                values="value",
            )
            .round(3)
        )
        ceiling
    """),
    md("""
        Adding the log of budget to the same matrix lifts
        regression performance modestly across families: RMSE on
        `log_roi` drops from 1.339 to 1.305 under the linear
        baseline, and from 1.327 to 1.296 under gradient boosting
        (lower is better). AUC on the classification targets
        moves modestly with budget added.

        The interpretation has two layers. First, the result
        confirms the survivorship structure of the corpus. Every
        film in the dataset was both produced and recognized
        enough to appear on a major metadata aggregator, which
        selects strongly for success. Within that already-selected
        population, budget does not separate hits from misses with
        much precision. Second, the modest budget lift means the
        deployable model is not competing against a dominant
        budget signal it cannot access. Whatever lift the
        engineered features in Part B contribute will be
        genuinely incremental information from the screenplay
        text, not a weak proxy for budget.
    """),

    # ============================================================
    # 7. Did the baseline pass our threshold for proceeding?
    # ============================================================
    md("---\n\n## 7. Did the baseline pass the threshold for proceeding?"),
    md("""
        Before starting feature engineering, we set a minimum
        performance threshold the baseline had to meet to justify
        the investment. The reasoning is straightforward: if a
        simple model on screenplay structure cannot beat chance by
        any meaningful margin, then the dialogue-only framing of
        the project is in trouble and we should pause rather than
        spend weeks engineering features.

        The original threshold was set in terms of R-squared (at
        least 0.05 on the regression target) and AUC-ROC (at least
        0.55 on each classification target). The R-squared
        criterion has since been retired from the reported metric
        set in favour of more robust absolute and normalized
        measures (MSE, RMSE, MAE, CVRMSE). The original gating
        decision was made under the R-squared rule and remains
        valid: the linear family's out-of-fold numbers cleared
        the AUC floors and the equivalent RMSE floor at the time.
        From this point forward, ablation lift over the floor is
        the primary signal rather than absolute thresholds.

        For interested readers, the equivalent RMSE check is:
        R-squared at least 0.05 on `log_roi` translates to RMSE at
        most about 1.38 (0.975 times the standard deviation of
        the target). The linear family's OOF RMSE is 1.339, and
        gradient boosting's is 1.327, both below this translated
        threshold. The classification AUC floors (0.55) are
        cleared by the linear family on both classification
        targets; KNN and SVM fall below 0.55 on `roi_gt_1`,
        consistent with their general weakness on this corpus.
    """),

    # ============================================================
    # 8. Interpretation
    # ============================================================
    md("---\n\n## 8. Interpretation and what to expect from Part B"),
    md("""
        Seven points summarize what we have learned from Part A.

        **The baseline floor lands where we expected.** Simple
        models on screenplay structure produce RMSE around 1.33
        on `log_roi` and AUC values in the upper 0.5s to low 0.6s
        on the two classification targets. This matches floor
        baselines reported in published screenplay-based
        prediction work before any text engineering.

        **Tree ensembles beat linear regression even on the
        structural baseline alone, on out-of-fold evaluation.**
        Gradient boosting reaches OOF RMSE of 1.327 versus 1.339
        under the linear family, and `roi_gt_2` AUC of 0.610
        versus 0.602. The lift comes from non-linear interactions
        among the structural features that linear regression
        cannot capture. This is a finding in its own right: the
        next phase's model selection should expect tree-based
        families to be competitive primary candidates.

        **HistGB substantially overfits in-sample.** Train RMSE
        for HistGB drops to 1.20 against an OOF RMSE of 1.33; on
        the classification targets the train-OOF AUC gap reaches
        about 0.20. The conservative defaults used here are not
        enough to prevent the model from memorizing
        idiosyncrasies of the training split. The next phase's
        model search should explore even more conservative
        regularization for tree ensembles.

        **`roi_gt_2` is the most tractable target across all
        families.** AUC values cluster between 0.53 and 0.61
        across the four families on OOF, with the
        gradient-boosting value of 0.610 having a confidence
        interval that fully clears the 0.55 floor. The "doubled
        budget" distinction tracks observable features (Action
        and Animation films lean blockbuster, smaller-genre films
        lean below 2x) more crisply than the gross-profitable
        distinction does. Engineered features in Part B should
        benefit `roi_gt_2` and the regression target more than
        they benefit `roi_gt_1`.

        **Budget alone barely lifts deployable performance.**
        Within the survived population the corpus represents,
        budget is only weakly informative about hit-versus-miss.
        Engineered features in Part B therefore have meaningful
        room to improve on the floor without competing against an
        obvious dominant signal.

        **PR-AUC for `roi_gt_1` is misleading on its own.** The
        80-percent positive base rate sets the random-guess PR-AUC
        near 0.80, so the headline value of 0.85 corresponds to
        only about five points of actual lift. AUC-ROC is the
        more honest summary for this target.

        **What to expect from Part B.** Based on lift patterns
        reported in published screenplay-prediction work, the
        engineered feature groups should plausibly lift
        dialogue-only `log_roi` RMSE downward by 0.05 to 0.10
        units (roughly the equivalent of an R-squared lift of
        0.10 to 0.20 in the older metric vocabulary) and
        `roi_gt_2` AUC into roughly the 0.65 to 0.72 range. These
        are not targets to anchor against; they are reference
        points. The first Part B ablation (lexical features) is
        documented in the next section. Subsequent groups will be
        appended as they land.
    """),

    # ============================================================
    # 8b. First Part B group: lexical features
    # ============================================================
    md("---\n\n## 9. First Part B group: lexical features"),
    md("""
        ### What lexical features try to capture

        The lexical group introduces fourteen features intended to
        capture stylistic properties of screenplay text. They fall
        into five sub-groups:

        * **Vocabulary diversity.** A length-robust diversity
          metric on dialogue text and on action text, plus a hapax
          legomena ratio capturing the proportion of vocabulary
          that appears exactly once.
        * **Lexical sophistication.** Mean log-frequency of
          dialogue tokens against an external English frequency
          reference, plus a rare-word proportion at the bottom
          quartile of that reference.
        * **Readability.** Flesch-Kincaid grade level on dialogue
          and on action text.
        * **Length statistics.** Mean and standard deviation of
          tokens per dialogue line, and the proportion of dialogue
          lines with fewer than five tokens.
        * **Punctuation and pronouns.** Question-mark and
          exclamation-mark rates per thousand dialogue tokens,
          plus a first-to-second-person pronoun ratio that
          includes archaic forms (`thou`, `thee`, `thy`,
          `thine`, `thyself`) given that the corpus extends back
          to 1932.

        Before implementing, we pre-registered an expected lift on
        each of the three targets. The original pre-registration
        used R-squared on the regression target; with that metric
        removed, the equivalent prediction translates to an RMSE
        improvement of -0.020 to -0.010 on `log_roi` (lower is
        better). We also predicted an AUC lift between 0.000 and
        0.010 on `roi_gt_1` and between 0.015 and 0.035 on
        `roi_gt_2`. The mechanism we hypothesized was that
        vocabulary richness, sophistication, and pacing would
        carry incremental information about screenplay craft (a
        rating-style signal) and audience-targeting clarity (a
        revenue-style signal).

        ### What the data showed

        We computed the fourteen features on the full corpus,
        joined them onto the structural baseline matrix, and ran
        the same multi-family evaluation. Restricting to the
        out-of-fold numbers and the headline metrics:
    """),
    code("""
        ablation = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3_ablation.csv")
        lexical = ablation[ablation["feature_group"] == "lexical"]

        # OOF lifts on the headline metrics: RMSE for log_roi,
        # AUC-ROC for the two classification targets.
        headline = lexical[
            (lexical["eval_set"] == "oof")
            & (
                ((lexical["target"] == "log_roi") & (lexical["metric"] == "rmse"))
                | ((lexical["target"].isin(["roi_gt_1", "roi_gt_2"])) & (lexical["metric"] == "auc_roc"))
            )
        ].copy()
        headline["lift"] = headline["lift"].round(3)
        headline["phase_3a_floor"] = headline["phase_3a_floor"].round(3)
        headline["phase_3b_actual"] = headline["phase_3b_actual"].round(3)
        headline.pivot_table(
            index=["target", "metric"],
            columns="model_family",
            values="lift",
        )
    """),
    md("""
        Reading the table (each cell is the actual lift on
        out-of-fold predictions, in absolute units, of the
        lexical-augmented matrix over the Phase 3a floor for the
        same family). For `log_roi` RMSE, lower is better, so
        positive lift means worse performance; for AUC, higher
        is better, so positive lift means improvement.

        * **Linear** gets worse on `log_roi` RMSE by 0.011 and
          is essentially flat on AUC. Pre-registered RMSE
          direction was wrong.
        * **Gradient boosting**, the strongest baseline family on
          OOF, gets worse on `log_roi` RMSE by 0.006 and loses
          substantially on the classification targets (-0.041 on
          `roi_gt_1` AUC, -0.024 on `roi_gt_2` AUC). This is the
          single clearest signal in the table: if the lexical
          features carried genuine non-linear signal, gradient
          boosting would extract some of it. It instead extracts
          noise.
        * **K-nearest-neighbours** loses on every metric, with
          RMSE rising by 0.015 and `roi_gt_1` AUC dropping
          0.032.
        * **RBF support vector machine** appears to gain modestly
          (RMSE drops 0.006, `roi_gt_2` AUC rises 0.031), but
          starting from the worst floor in the matrix; its
          lexical-augmented OOF numbers are still well below the
          other three families' floor numbers without lexical
          features.

        We confirmed the cause by computing each lexical feature's
        Pearson correlation with each of the three targets on the
        training split: no feature exceeds an absolute value of
        0.10 with any target.

        ### Why the features behave this way

        The most likely mechanism is not that lexical features
        carry no information at all. The mechanism is that they
        carry information genre, era, and structural counts have
        already absorbed. Action films systematically have shorter
        dialogue and lower Flesch-Kincaid scores. Drama screenplays
        use more sophisticated vocabulary on average. Period pieces
        use longer words. The structural baseline already includes
        thirteen genre dummies and a release-year column, so by the
        time the lexical features are introduced they are competing
        for residual signal after the strongest confounds are
        controlled. At the corpus size and feature count of this
        ablation, with these four model families, the residual is
        too thin to extract reliably.

        This framing matters for two reasons. First, it generates
        a testable prediction: features whose signal is more
        orthogonal to genre (graph-structural features of the
        character network, sentiment-trajectory shape that does not
        track genre as cleanly) should fare better. Second, it
        motivates a methodology addition: a small pre-specified set
        of feature-group combinations evaluated jointly, after the
        Part B groups have produced their standalone numbers. A
        group that looks dead alone may carry meaningful lift in
        combination with other groups whose signal lives in
        different parts of the residual.

        ### What we do with this finding

        Three actions follow.

        * The negative-lift row is recorded honestly in the
          ablation table at `reports/tables/phase3_ablation.csv`.
          The lexical features stay computed on disk so the
          modelling phase can re-evaluate them in any model family
          its benchmark tests, and so they are available to the
          combinations evaluation described above.
        * Both frequency features (`mean_log_frequency` and
          `rare_word_proportion`) are retained despite their high
          within-pair correlation. They measure conceptually
          different things (mean log-frequency versus the
          bottom-quartile share of the frequency distribution),
          and the modelling phase may handle the redundancy
          differently than the four families used here.
        * The next Part B ablation (sentiment) proceeds. After
          all five Part B groups land, a pre-specified
          combinations evaluation runs against the same floor.
    """),

    # ============================================================
    # 10. Outputs and what comes next
    # ============================================================
    md("---\n\n## 10. Outputs and next steps"),
    md("""
        ### Files produced so far

        * `data/processed/split_assignments.parquet`. One row per
          film with columns for the IMDb identifier, the
          stratification cell, and the assigned split. The
          authoritative split definition used by every downstream
          phase.
        * `data/processed/features_lexical.parquet`. The fourteen
          lexical features, one row per film. Kept on disk for the
          modelling phase to re-evaluate.
        * `reports/tables/phase3_split_diagnostics.csv`. The full
          per-stratum split-count table.
        * `reports/tables/phase3a_baseline.csv`. Headline metrics
          for both feature configurations under all four model
          families, with both the original and the log-transformed
          parameterizations preserved. One hundred twelve rows.
        * `reports/tables/phase3_ablation.csv`. The Part B
          ablation table. Currently contains the lexical group
          (twenty-eight rows: four families x seven metric rows).
        * `reports/figures/phase3_target_distributions.png`.
          Visual diagnostics of the three prediction targets.
        * `reports/figures/phase3_log_transform_effect.png`.
          Before-and-after histograms for three representative
          structural counts.

        ### What remains in Part B

        Four feature groups still to evaluate, each with its own
        proposal, pre-registration, implementation, and
        multi-family ablation.

        1. **Sentiment features.** Aggregate sentiment over
           dialogue, plus measures of sentiment trajectory across
           the screenplay (does the emotional arc rise, fall, or
           peak in the middle). Shares NLTK preprocessing with
           lexical and is therefore inexpensive to add.
        2. **Topic features.** Latent topic distributions
           computed on the screenplay text. Fit on training data
           only and applied to the calibration and test sets.
        3. **Embedding features.** Sentence-transformer
           embeddings of dialogue and action text, pooled to film
           level. Most computationally expensive and saved for
           last.
        4. **Character network features.** Graph metrics derived
           from a character-cooccurrence graph: density, number
           of components, dominance of leading characters. We
           expect this group to add information that is
           orthogonal to genre and likely lifts ROI more than
           rating.

        Each group will produce its own ablation rows in
        `phase3_ablation.csv`, evaluated under all four model
        families for the same diagnostic disambiguation we used
        on lexical. After all groups have landed, the modelling
        phase selects the strongest feature combination, runs a
        full benchmark of candidate models with hyperparameter
        tuning, wraps the chosen model with a calibration layer,
        and connects it to the asymmetric-cost decision rule.
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
