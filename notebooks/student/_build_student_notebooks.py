"""Build the single all-in-one student notebook.

Regenerate with:

    python -m notebooks.student._build_student_notebooks

One notebook end-to-end: load data, train + tune one model,
calibrate it, run the cost-asymmetric decision rule, produce SHAP
explanations, and (optionally) evaluate on the held-out test set.
The whole pipeline lives in a single file with one CONFIG cell at
the top.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUT = Path(__file__).resolve().parent
OUTFILE = OUT / "student_pipeline.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Student Notebook — Full Pipeline (one file, one CONFIG cell)

        This single notebook runs the entire four-layer pipeline:

        1. **Train** a model on dialogue features
        2. **Calibrate** its probability outputs
        3. **Decide** Greenlight / Pass / Refer using a cost matrix
        4. **Explain** with SHAP
        5. **Evaluate** on the held-out test set (optional toggle)

        Edit the CONFIG cell once, run all cells in order, copy the
        headline numbers into the team comparison table.

        ## What to run first

        1. From the project root: ``pip install -r requirements.txt``
        2. Confirm ``data/processed/features.parquet`` exists (it
           does in this checkout).
        3. Open this notebook in Jupyter or VS Code and run all cells.

        ## Team workflow

        Each teammate edits the CONFIG cell to set:

        - a unique ``run_name`` (so artifacts don't clobber each other)
        - their chosen ``model_family`` (test logistic vs random_forest
          vs xgboost vs svm_rbf — see the bake-off table at the bottom)
        - any of the three improvement toggles
          (``use_grid_search`` / ``use_repeated_cv`` / ``use_class_balance``)

        Then everyone runs the notebook top-to-bottom and shares results.

        ## The test-set rule

        ``EVALUATE_ON_TEST`` at the bottom of CONFIG defaults to
        ``False``. **Leave it False** while iterating on model choice.
        Only flip to ``True`` once for the final, frozen run that
        goes in the report.
    """),

    # ============================================================
    # CONFIG
    # ============================================================
    md("""
        ## CONFIG — edit me

        Every knob the team will play with lives in this one cell.
        Change values, re-run all cells.
    """),
    code("""
        CONFIG = {
            # YOUR run_name — unique label, e.g. "alice_xgboost".
            # Each run lands in data/processed/student/<run_name>/.
            "run_name": "team_baseline_rf",

            # Target. Headline = roi_gt_2 (was the film net-profitable?).
            "target":       "roi_gt_2",            # "roi_gt_2" | "roi_gt_1"
            "model_family": "random_forest",       # "logistic" | "random_forest" | "xgboost" | "svm_rbf"

            # ---- Feature selection. Three modes: ----
            #
            # MODE A - named group (string):
            #   "feature_set": "all"          all 127 features
            #   "feature_set": "structural"   structural baseline only
            #   "feature_set": "topic"        22 LDA topic proportions
            #   "feature_set": "embedding"    32 mpnet PCA components
            #   "feature_set": "network"      character-network features
            #
            # MODE B - mix groups + specific features (list; each
            # string is either a group name or an exact column name):
            #   "feature_set": ["structural", "topic"]
            #   "feature_set": ["structural", "release_year_parsed"]
            #
            # MODE C - cherry-pick exact features:
            #   "feature_set": ["log_n_scenes", "topic_08_proportion",
            #                    "release_year_parsed", "embed_pc_11"]
            #
            # Run the "Discover available features" cell after data load
            # to see all 127 column names you can pick from.
            "feature_set":     "all",
            "feature_exclude": [],                 # optional: column names to DROP

            # ---- Improvement toggles ----
            # All three on by default. Flip to False to ablate.
            "use_grid_search":   True,             # tune hyperparameters via GridSearchCV
            "use_repeated_cv":   True,             # 5 folds x 3 repeats (15 fold values)
            "use_class_balance": True,             # class_weight='balanced' / scale_pos_weight

            # ---- Calibration ----
            "calibration_method": "isotonic",      # "sigmoid" | "isotonic"
            "prob_clip_max": 0.95,                 # cap calibrated probabilities (None = off)
                                                    # Improvement #5: prevents overconfident
                                                    # 1.0 commits like the *Something Wicked*
                                                    # test-set Greenlight flop.

            # ---- Cost matrix (Greenlight / Pass / Refer) ----
            "flop_cost":   50_000_000,             # $50M  - greenlight a flop
            "miss_cost":  100_000_000,             # $100M - pass on a hit
            "refer_cost":      5_000,              # $5K   - human reader pass

            # ---- Deployment guards (improvement #4) ----
            # Even if the cost rule says "Greenlight", the system
            # downgrades to Refer when these guards trigger. Stops
            # commits in regions where the model is unreliable.
            "guard_min_year": 1990,                # no Greenlight for films older than this (None = off)
            "guard_min_genre_train_n": 30,         # no Greenlight if the film's genre cell has <N training films
            "guard_min_budget_train_n": 30,        # no Greenlight if its budget tier has <N training films

            # ---- Triage / ranking metrics (improvement #9) ----
            "top_k_for_ranking": [10, 20, 50],     # Precision@K / Recall@K at these K values

            # ---- SHAP ----
            "shap_top_k_global":   20,             # top features in the global plot
            "shap_top_k_per_film": 5,              # top contributors per film
            "shap_example_imdb_id": "auto",        # "auto" picks highest-prob film

            # ---- THE TEST-SET TOGGLE ----
            # Leave False while iterating. Flip to True ONCE for the
            # final report number; you only get to run on test once.
            "EVALUATE_ON_TEST": False,

            # CV / random seed
            "n_splits":   5,
            "n_repeats":  3,
            "random_seed": 42,
        }
    """),

    # ============================================================
    # Setup: paths and imports
    # ============================================================
    md("""
        ## Setup — paths and imports

        Path-finder walks up the filesystem to locate the project
        root by looking for ``docs/PROJECT_CONTEXT.md``. Works whether
        you launch Jupyter from the project root or from the
        ``notebooks/student/`` directory.
    """),
    code("""
        from pathlib import Path

        def _find_project_root() -> Path:
            p = Path.cwd().resolve()
            for cand in [p, *p.parents]:
                if (cand / "docs" / "PROJECT_CONTEXT.md").is_file():
                    return cand
            raise RuntimeError(f"Could not locate project root from {Path.cwd()!s}")

        PROJECT_ROOT = _find_project_root()
        DATA = PROJECT_ROOT / "data" / "processed"
        STUDENT = DATA / "student" / CONFIG["run_name"]
        STUDENT.mkdir(parents=True, exist_ok=True)
        print(f"Project root:  {PROJECT_ROOT}")
        print(f"Run name:      {CONFIG['run_name']!r}")
        print(f"Artifacts go:  {STUDENT.relative_to(PROJECT_ROOT)}")
    """),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.frozen import FrozenEstimator
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score, brier_score_loss, f1_score,
            log_loss, roc_auc_score, roc_curve,
        )
        from sklearn.model_selection import (
            GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        try:
            from xgboost import XGBClassifier
            HAS_XGB = True
        except ImportError:
            HAS_XGB = False

        try:
            import shap
            HAS_SHAP = True
        except ImportError:
            HAS_SHAP = False
    """),

    # ============================================================
    # 1. Load data
    # ============================================================
    md("""
        # 1. Load data

        We use the prebuilt 127-feature matrix from Phase 3 plus the
        train/cal/test split assignment. The test split is held out
        until the final section.
    """),
    code("""
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        master = pd.read_parquet(DATA / "films_joined.parquet")
        print(f"Total films: {len(feat)} | split breakdown: {feat['split'].value_counts().to_dict()}")

        # Train + cal pool (everything except the test split).
        df = feat[feat["split"].isin(["train", "cal"])].reset_index(drop=True)
        non_feat = {"imdb_id", "split", "log_roi", "roi_gt_1", "roi_gt_2"}
        all_cols = [c for c in df.columns if c not in non_feat]

        # Named groups. Add or rearrange these if your team wants
        # different bundles.
        groups = {
            "all":        all_cols,
            "structural": [c for c in all_cols if c.startswith(("log_", "n_", "dialogue_to_", "release_year", "log_runtime", "genre_"))],
            "topic":      [c for c in all_cols if c.startswith("topic_")],
            "embedding":  [c for c in all_cols if c.startswith("embed_pc_")],
            "network":    [c for c in all_cols if c.startswith("network_")],
            "lexical":    [c for c in all_cols if "mtld" in c or "flesch" in c or "type_token" in c or c.startswith("lex_")],
            "sentiment":  [c for c in all_cols if c.startswith(("sentiment_", "vader_", "nrc_", "archetype_"))],
        }
        target_col = CONFIG["target"]
    """),

    md("""
        ### Discover available features

        Run this cell to see what's in the matrix. Copy/paste any
        column name into ``CONFIG["feature_set"]`` to cherry-pick.
    """),
    code("""
        print("Named groups (use any of these as a string, or in a list):")
        for g, cols in groups.items():
            sample = ", ".join(cols[:3])
            print(f"  {g:11s}: {len(cols):>3d} features  (e.g. {sample}...)")

        print(f"\\nTotal feature columns available: {len(all_cols)}")
        print("\\nFull list of feature column names (copy any to feature_set):")
        for i in range(0, len(all_cols), 4):
            print("  " + "  ".join(f"{c:30s}" for c in all_cols[i:i+4]))
    """),

    md("""
        ### Resolve `feature_set` to a concrete column list

        Handles all three modes (named group, mixed groups+features,
        explicit feature list) plus the optional `feature_exclude`.
    """),
    code("""
        def resolve_features(spec, exclude, groups, all_cols):
            \"\"\"Turn the CONFIG['feature_set'] knob into a list of column names.\"\"\"
            cols = []
            if isinstance(spec, str):
                if spec not in groups:
                    raise ValueError(f"Unknown feature group {spec!r}; known: {list(groups)}")
                cols = list(groups[spec])
            elif isinstance(spec, (list, tuple)):
                for item in spec:
                    if item in groups:
                        cols.extend(groups[item])
                    elif item in all_cols:
                        cols.append(item)
                    else:
                        raise ValueError(
                            f"{item!r} is not a known group {list(groups)} "
                            f"nor a known feature column."
                        )
            else:
                raise TypeError(f"feature_set must be str or list, got {type(spec).__name__}")

            # Dedupe while preserving order.
            seen = set()
            cols = [c for c in cols if not (c in seen or seen.add(c))]

            # Drop excludes.
            if exclude:
                exclude_set = set(exclude)
                cols = [c for c in cols if c not in exclude_set]
            return cols

        feat_cols = resolve_features(
            CONFIG["feature_set"], CONFIG.get("feature_exclude", []), groups, all_cols,
        )
        print(f"feature_set spec : {CONFIG['feature_set']!r}")
        if CONFIG.get("feature_exclude"):
            print(f"feature_exclude  : {CONFIG['feature_exclude']!r}")
        print(f"Resolved to {len(feat_cols)} columns:")
        if len(feat_cols) <= 30:
            for c in feat_cols:
                print(f"  {c}")
        else:
            for c in feat_cols[:10]:
                print(f"  {c}")
            print(f"  ... and {len(feat_cols) - 10} more")

        # Build X / y.
        X = df[feat_cols].fillna(0).values
        y = df[target_col].astype(int).values

        train_mask = (df["split"] == "train").values
        cal_mask   = (df["split"] == "cal").values
        X_train, y_train = X[train_mask], y[train_mask]
        X_cal,   y_cal   = X[cal_mask],   y[cal_mask]
        ids_cal = df.loc[cal_mask, "imdb_id"].astype(str).tolist()

        print(f"\\nTrain: {len(X_train)} films, positive rate {y_train.mean():.3f}")
        print(f"Cal:   {len(X_cal)} films, positive rate {y_cal.mean():.3f}")
    """),

    # ============================================================
    # 1.5. Baseline ladder (improvement #1)
    # ============================================================
    md("""
        # 1.5. Baseline ladder (improvement #1)

        Before training the screenplay model, establish how much AUC
        you can get from **context-only baselines** (genre, decade,
        budget tier). If the screenplay model can't clearly beat
        ``genre + decade + budget tier``, the script features are
        not adding generalizable value.

        Each baseline is a per-cell training-set hit rate looked up
        for each cal film. Cells with fewer than 10 training films
        fall back to the corpus base rate (smoothing).
    """),
    code("""
        # release_year_parsed is already in features.parquet (it's a
        # model feature). genre and budget are NOT in features.parquet
        # (genre is one-hot-encoded into genre_* dummies; budget is
        # leak-prone for prediction). We pull both from master.
        meta_cols = ["imdb_id", "primary_genre_bucketed", "budget"]
        df_with_meta = df.merge(master[meta_cols], on="imdb_id", how="left")

        def assign_decade_bucket(year):
            try: y = int(year)
            except (TypeError, ValueError): return "unknown"
            if y < 1980: return "pre_1980"
            if y < 1990: return "1980s"
            if y < 2000: return "1990s"
            if y < 2010: return "2000s"
            if y < 2020: return "2010s"
            return "2020s"

        def assign_budget_tier(budget):
            if pd.isna(budget) or budget <= 0: return "unknown"
            b = float(budget)
            if b < 10_000_000:  return "under_10M"
            if b < 50_000_000:  return "10M_50M"
            if b < 150_000_000: return "50M_150M"
            return "over_150M"

        df_with_meta["decade_bucket"] = df_with_meta["release_year_parsed"].apply(assign_decade_bucket)
        df_with_meta["budget_tier"]   = df_with_meta["budget"].apply(assign_budget_tier)

        df_train_meta = df_with_meta[train_mask].reset_index(drop=True)
        df_cal_meta   = df_with_meta[cal_mask].reset_index(drop=True)
        print(f"Train: {len(df_train_meta)} films | Cal: {len(df_cal_meta)} films")
        print(f"Train genres: {df_train_meta['primary_genre_bucketed'].nunique()} | "
              f"decades: {df_train_meta['decade_bucket'].nunique()} | "
              f"budget tiers: {df_train_meta['budget_tier'].nunique()}")
    """),
    code("""
        def compute_prior(df_train, df_eval, by_cols, target_col, min_cell_n=10):
            \"\"\"Predict per-eval-film P(target=1) using per-cell training hit rates.

            Cells with fewer than ``min_cell_n`` training films fall
            back to the overall training mean.
            \"\"\"
            overall = float(df_train[target_col].mean())
            if isinstance(by_cols, str):
                by_cols = [by_cols]
            grouped = df_train.groupby(by_cols)[target_col].agg(["mean", "count"])
            rate_dict = {}
            for idx, row in grouped.iterrows():
                key = idx if isinstance(idx, tuple) else (idx,)
                if row["count"] >= min_cell_n:
                    rate_dict[key] = row["mean"]
            keys = list(zip(*[df_eval[c].astype(str).values for c in by_cols]))
            return np.array([rate_dict.get(k, overall) for k in keys])

        # Build the baseline ladder.
        BASELINES = [
            ("Majority class (corpus mean)",     []),
            ("Genre prior",                       ["primary_genre_bucketed"]),
            ("Genre + decade",                    ["primary_genre_bucketed", "decade_bucket"]),
            ("Genre + decade + budget tier",      ["primary_genre_bucketed", "decade_bucket", "budget_tier"]),
        ]

        baseline_rows = []
        for name, by in BASELINES:
            if not by:
                # Majority-class baseline: predict overall train mean for every film.
                proba_b = np.full(len(df_cal_meta), df_train_meta[target_col].mean())
            else:
                proba_b = compute_prior(df_train_meta, df_cal_meta, by, target_col)
            try:
                auc_b = roc_auc_score(y_cal, proba_b) if len(np.unique(y_cal)) > 1 and len(np.unique(proba_b)) > 1 else float("nan")
            except ValueError:
                auc_b = float("nan")
            baseline_rows.append({"baseline": name, "n_features": len(by), "cal_AUC": auc_b})

        baseline_ladder = pd.DataFrame(baseline_rows)
        print(baseline_ladder.round(3).to_string(index=False))
        print("\\nThis is the BAR your screenplay model has to clear. If your")
        print("CV AUC isn't materially above the 'Genre + decade + budget tier'")
        print("row, the script features are not adding generalizable value.")
    """),

    # ============================================================
    # 2. Build + tune model
    # ============================================================
    md("""
        # 2. Train and tune the model (Layer 1)

        Three improvement toggles bundled here:

        1. **`use_class_balance`** — `class_weight='balanced'` or
           XGBoost's `scale_pos_weight = n_neg / n_pos`. Compensates
           the 60/40 imbalance on roi_gt_2.
        2. **`use_repeated_cv`** — 5 folds × 3 repeats = 15 fold
           values per CV cell, instead of 5. Tighter confidence
           intervals.
        3. **`use_grid_search`** — small per-family grid
           (3-18 candidates). Often the biggest single lift.
    """),
    code("""
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Class balance: {n_neg} negative, {n_pos} positive  ->  scale_pos_weight = {scale_pos_weight:.3f}")

        def build_model(family, seed=42, balance=False):
            cw = "balanced" if balance else None
            if family == "logistic":
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw, random_state=seed)
            elif family == "random_forest":
                clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                              min_samples_leaf=2, class_weight=cw,
                                              random_state=seed, n_jobs=-1)
            elif family == "xgboost":
                if not HAS_XGB:
                    raise RuntimeError("xgboost not installed. pip install xgboost")
                spw = scale_pos_weight if balance else 1.0
                clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=spw, eval_metric="logloss",
                                    random_state=seed, n_jobs=-1)
            elif family == "svm_rbf":
                clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                          class_weight=cw, random_state=seed)
            else:
                raise ValueError(f"Unknown model_family: {family!r}")
            return Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", clf)])

        base_model = build_model(CONFIG["model_family"], seed=CONFIG["random_seed"],
                                  balance=CONFIG["use_class_balance"])
        print(f"Built {CONFIG['model_family']!r} | class_balance={CONFIG['use_class_balance']}")
    """),
    code("""
        # CV scheme.
        if CONFIG["use_repeated_cv"]:
            cv = RepeatedStratifiedKFold(n_splits=CONFIG["n_splits"],
                                          n_repeats=CONFIG["n_repeats"],
                                          random_state=CONFIG["random_seed"])
            cv_label = f"{CONFIG['n_splits']} folds x {CONFIG['n_repeats']} repeats = {CONFIG['n_splits']*CONFIG['n_repeats']} fold values"
        else:
            cv = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True,
                                  random_state=CONFIG["random_seed"])
            cv_label = f"{CONFIG['n_splits']} folds"
        print(f"CV: {cv_label}")

        # Tune (or just CV).
        GRIDS = {
            "logistic":      {"model__C": [0.1, 1.0, 10.0]},
            "random_forest": {"model__n_estimators": [200, 500],
                              "model__max_depth":    [None, 10, 20],
                              "model__min_samples_leaf": [1, 5, 10]},
            "xgboost":       {"model__max_depth":     [3, 4, 6],
                              "model__learning_rate": [0.05, 0.1],
                              "model__n_estimators":  [200, 500]},
            "svm_rbf":       {"model__C":     [0.5, 1.0, 3.0],
                              "model__gamma": ["scale", 0.01, 0.1]},
        }

        if CONFIG["use_grid_search"]:
            grid = GRIDS[CONFIG["model_family"]]
            n_combos = int(np.prod([len(v) for v in grid.values()]))
            print(f"Searching {n_combos} hyperparameter combinations...")
            search = GridSearchCV(base_model, grid, cv=cv, scoring="roc_auc",
                                   n_jobs=-1, refit=True)
            search.fit(X, y)
            cv_auc_mean = float(search.best_score_)
            cv_auc_std  = float(search.cv_results_["std_test_score"][search.best_index_])
            best_params = search.best_params_
            tuned_model = search.best_estimator_
            print(f"\\nBest params: {best_params}")
            print(f"Best CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")
        else:
            scores = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_auc_mean = float(scores.mean())
            cv_auc_std  = float(scores.std())
            best_params = "(no grid search; defaults)"
            tuned_model = clone(base_model).fit(X, y)
            print(f"CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")
    """),
    md("""
        ### Held-out evaluation on the cal split

        ``GridSearchCV(refit=True)`` already fits the best model on
        train + cal, so the cal AUC there is in-sample. To get an
        honest held-out figure, we re-fit a clone on **train only**
        and score the **cal split**.
    """),
    code("""
        held_out_model = clone(tuned_model).fit(X_train, y_train)
        cal_proba_uncal = held_out_model.predict_proba(X_cal)[:, 1]
        cal_auc = roc_auc_score(y_cal, cal_proba_uncal)
        print(f"Held-out cal AUC: {cal_auc:.3f}")
    """),
    md("### ROC curve on the cal set"),
    code("""
        fpr, tpr, _ = roc_curve(y_cal, cal_proba_uncal)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"{CONFIG['model_family']} (AUC = {cal_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"Layer 1 ROC - {CONFIG['target']} on cal set")
        plt.legend(); plt.grid(alpha=0.3); plt.show()
    """),

    # ============================================================
    # 3. Calibration
    # ============================================================
    md("""
        # 3. Calibrate the probability outputs (Layer 2)

        A model's `predict_proba` doesn't always mean what it
        literally says. ``CalibratedClassifierCV`` re-fits a small
        post-hoc transformer (sigmoid or isotonic) so the predicted
        0.7 actually corresponds to a ~70% empirical positive rate.

        We measure ECE (Expected Calibration Error) before vs after.
        Lower is better; 0 = perfect.
    """),
    code("""
        def ece(y_true, y_prob, n_bins=10):
            edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.maximum.accumulate(edges)
            bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
            tot = 0.0
            for b in range(n_bins):
                m = bin_idx == b
                if m.any():
                    tot += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
            return float(tot)

        ece_uncal   = ece(y_cal, cal_proba_uncal)
        brier_uncal = brier_score_loss(y_cal, cal_proba_uncal)
        print(f"BEFORE calibration: ECE = {ece_uncal:.4f}, Brier = {brier_uncal:.4f}")

        calibrator = CalibratedClassifierCV(
            FrozenEstimator(held_out_model),
            method=CONFIG["calibration_method"], cv=5,
        )
        calibrator.fit(X_cal, y_cal)
        cal_proba_raw = calibrator.predict_proba(X_cal)[:, 1]

        # ---- Improvement #5: probability clipping ----
        # Cap calibrated probabilities so the system never sees a
        # 1.0. Stops the *Something Wicked* failure pattern
        # (probability=1.0 -> Greenlight -> $50M loss on a flop).
        if CONFIG.get("prob_clip_max") is not None:
            cap = float(CONFIG["prob_clip_max"])
            cal_proba = np.minimum(cal_proba_raw, cap)
            n_clipped = int((cal_proba_raw > cap).sum())
            print(f"Probability clipping at {cap}: {n_clipped} of {len(cal_proba_raw)} predictions clipped down")
        else:
            cal_proba = cal_proba_raw

        ece_cal   = ece(y_cal, cal_proba)
        brier_cal = brier_score_loss(y_cal, cal_proba)
        print(f"AFTER  calibration: ECE = {ece_cal:.4f}, Brier = {brier_cal:.4f}")
    """),
    md("### Reliability diagram"),
    code("""
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, p, color in [("uncalibrated", cal_proba_uncal, "#d7301f"),
                                 (CONFIG["calibration_method"], cal_proba, "#2c7fb8")]:
            mp, fp = calibration_curve(y_cal, p, n_bins=10, strategy="quantile")
            ax.plot(mp, fp, marker="o", linewidth=2, color=color, label=label)
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        ax.set_xlabel("Predicted probability (mean per bin)")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title(f"Layer 2 reliability - {CONFIG['model_family']} on {CONFIG['target']}")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.show()
    """),

    # ============================================================
    # 4. Cost-asymmetric decision
    # ============================================================
    md("""
        # 4. Apply the cost matrix (Layer 3)

        For each calibrated probability ``p``, expected cost of each
        action:

        - Greenlight: ``(1 - p) * flop_cost``
        - Pass:       ``p * miss_cost``
        - Refer:      ``refer_cost`` (independent of p)

        Pick the lowest-expected-cost action; tie-break to Refer.
    """),
    code("""
        def decide(p, fc, mc, rc):
            costs = {"Greenlight": (1 - p) * fc, "Pass": p * mc, "Refer": rc}
            mn = min(costs.values())
            return ("Refer" if costs["Refer"] == mn else min(costs, key=costs.get)), costs

        # ---- Improvement #4: deployment guards ----
        # Even if the cost rule says Greenlight, downgrade to Refer
        # when the film is in a region the model is not reliable on.
        train_genres_n = df_train_meta["primary_genre_bucketed"].value_counts().to_dict()
        train_budget_n = df_train_meta["budget_tier"].value_counts().to_dict()

        def apply_guards(action, year, genre, budget_tier):
            \"\"\"Return possibly-downgraded action plus reason if guarded.\"\"\"
            if action != "Greenlight":
                return action, None
            min_year = CONFIG.get("guard_min_year")
            if min_year is not None and not pd.isna(year) and int(year) < int(min_year):
                return "Refer", f"guard:pre_{min_year}"
            min_g_n = CONFIG.get("guard_min_genre_train_n", 0)
            if train_genres_n.get(str(genre), 0) < int(min_g_n):
                return "Refer", f"guard:thin_genre({genre})"
            min_b_n = CONFIG.get("guard_min_budget_train_n", 0)
            if train_budget_n.get(str(budget_tier), 0) < int(min_b_n):
                return "Refer", f"guard:thin_budget({budget_tier})"
            return action, None

        # Compute raw cost-rule actions, then apply guards.
        raw_actions, all_costs = zip(*[decide(p, CONFIG["flop_cost"],
                                              CONFIG["miss_cost"], CONFIG["refer_cost"])
                                        for p in cal_proba])
        guard_reasons = []
        actions_list = []
        for raw_a, year, genre, btier in zip(
            raw_actions,
            df_cal_meta["release_year_parsed"].values,
            df_cal_meta["primary_genre_bucketed"].astype(str).values,
            df_cal_meta["budget_tier"].astype(str).values,
        ):
            a, reason = apply_guards(raw_a, year, genre, btier)
            actions_list.append(a)
            guard_reasons.append(reason)
        actions = np.array(actions_list)

        # Diagnostic.
        n_raw_gl = sum(1 for a in raw_actions if a == "Greenlight")
        n_final_gl = int((actions == "Greenlight").sum())
        n_guarded = sum(1 for r in guard_reasons if r is not None)
        print(f"Decision rule produced {n_raw_gl} raw Greenlights")
        print(f"Deployment guards downgraded {n_guarded} -> {n_final_gl} final Greenlights")
        if n_guarded:
            from collections import Counter
            reasons = Counter(r for r in guard_reasons if r is not None)
            for r, n in reasons.most_common():
                print(f"  guard fired: {r}  ({n} films)")
        print()
        for a in ["Greenlight", "Pass", "Refer"]:
            print(f"  {a:11s}: {(actions == a).mean()*100:5.1f}%  ({(actions == a).sum():>3d} films)")
    """),
    md("### System total cost vs five baselines"),
    code("""
        def realized(action, true, fc, mc, rc):
            if action == "Greenlight": return 0 if true == 1 else fc
            if action == "Pass":       return mc if true == 1 else 0
            return rc

        rng = np.random.default_rng(CONFIG["random_seed"])
        baselines = {
            "Always-Greenlight": ["Greenlight"] * len(y_cal),
            "Always-Pass":       ["Pass"] * len(y_cal),
            "Read-Everything":   ["Refer"] * len(y_cal),
            "Random":            rng.choice(["Greenlight", "Pass", "Refer"], size=len(y_cal)).tolist(),
            "System (you!)":     actions.tolist(),
        }
        rows = []
        for name, acts in baselines.items():
            tot = sum(realized(a, t, CONFIG["flop_cost"], CONFIG["miss_cost"], CONFIG["refer_cost"])
                       for a, t in zip(acts, y_cal))
            rows.append({"strategy": name, "total_cost_USD": tot,
                         "cost_per_film_M": tot / len(y_cal) / 1e6})
        bench = pd.DataFrame(rows).sort_values("total_cost_USD")
        print(bench.to_string(index=False))
    """),
    code("""
        sb = bench.sort_values("total_cost_USD", ascending=True)
        colors = ["#2c7fb8" if s == "System (you!)" else "#a6cee3" for s in sb["strategy"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sb["strategy"], sb["total_cost_USD"].clip(lower=1), color=colors)
        ax.set_xscale("log")
        ax.set_xlabel("Total realized cost on cal set (USD, log scale)")
        ax.set_title("Layer 3: system vs baselines")
        for s, v in zip(sb["strategy"], sb["total_cost_USD"]):
            ax.text(max(v, 1) * 1.2, s, f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K",
                    va="center", fontsize=9)
        plt.show()
    """),
    md("### Per-film recommendations (preview)"),
    code("""
        master_lookup = master.set_index("imdb_id")["movie_name"].to_dict()
        decisions_df = pd.DataFrame({
            "imdb_id":                ids_cal,
            "movie_name":             [master_lookup.get(i, i) for i in ids_cal],
            "calibrated_probability": cal_proba,
            "true_label":             y_cal,
            "raw_action":             list(raw_actions),
            "recommended_action":     actions,
            "guard_reason":           guard_reasons,
            "release_year":           df_cal_meta["release_year_parsed"].values,
            "genre":                  df_cal_meta["primary_genre_bucketed"].astype(str).values,
            "budget_tier":            df_cal_meta["budget_tier"].astype(str).values,
        })
        decisions_df.to_csv(STUDENT / "student_decisions.csv", index=False)
        print(f"Saved {STUDENT / 'student_decisions.csv'}: {len(decisions_df)} rows\\n")
        gl = decisions_df[decisions_df["recommended_action"] == "Greenlight"].head(8)
        if len(gl):
            print("Greenlight films (preview, first 8):")
            print(gl[["movie_name", "calibrated_probability", "true_label",
                      "release_year", "genre"]].to_string(index=False))
        else:
            print("System recommends 0 Greenlights on the cal set under this cost matrix.")
        guarded = decisions_df[decisions_df["guard_reason"].notna()]
        if len(guarded):
            print(f"\\n{len(guarded)} films had Greenlight downgraded to Refer by deployment guards:")
            print(guarded[["movie_name", "calibrated_probability", "true_label", "guard_reason"]]
                  .head(8).to_string(index=False))
    """),

    # ============================================================
    # 4.5. Triage / ranking metrics (improvement #9)
    # ============================================================
    md("""
        # 4.5. Triage / ranking metrics (improvement #9)

        The cost-asymmetric decision rule answers "should we
        commit?". But studios in practice want to know **which
        scripts deserve human review first**. That's a ranking
        problem, not a binary one.

        We report:

        - **Precision@K**: of the top-K films by calibrated
          probability, what fraction are actual hits?
        - **Recall@K**: of all hits in the cal set, what fraction
          are in the top-K?
        - **Lift@K**: hit rate in the top-K divided by the corpus
          base rate. > 1 means the system concentrates hits at
          the top.

        At a 0.51 test AUC the model is weak as a binary classifier
        but might still be useful as a triage prioritizer if the
        top decile has a noticeably higher hit rate than the rest.
    """),
    code("""
        def precision_at_k(y_true, y_score, k):
            order = np.argsort(-np.asarray(y_score, dtype=float))
            top = order[:k]
            return float(np.asarray(y_true)[top].mean()) if k else float("nan")

        def recall_at_k(y_true, y_score, k):
            order = np.argsort(-np.asarray(y_score, dtype=float))
            top = order[:k]
            n_pos = int(np.asarray(y_true).sum())
            return float(np.asarray(y_true)[top].sum() / max(n_pos, 1))

        def lift_at_k(y_true, y_score, k):
            base = float(np.asarray(y_true).mean())
            return precision_at_k(y_true, y_score, k) / max(base, 1e-9)

        rows = []
        for k in CONFIG["top_k_for_ranking"]:
            rows.append({
                "K":              k,
                "precision_at_K": precision_at_k(y_cal, cal_proba, k),
                "recall_at_K":    recall_at_k(y_cal, cal_proba, k),
                "lift_at_K":      lift_at_k(y_cal, cal_proba, k),
            })
        rank_df = pd.DataFrame(rows)
        print(f"Cal corpus base rate (P(hit)): {y_cal.mean():.3f}")
        print(rank_df.round(3).to_string(index=False))
    """),
    md("### Lift chart by decile"),
    code("""
        def lift_by_decile(y_true, y_score, n_deciles=10):
            order = np.argsort(-np.asarray(y_score, dtype=float))
            base = float(np.asarray(y_true).mean())
            n = len(y_score)
            chunk = max(n // n_deciles, 1)
            rows = []
            for d in range(n_deciles):
                idx = order[d * chunk : (d + 1) * chunk if d < n_deciles - 1 else n]
                if len(idx) == 0:
                    continue
                hr = float(np.asarray(y_true)[idx].mean())
                rows.append({
                    "decile":     d + 1,
                    "n":          len(idx),
                    "mean_pred":  float(np.asarray(y_score)[idx].mean()),
                    "hit_rate":   hr,
                    "lift":       hr / max(base, 1e-9),
                })
            return pd.DataFrame(rows)

        decile_df = lift_by_decile(y_cal, cal_proba, n_deciles=10)
        print(decile_df.round(3).to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(decile_df["decile"], decile_df["lift"], color="#2c7fb8")
        ax.axhline(1.0, color="grey", linestyle="--", label="corpus base rate")
        ax.set_xlabel("Decile (1 = top scoring)")
        ax.set_ylabel("Lift over corpus base rate")
        ax.set_title("Triage lift by decile - is the top decile enriched in hits?")
        ax.legend(); ax.grid(axis="y", alpha=0.3); plt.show()
    """),

    # ============================================================
    # 5. SHAP explanation
    # ============================================================
    md("""
        # 5. SHAP attribution (Layer 4)

        TreeSHAP for tree models (xgboost, random_forest); for
        logistic we use coefficient * feature_value. SVM-RBF
        falls back to permutation importance.

        Outputs:
        - Global feature ranking by mean |contribution|
        - Per-film breakdown for one example film
    """),
    code("""
        family = CONFIG["model_family"]

        def attribute(pipe, X, family):
            scaler = pipe.named_steps["scaler"]
            clf = pipe.named_steps["model"]
            X_scaled = scaler.transform(X)
            if family in ("xgboost", "random_forest"):
                if not HAS_SHAP:
                    raise RuntimeError("shap not installed. pip install shap")
                explainer = shap.TreeExplainer(clf)
                arr = np.asarray(explainer.shap_values(X_scaled))
                if arr.ndim == 3:
                    arr = arr[:, :, 1]
                return arr, "TreeSHAP"
            if family == "logistic":
                coefs = clf.coef_.ravel()
                return X_scaled * coefs[np.newaxis, :], "linear coefficients * x"
            if family == "svm_rbf":
                from sklearn.inspection import permutation_importance
                pi = permutation_importance(pipe, X, pipe.predict(X), n_repeats=5,
                                              random_state=42, n_jobs=-1)
                return np.tile(pi.importances_mean, (len(X), 1)), "permutation importance"
            raise ValueError(family)

        contribs, method = attribute(held_out_model, X_cal, family)
        print(f"Attribution method: {method}")
        print(f"Per-film contribution matrix shape: {contribs.shape}")
    """),
    md("### Global feature ranking"),
    code("""
        ranking = pd.DataFrame({
            "feature":     feat_cols,
            "mean_abs":    np.mean(np.abs(contribs), axis=0),
            "mean_signed": np.mean(contribs, axis=0),
        }).sort_values("mean_abs", ascending=False).reset_index(drop=True)

        K = CONFIG["shap_top_k_global"]
        print(ranking.head(15).round(4).to_string(index=False))

        top = ranking.head(K).iloc[::-1]
        colors = ["#2c7fb8" if s >= 0 else "#d7301f" for s in top["mean_signed"]]
        fig, ax = plt.subplots(figsize=(8, K * 0.3 + 1))
        ax.barh(top["feature"], top["mean_abs"], color=colors)
        ax.set_xlabel("Mean |contribution|")
        ax.set_title(f"Top-{K} feature importance ({method})")
        ax.grid(axis="x", alpha=0.3); plt.show()
        print("Blue = pushes probability UP on average; Red = pulls DOWN")
    """),
    md("### Per-film breakdown for one example"),
    code("""
        if CONFIG["shap_example_imdb_id"] == "auto":
            ex_idx = int(np.argmax(cal_proba))
            ex_id = ids_cal[ex_idx]
        else:
            ex_id = CONFIG["shap_example_imdb_id"]
            ex_idx = ids_cal.index(ex_id)
        movie = master_lookup.get(ex_id, ex_id)
        print(f"Inspecting: {movie} ({ex_id}) | predicted P = {cal_proba[ex_idx]:.3f}")

        K2 = CONFIG["shap_top_k_per_film"]
        film = pd.DataFrame({"feature": feat_cols, "contribution": contribs[ex_idx]})
        top_pos = film[film["contribution"] > 0].nlargest(K2, "contribution")
        top_neg = film[film["contribution"] < 0].nsmallest(K2, "contribution")
        show = pd.concat([top_pos, top_neg]).iloc[::-1]
        colors = ["#2c7fb8" if v >= 0 else "#d7301f" for v in show["contribution"]]
        fig, ax = plt.subplots(figsize=(8, len(show) * 0.4 + 1))
        ax.barh(show["feature"], show["contribution"], color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Per-film contribution"); ax.set_title(f"{movie}")
        ax.grid(axis="x", alpha=0.3); plt.show()
    """),

    # ============================================================
    # 6. Save the bundle
    # ============================================================
    md("""
        # 6. Save the run bundle

        Everything trained and computed above goes into one joblib
        file under your run_name's folder. Useful for the team
        comparison meeting.
    """),
    code("""
        bundle = {
            "config":          CONFIG,
            "feature_columns": feat_cols,
            "tuned_model":     tuned_model,
            "held_out_model":  held_out_model,
            "calibrator":      calibrator,
            "cv_auc_mean":     cv_auc_mean,
            "cv_auc_std":      cv_auc_std,
            "cal_auc":         float(cal_auc),
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated":   ece_cal,
            "best_params":     str(best_params),
            "decisions_df":    decisions_df,
            "shap_ranking":    ranking,
        }
        joblib.dump(bundle, STUDENT / "student_full_pipeline.joblib")
        print(f"Saved {STUDENT / 'student_full_pipeline.joblib'}")
    """),

    # ============================================================
    # 7. Test set (only if EVALUATE_ON_TEST is True)
    # ============================================================
    md("""
        # 7. Test-set evaluation (gated)

        **Only runs when ``CONFIG["EVALUATE_ON_TEST"]`` is True.**
        Leave it False while you iterate on model choice. Flip to
        True ONCE for the final number that goes in the report.

        The test set is touched once across the whole project.
    """),
    code("""
        if not CONFIG["EVALUATE_ON_TEST"]:
            print("EVALUATE_ON_TEST=False -> skipping test-set evaluation.")
            print("Flip CONFIG['EVALUATE_ON_TEST']=True for the final report run.")
        else:
            df_test = feat[feat["split"] == "test"].reset_index(drop=True)
            X_test = df_test[feat_cols].fillna(0).values
            y_test = df_test[target_col].astype(int).values
            ids_test = df_test["imdb_id"].astype(str).tolist()
            print(f"Test set: {len(df_test)} films, positive rate {y_test.mean():.3f}")

            # Calibrated probabilities on test, with the same clipping
            # as cal so the deployed system is consistent.
            test_proba_raw = calibrator.predict_proba(X_test)[:, 1]
            if CONFIG.get("prob_clip_max") is not None:
                test_proba = np.minimum(test_proba_raw, float(CONFIG["prob_clip_max"]))
            else:
                test_proba = test_proba_raw

            # Test metadata (genre, budget tier) for guards. release_year
            # is already in features.parquet on df_test.
            df_test_meta = df_test.merge(master[meta_cols], on="imdb_id", how="left")
            df_test_meta["decade_bucket"] = df_test_meta["release_year_parsed"].apply(assign_decade_bucket)
            df_test_meta["budget_tier"]   = df_test_meta["budget"].apply(assign_budget_tier)

            # Layer 1 metrics with bootstrap CIs.
            rng2 = np.random.default_rng(CONFIG["random_seed"])
            def boot_ci(metric, *args, n=2000):
                arrs = [np.asarray(a) for a in args]
                pt = float(metric(*arrs))
                samples = []
                for _ in range(n):
                    idx = rng2.integers(0, len(arrs[0]), size=len(arrs[0]))
                    try:    samples.append(float(metric(*[a[idx] for a in arrs])))
                    except Exception: pass
                lo, hi = np.quantile(samples, [0.025, 0.975])
                return pt, lo, hi
            print("\\n--- Layer 1: predictive performance on test ---")
            for name, fn in [
                ("AUC-ROC",  lambda y, p: roc_auc_score(y, p)),
                ("PR-AUC",   lambda y, p: average_precision_score(y, p)),
                ("F1@0.5",   lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
                ("log-loss", lambda y, p: log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
                ("Brier",    lambda y, p: brier_score_loss(y, p)),
            ]:
                pt, lo, hi = boot_ci(fn, y_test, test_proba)
                print(f"  {name:9s}: {pt:.3f} [{lo:.3f}, {hi:.3f}]")

            # Layer 2 calibration on test.
            print("\\n--- Layer 2: calibration on test ---")
            print(f"  ECE on test:   {ece(y_test, test_proba):.4f}")
            print(f"  Brier on test: {brier_score_loss(y_test, test_proba):.4f}")

            # Layer 3: decision rule + deployment guards on test.
            raw_test_actions = [decide(p, CONFIG["flop_cost"], CONFIG["miss_cost"],
                                        CONFIG["refer_cost"])[0] for p in test_proba]
            test_actions = []
            test_guard_reasons = []
            for raw_a, year, genre, btier in zip(
                raw_test_actions,
                df_test_meta["release_year_parsed"].values,
                df_test_meta["primary_genre_bucketed"].astype(str).values,
                df_test_meta["budget_tier"].astype(str).values,
            ):
                a, reason = apply_guards(raw_a, year, genre, btier)
                test_actions.append(a)
                test_guard_reasons.append(reason)

            n_raw_gl_t = sum(1 for a in raw_test_actions if a == "Greenlight")
            n_final_gl_t = sum(1 for a in test_actions    if a == "Greenlight")
            n_guarded_t = sum(1 for r in test_guard_reasons if r is not None)
            print(f"\\n--- Layer 3: decisions on test ---")
            print(f"Raw cost-rule Greenlights: {n_raw_gl_t} | guards downgraded {n_guarded_t} | final Greenlights: {n_final_gl_t}")
            for a in ["Greenlight", "Pass", "Refer"]:
                print(f"  {a:11s}: {(np.array(test_actions) == a).mean()*100:5.1f}%")

            test_baselines = {
                "Always-Greenlight": ["Greenlight"] * len(y_test),
                "Always-Pass":       ["Pass"] * len(y_test),
                "Read-Everything":   ["Refer"] * len(y_test),
                "System (you!)":     test_actions,
            }
            test_rows = []
            for nm, acts in test_baselines.items():
                t = sum(realized(a, t_, CONFIG["flop_cost"], CONFIG["miss_cost"],
                                  CONFIG["refer_cost"]) for a, t_ in zip(acts, y_test))
                test_rows.append({"strategy": nm, "total_cost_M": t / 1e6})
            print("\\n", pd.DataFrame(test_rows).sort_values("total_cost_M").to_string(index=False))

            # Layer 4: ranking metrics on test.
            print("\\n--- Triage / ranking on test ---")
            for k in CONFIG["top_k_for_ranking"]:
                print(f"  K={k:>3d}  P@K={precision_at_k(y_test, test_proba, k):.3f}  "
                      f"R@K={recall_at_k(y_test, test_proba, k):.3f}  "
                      f"lift={lift_at_k(y_test, test_proba, k):.2f}")

            # Save test predictions with guard info.
            test_report = pd.DataFrame({
                "imdb_id":                ids_test,
                "movie_name":             [master_lookup.get(i, i) for i in ids_test],
                "calibrated_probability": test_proba,
                "true_label":             y_test,
                "raw_action":             raw_test_actions,
                "recommended_action":     test_actions,
                "guard_reason":           test_guard_reasons,
                "release_year":           df_test_meta["release_year_parsed"].values,
                "genre":                  df_test_meta["primary_genre_bucketed"].astype(str).values,
                "budget_tier":            df_test_meta["budget_tier"].astype(str).values,
            })
            test_report.to_csv(STUDENT / "student_test_predictions.csv", index=False)
            print(f"\\nSaved {STUDENT / 'student_test_predictions.csv'}")
    """),

    # ============================================================
    # 8. Compare to teammates
    # ============================================================
    md("""
        # 8. Compare to your teammates

        Copy your headline numbers into a shared table:

        | Person | run_name | model_family | Tuned CV AUC | Cal AUC | ECE after | System cost (cal) |
        |---|---|---|---|---|---|---|
        | Alice | alice_xgboost | xgboost | … | … | … | … |
        | Bob | bob_random_forest | random_forest | 0.612 | 0.614 | 0.0035 | … |

        ### Reference numbers (default RF + all 3 toggles ON)

        | Stage | Value |
        |---|---|
        | Tuned CV AUC | 0.612 ± 0.022 |
        | Held-out cal AUC | 0.614 |
        | ECE before / after calibration | 0.344 / 0.0035 |
        | Best RF params | max_depth=10, min_samples_leaf=10, n_estimators=500 |

        ### Bare baseline numbers (all 3 toggles OFF) for comparison

        | model_family | CV AUC | Cal AUC |
        |---|---|---|
        | random_forest | 0.595 | 0.612 |
        | xgboost       | 0.597 | 0.569 |
        | logistic      | 0.576 | 0.573 |
        | svm_rbf       | 0.571 | 0.530 |

        ### Rigorous Phase 4 reference (curated 92 features, full search)

        | model_family | OOF AUC |
        |---|---|
        | xgboost | 0.652 |
        | svm_rbf | 0.635 |

        ### Ablating the three toggles

        Run with all three on (default), then flip one at a time to
        False and see what each contributes:

        - Off ``use_grid_search``: cal AUC drops ~0.01-0.02 for tree
          models. The single biggest knob.
        - Off ``use_repeated_cv``: headline mean barely moves; only
          the CI gets wider.
        - Off ``use_class_balance``: cal AUC drops ~0.005. Small but
          free.

        That's it. One notebook, one CONFIG, end-to-end.
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
    OUTFILE.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {OUTFILE.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
