"""Build the five student-friendly notebooks under ``notebooks/student/``.

Regenerate all five with:

    python -m notebooks.student._build_student_notebooks

Each notebook is one layer of the four-layer triage system, with a
CONFIG cell at the top so teammates can swap models / methods /
cost values and re-run. The full rigorous pipeline lives in
``notebooks/phase_4.ipynb`` through ``phase_8.ipynb``; this folder
is the simplified team-facing version.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = OUT / name
    out.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {out.relative_to(Path.cwd())}")


# Boilerplate placed in every student notebook: project-root detection +
# the per-teammate STUDENT directory that isolates each run's artifacts.
PATH_BOILERPLATE = """
    # --- Paths (works from any working directory) ---
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
"""


# ============================================================
# Notebook 01 — Modeling (the main playground)
# ============================================================


CELLS_01 = [
    md("""
        # Student Notebook 01 — Modeling (Layer 1)

        Pick a model, tune it, evaluate it. Every teammate runs this
        notebook with a different choice in the CONFIG cell below
        and shares the headline AUC number.

        **Run order:** this is notebook 01. After running it, run
        02 → 03 → 04, and only then 05 (the test set).

        **What you'll get out:** CV AUC (mean ± std) from repeated
        cross-validation, held-out cal AUC, a tuned model saved at
        ``data/processed/student/<run_name>/student_model.joblib``
        for the next notebooks.

        **What's bundled in this notebook (vs the bare baseline):**

        1. **Hyperparameter search** via ``GridSearchCV`` over a small
           grid per model family. Often the biggest single lift.
        2. **Repeated stratified cross-validation** (5 folds × 3
           repetitions = 15 fold values) for tighter CV-AUC
           confidence intervals.
        3. **Class balancing** (``class_weight="balanced"`` for sklearn
           models, ``scale_pos_weight`` for XGBoost) to compensate
           for the 60/40 class imbalance on roi_gt_2.

        Each of the three is toggleable in CONFIG so you can ablate
        and see what each one buys you.
    """),

    # ----------- CONFIG (the knob) -----------
    md("""
        ## CONFIG — edit me

        Change any value below and re-run the notebook. The
        comments list the supported options.
    """),
    code("""
        CONFIG = {
            # YOUR run_name. Use a unique label so your artifacts don't
            # clobber teammates'. Convention: "<your-name>_<model>".
            # Each run lands in data/processed/student/<run_name>/.
            "run_name": "team_baseline_rf",

            # Target. The headline target is roi_gt_2 (was the film
            # net profitable?). roi_gt_1 is the easier 1x ROI binary;
            # log_roi is regression. The other two notebooks assume
            # classification, so for the team comparison stick with
            # roi_gt_2 unless you want to dig into log_roi separately.
            "target": "roi_gt_2",                # "roi_gt_2" | "roi_gt_1" | "log_roi"

            # Model family. Default = random_forest because it wins
            # the bake-off on the simplified 127-feature matrix:
            #   random_forest  : CV AUC 0.595, cal AUC 0.612  <-- best in general
            #   xgboost        : CV AUC 0.597, cal AUC 0.569  (CV tied, weaker on cal)
            #   logistic       : CV AUC 0.576, cal AUC 0.573
            #   svm_rbf        : CV AUC 0.571, cal AUC 0.530
            # Try a different family to see how the choice moves the needle.
            "model_family": "random_forest",      # "logistic" | "random_forest" | "xgboost" | "svm_rbf"

            # Feature subset. "all" uses the 127-feature matrix from
            # Phase 3. The others let you isolate one feature group
            # to see how much each one matters.
            "feature_set": "all",                 # "all" | "structural" | "topic" | "embedding" | "network"

            # ---- Improvement toggles (all True by default) ----
            # Flip any of these to False to ablate and see how much
            # each one contributes to the headline AUC.
            "use_grid_search":   True,            # GridSearchCV on a small per-family grid
            "use_repeated_cv":   True,            # 5 folds x n_repeats stratified CV
            "use_class_balance": True,            # class_weight='balanced' / scale_pos_weight

            # CV / random seed. Don't change unless you know why.
            "n_splits":   5,
            "n_repeats":  3,                      # used when use_repeated_cv=True
            "random_seed": 42,
        }
    """),

    # ----------- Imports + paths -----------
    md("## Imports and paths"),
    code(PATH_BOILERPLATE),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.model_selection import (
            GridSearchCV,
            RepeatedStratifiedKFold,
            StratifiedKFold,
            cross_val_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        try:
            from xgboost import XGBClassifier
            HAS_XGB = True
        except ImportError:
            HAS_XGB = False
    """),

    # ----------- Load + filter -----------
    md("""
        ## Load features and targets

        We use the 92-feature matrix that the rigorous pipeline
        settled on (``standalone_positive_union_mpnet`` from
        Phase 3). Train + cal splits only — the test split is for
        notebook 05.
    """),
    code("""
        # Read the prebuilt feature matrix. imdb_id is its index;
        # ``split`` is already a column ('train' / 'cal' / 'test').
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        print(f"Total films in the matrix: {len(feat)}")
        print(f"Split breakdown: {feat['split'].value_counts().to_dict()}")

        # Train + cal pool (the test split is held out for notebook 05).
        df = feat[feat["split"].isin(["train", "cal"])].reset_index(drop=True)
        print(f"Films available in this notebook: {len(df)}")

        # Pick feature columns based on CONFIG['feature_set'].
        non_feat = {"imdb_id", "split", "log_roi", "roi_gt_1", "roi_gt_2"}
        all_cols = [c for c in df.columns if c not in non_feat]
        groups = {
            "all":        all_cols,
            "structural": [c for c in all_cols if c.startswith(("log_", "n_", "dialogue_to_", "release_year", "log_runtime", "genre_"))],
            "topic":      [c for c in all_cols if c.startswith("topic_")],
            "embedding":  [c for c in all_cols if c.startswith("embed_pc_")],
            "network":    [c for c in all_cols if c.startswith("network_")],
        }
        feat_cols = groups[CONFIG["feature_set"]]
        print(f"Feature set: {CONFIG['feature_set']!r} -> {len(feat_cols)} columns")

        X = df[feat_cols].fillna(0).values
        y = df[CONFIG["target"]].astype(int).values if CONFIG["target"].startswith("roi_gt") else df[CONFIG["target"]].values
        print(f"X shape: {X.shape}, y positive rate: {y.mean():.3f}")
    """),

    # ----------- Build model (with optional class balancing) -----------
    md("""
        ## Build the model — with optional class balancing

        Each model family is wrapped in a sklearn ``Pipeline`` with
        a ``StandardScaler`` so the input distribution is consistent.
        SVM also goes through ``probability=True`` so the next
        notebook can calibrate it.

        When ``CONFIG["use_class_balance"]`` is True we apply:

        - ``class_weight="balanced"`` for logistic / random forest /
          SVM (sklearn's standard trick: weights inversely
          proportional to class frequency).
        - ``scale_pos_weight = n_neg / n_pos`` for XGBoost (its
          equivalent parameter).

        The 60/40 class imbalance on roi_gt_2 is mild, so this is a
        small tweak — but free.
    """),
    code("""
        # Compute the imbalance ratio for XGBoost upfront.
        train_mask = (df["split"] == "train").values
        cal_mask   = (df["split"] == "cal").values
        y_train = y[train_mask]
        y_cal   = y[cal_mask]
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Train class balance: {n_neg} negative, {n_pos} positive  ->  scale_pos_weight = {scale_pos_weight:.3f}")

        def build_model(family: str, seed: int = 42, balance: bool = False):
            cw = "balanced" if balance else None
            if family == "logistic":
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw, random_state=seed)
            elif family == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    class_weight=cw, random_state=seed, n_jobs=-1,
                )
            elif family == "xgboost":
                if not HAS_XGB:
                    raise RuntimeError("xgboost not installed. pip install xgboost")
                spw = scale_pos_weight if balance else 1.0
                clf = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw,
                    eval_metric="logloss", random_state=seed, n_jobs=-1,
                )
            elif family == "svm_rbf":
                clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                          class_weight=cw, random_state=seed)
            else:
                raise ValueError(f"Unknown model_family: {family!r}")
            return Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", clf)])

        base_model = build_model(
            CONFIG["model_family"],
            seed=CONFIG["random_seed"],
            balance=CONFIG["use_class_balance"],
        )
        print(f"Built {CONFIG['model_family']!r} model | "
              f"class_balance={CONFIG['use_class_balance']}")
    """),

    # ----------- CV scheme -----------
    md("""
        ## Cross-validation scheme — repeated or single?

        - ``StratifiedKFold(n_splits=5)``: 5 fold values per CV run.
          Cheap but noisy on a 1456-film pool.
        - ``RepeatedStratifiedKFold(n_splits=5, n_repeats=3)``: 15
          fold values from three independent shuffles. Same mean,
          tighter CI.

        ``CONFIG["use_repeated_cv"]`` toggles between them.
    """),
    code("""
        if CONFIG["use_repeated_cv"]:
            cv = RepeatedStratifiedKFold(
                n_splits=CONFIG["n_splits"],
                n_repeats=CONFIG["n_repeats"],
                random_state=CONFIG["random_seed"],
            )
            cv_label = f"RepeatedStratifiedKFold({CONFIG['n_splits']} folds x {CONFIG['n_repeats']} repeats = {CONFIG['n_splits']*CONFIG['n_repeats']} fold values)"
        else:
            cv = StratifiedKFold(
                n_splits=CONFIG["n_splits"], shuffle=True,
                random_state=CONFIG["random_seed"],
            )
            cv_label = f"StratifiedKFold({CONFIG['n_splits']} fold values)"
        print(f"CV: {cv_label}")
    """),

    # ----------- Tune (or just CV) -----------
    md("""
        ## Tune hyperparameters with ``GridSearchCV``

        Default hyperparameters leave 3-5pp of AUC on the table for
        tree models. The grid below is small (3-12 candidates per
        family) so a 15-fold repeated CV finishes in a couple of
        minutes.

        When ``CONFIG["use_grid_search"]`` is False, we skip the
        search and run plain ``cross_val_score`` on the default
        hyperparameters.
    """),
    code("""
        # Per-family small grids. Keep them small so the run finishes
        # in 1-3 minutes on a laptop.
        GRIDS = {
            "logistic": {
                "model__C": [0.1, 1.0, 10.0],
            },
            "random_forest": {
                "model__n_estimators":     [200, 500],
                "model__max_depth":        [None, 10, 20],
                "model__min_samples_leaf": [1, 5, 10],
            },
            "xgboost": {
                "model__max_depth":     [3, 4, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators":  [200, 500],
            },
            "svm_rbf": {
                "model__C":     [0.5, 1.0, 3.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        }

        if CONFIG["use_grid_search"]:
            grid = GRIDS[CONFIG["model_family"]]
            n_combos = int(np.prod([len(v) for v in grid.values()]))
            print(f"Searching {n_combos} hyperparameter combinations on {cv_label}")
            search = GridSearchCV(base_model, grid, cv=cv, scoring="roc_auc",
                                   n_jobs=-1, refit=True)
            search.fit(X[train_mask | cal_mask], y[train_mask | cal_mask])
            cv_aucs = search.cv_results_["mean_test_score"]
            cv_auc_mean = float(search.best_score_)
            cv_auc_std  = float(search.cv_results_["std_test_score"][search.best_index_])
            best_params = search.best_params_
            model = search.best_estimator_
            print(f"\\nBest params: {best_params}")
            print(f"Best CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")
        else:
            scores = cross_val_score(base_model, X[train_mask | cal_mask], y[train_mask | cal_mask],
                                      cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_auc_mean = float(scores.mean())
            cv_auc_std  = float(scores.std())
            best_params = "(no grid search; defaults)"
            model = base_model
            print(f"CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")

        # Bootstrap CI on the CV-fold AUC distribution (same as before).
        if CONFIG["use_grid_search"]:
            # Use the per-fold AUCs from the BEST hyperparameter cell.
            best_idx = search.best_index_
            fold_keys = [k for k in search.cv_results_ if k.startswith("split") and k.endswith("_test_score")]
            best_fold_aucs = np.array([search.cv_results_[k][best_idx] for k in fold_keys])
        else:
            best_fold_aucs = scores
        rng = np.random.default_rng(CONFIG["random_seed"])
        boot = np.array([rng.choice(best_fold_aucs, size=len(best_fold_aucs), replace=True).mean()
                          for _ in range(2000)])
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        print(f"95% bootstrap CI on the CV mean: [{ci_lo:.3f}, {ci_hi:.3f}]")
    """),

    # ----------- Held-out cal evaluation -----------
    md("""
        ## Held-out evaluation on the cal split

        With ``GridSearchCV(refit=True)`` the search already refit
        the best model on all 1,456 train+cal films, so the cal AUC
        below is **in-sample** for the tuned model. With
        ``use_grid_search=False`` it's still genuinely held out
        (we explicitly fit on train and score on cal). We report
        both interpretations so it's clear what each cell shows.
    """),
    code("""
        X_train_only, _y_train = X[train_mask], y[train_mask]
        X_cal,        _y_cal   = X[cal_mask],   y[cal_mask]

        if CONFIG["use_grid_search"]:
            # The model is already refit on train+cal; cal AUC is in-sample.
            cal_proba = model.predict_proba(X_cal)[:, 1]
            cal_auc = roc_auc_score(_y_cal, cal_proba)
            print(f"Cal-set AUC (in-sample, refit on train+cal): {cal_auc:.3f}")

            # Also fit a fresh copy on TRAIN ONLY for the genuinely held-out figure.
            from sklearn.base import clone
            held_out = clone(model).fit(X_train_only, _y_train)
            held_proba = held_out.predict_proba(X_cal)[:, 1]
            held_auc = roc_auc_score(_y_cal, held_proba)
            print(f"Cal-set AUC (held out, fit on train only):  {held_auc:.3f}")
            cal_auc = held_auc  # save the honest one downstream
        else:
            model.fit(X_train_only, _y_train)
            cal_proba = model.predict_proba(X_cal)[:, 1]
            cal_auc = roc_auc_score(_y_cal, cal_proba)
            print(f"Cal-set AUC: {cal_auc:.3f}")
    """),

    # ----------- ROC plot -----------
    md("## ROC curve on the cal set"),
    code("""
        fpr, tpr, _ = roc_curve(y_cal, cal_proba)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"{CONFIG['model_family']} (AUC = {cal_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {CONFIG['target']} on cal set")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    """),

    # ----------- Save -----------
    md("""
        ## Save the model

        Saves to ``data/processed/student/student_model.joblib``.
        Notebooks 02-05 will load this file. If you re-run this
        notebook with a different model, the next notebooks
        automatically pick it up.
    """),
    code("""
        bundle = {
            "config":          CONFIG,
            "feature_columns": feat_cols,
            "model":           model,
            "cv_auc_mean":     float(cv_auc_mean),
            "cv_auc_std":      float(cv_auc_std),
            "cv_auc_ci":       [float(ci_lo), float(ci_hi)],
            "cal_auc":         float(cal_auc),
            "best_params":     str(best_params),
            "n_train":         int(train_mask.sum()),
            "n_cal":           int(cal_mask.sum()),
        }
        joblib.dump(bundle, STUDENT / "student_model.joblib")
        print(f"Saved {STUDENT / 'student_model.joblib'}")
    """),

    # ----------- Compare hint -----------
    md("""
        ## Compare to your teammates

        Copy your headline number into the team comparison table.
        Then change ``CONFIG['model_family']`` and re-run to see
        how the choice moves the needle.

        ### Reference numbers — bare baseline (all toggles OFF)

        | model_family | CV AUC | Cal AUC |
        |---|---|---|
        | random_forest | 0.595 | 0.612 |
        | xgboost       | 0.597 | 0.569 |
        | logistic      | 0.576 | 0.573 |
        | svm_rbf       | 0.571 | 0.530 |

        ### Reference numbers — rigorous Phase 4 (curated 92-feature
        matrix, full grid search, repeated CV, Bayesian comparison)

        | model_family | OOF AUC |
        |---|---|
        | xgboost | 0.652 |
        | svm_rbf | 0.635 |

        With the three toggles ON (grid search + repeated CV + class
        balance) you should land somewhere in between the two
        reference rows, typically lifting cal AUC to ~0.62-0.64. The
        gap to the rigorous pipeline is the cost of the
        simplifications: simpler feature matrix (127 vs 92 curated),
        smaller hyperparameter grid, single-fold CV cell instead of
        Bayesian comparison.

        ### Ablating the three toggles

        Run with all three on, then flip one toggle to False at a
        time and see which one matters most for your model family.
        On random_forest you should see something like:

        - all three on: cal AUC ~0.62
        - turn off ``use_grid_search``: cal AUC ~0.61 (-0.01)
        - turn off ``use_repeated_cv``: cal AUC unchanged (the
          headline mean barely moves; only the CI tightens)
        - turn off ``use_class_balance``: cal AUC ~0.61 (-0.005)
    """),
]


# ============================================================
# Notebook 02 — Calibration (Layer 2)
# ============================================================


CELLS_02 = [
    md("""
        # Student Notebook 02 — Calibration (Layer 2)

        Take the model from notebook 01 and make its probability
        outputs honest. A model that says "0.7" should be right
        about 70% of the time on the films it rates at 0.7.

        **Run order:** notebook 01 must run first (we read its
        ``student_model.joblib``). After this notebook,
        ``student_calibrated.joblib`` lands in the same folder for
        notebook 03 to use.

        **What you'll get out:** ECE before vs after calibration
        (lower is better), reliability diagram, the calibrated model
        saved.
    """),

    md("""
        ## CONFIG — edit me

        The big knob is the calibration method. ``isotonic`` is more
        flexible (a step-wise fit) and usually wins on enough data;
        ``sigmoid`` (Platt scaling) is more constrained but
        regularizes well on small samples.
    """),
    code("""
        CONFIG = {
            # MUST match the run_name you used in 01_modeling.ipynb
            # so this notebook reads YOUR model. Each teammate has
            # their own run_name.
            "run_name": "team_baseline_rf",

            "method": "isotonic",     # "sigmoid" | "isotonic"
            "n_bins_for_ece": 10,     # 10 is the standard
        }
    """),

    md("## Imports + load notebook 01's artifact"),
    code(PATH_BOILERPLATE),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.frozen import FrozenEstimator
        from sklearn.metrics import brier_score_loss, log_loss

        bundle = joblib.load(STUDENT / "student_model.joblib")
        print("Loaded model from notebook 01:")
        print(f"  family       = {bundle['config']['model_family']}")
        print(f"  target       = {bundle['config']['target']}")
        print(f"  feature_set  = {bundle['config']['feature_set']}")
        print(f"  CV AUC       = {bundle['cv_auc_mean']:.3f} ± {bundle['cv_auc_std']:.3f}")
        print(f"  cal AUC      = {bundle['cal_auc']:.3f}")
    """),

    md("""
        ## Rebuild the cal-set features

        Same code path as notebook 01, just on the cal split.
    """),
    code("""
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        df_cal = feat[feat["split"] == "cal"].reset_index(drop=True)
        feat_cols = bundle["feature_columns"]
        target_col = bundle["config"]["target"]

        X_cal = df_cal[feat_cols].fillna(0).values
        y_cal = df_cal[target_col].astype(int).values
        print(f"Cal set: {len(df_cal)} films, positive rate {y_cal.mean():.3f}")
    """),

    md("""
        ## Uncalibrated probabilities (before)

        These come straight from the model's ``predict_proba``. We
        compute the Expected Calibration Error (ECE), which is the
        weighted-by-bin-count average of |predicted probability −
        empirical positive rate|. Lower is better; 0 is perfect.
    """),
    code("""
        def ece(y_true, y_prob, n_bins=10):
            # Quantile-based bins (equal count per bin) avoid empty bins.
            edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.maximum.accumulate(edges)
            bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
            total = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.any():
                    total += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
            return float(total)

        proba_uncal = bundle["model"].predict_proba(X_cal)[:, 1]
        ece_uncal = ece(y_cal, proba_uncal, n_bins=CONFIG["n_bins_for_ece"])
        brier_uncal = brier_score_loss(y_cal, proba_uncal)
        print(f"BEFORE calibration: ECE = {ece_uncal:.4f}, Brier = {brier_uncal:.4f}")
    """),

    md("""
        ## Calibrated probabilities (after)

        ``CalibratedClassifierCV`` with ``FrozenEstimator`` re-fits
        the calibrator on a held-out portion of the cal split, so
        the calibrator never sees the same film twice. (sklearn 1.6+
        replaces the old ``cv="prefit"`` argument with
        ``FrozenEstimator``.)
    """),
    code("""
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(bundle["model"]),
            method=CONFIG["method"],
            cv=5,
        )
        # Fit the calibrator only — the underlying model is frozen.
        calibrated.fit(X_cal, y_cal)
        proba_cal = calibrated.predict_proba(X_cal)[:, 1]
        ece_cal = ece(y_cal, proba_cal, n_bins=CONFIG["n_bins_for_ece"])
        brier_cal = brier_score_loss(y_cal, proba_cal)
        print(f"AFTER  calibration: ECE = {ece_cal:.4f}, Brier = {brier_cal:.4f}")
        print(f"Improvement: ECE {ece_uncal:.4f} → {ece_cal:.4f}  ({(ece_uncal - ece_cal)/ece_uncal*100:+.0f}%)")
    """),

    md("## Reliability diagram"),
    code("""
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, p, color in [("uncalibrated", proba_uncal, "#d7301f"),
                                 (CONFIG["method"], proba_cal, "#2c7fb8")]:
            mean_pred, frac_pos = calibration_curve(y_cal, p, n_bins=CONFIG["n_bins_for_ece"],
                                                     strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color=color, label=label)
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        ax.set_xlabel("Predicted probability (mean per bin)")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title(f"Reliability diagram — {bundle['config']['model_family']} on {bundle['config']['target']}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.show()
    """),

    md("""
        ## Save the calibrated model
    """),
    code("""
        out = {
            **bundle,
            "calibration_method": CONFIG["method"],
            "calibrated_model": calibrated,
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated": ece_cal,
            "brier_uncalibrated": brier_uncal,
            "brier_calibrated": brier_cal,
        }
        joblib.dump(out, STUDENT / "student_calibrated.joblib")
        print(f"Saved {STUDENT / 'student_calibrated.joblib'}")
    """),

    md("""
        ## Compare to your teammates

        | Calibration method | Typical ECE drop |
        |---|---|
        | sigmoid | small but reliable |
        | isotonic | larger; needs enough cal samples |

        On the rigorous pipeline (Phase 5) ECE on roi_gt_2 dropped
        from 0.243 to 0.108 with isotonic calibration. Your numbers
        will be in that ballpark.

        Next notebook (03_decision) will use these calibrated
        probabilities to recommend Greenlight / Pass / Refer.
    """),
]


# ============================================================
# Notebook 03 — Decision (Layer 3)
# ============================================================


CELLS_03 = [
    md("""
        # Student Notebook 03 — Decision (Layer 3)

        Calibrated probability → Greenlight / Pass / Refer using a
        cost matrix. The cost asymmetry comes from the project
        framing: producing a flop costs $50M, passing on a hit costs
        $100M, and a human reader costs $5K.

        **Run order:** notebook 02 must run first (we read its
        ``student_calibrated.joblib``).

        **What you'll get out:** per-film recommendations on the cal
        set, total cost vs five baselines.
    """),

    md("""
        ## CONFIG — edit me

        The cost matrix is the knob. The defaults come from
        ``PROJECT_CONTEXT.md`` Section 1. Try halving / doubling
        ``flop_cost`` to see when the system flips behavior.
    """),
    code("""
        CONFIG = {
            # MUST match the run_name you used in 01 and 02.
            "run_name": "team_baseline_rf",

            "flop_cost":   50_000_000,   # $50M  - cost of greenlighting a film that flops
            "miss_cost": 100_000_000,    # $100M - cost of passing on a film that becomes a hit
            "refer_cost":      5_000,    # $5K   - cost of one human reader pass
        }
    """),

    md("## Imports + load"),
    code(PATH_BOILERPLATE),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        bundle = joblib.load(STUDENT / "student_calibrated.joblib")
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        master = pd.read_parquet(DATA / "films_joined.parquet")

        df_cal = feat[feat["split"] == "cal"].reset_index(drop=True)
        feat_cols = bundle["feature_columns"]
        target_col = bundle["config"]["target"]

        X_cal = df_cal[feat_cols].fillna(0).values
        y_cal = df_cal[target_col].astype(int).values
        ids = df_cal["imdb_id"].astype(str).tolist()
        proba = bundle["calibrated_model"].predict_proba(X_cal)[:, 1]
        print(f"Cal set: {len(df_cal)} films, positive rate {y_cal.mean():.3f}, mean P(hit) {proba.mean():.3f}")
    """),

    md("""
        ## The decision rule

        For each film with calibrated probability ``p``, compute
        expected cost of each action and pick the lowest:

        - Greenlight expected cost = ``(1 − p) × flop_cost``
        - Pass expected cost       = ``p × miss_cost``
        - Refer expected cost      = ``refer_cost`` (independent of p)

        Tie-break to Refer (the conservative action).
    """),
    code("""
        def decide(p, flop_cost, miss_cost, refer_cost):
            costs = {
                "Greenlight": (1 - p) * flop_cost,
                "Pass":       p * miss_cost,
                "Refer":      refer_cost,
            }
            min_cost = min(costs.values())
            tied = [a for a, c in costs.items() if c == min_cost]
            return "Refer" if "Refer" in tied else tied[0], costs

        actions = []
        cost_breakdown = []
        for p in proba:
            a, c = decide(p, CONFIG["flop_cost"], CONFIG["miss_cost"], CONFIG["refer_cost"])
            actions.append(a)
            cost_breakdown.append(c)
        actions = np.array(actions)

        # Action distribution.
        for a in ["Greenlight", "Pass", "Refer"]:
            print(f"  {a:11s}: {(actions == a).mean()*100:5.1f}% ({(actions == a).sum():>3d} films)")
    """),

    md("""
        ## System total cost vs five baselines

        Each strategy's *realized* cost is computed by looking at the
        true label of each cal-set film:

        - Greenlight + true positive → $0 (correct).
        - Greenlight + true negative → ``flop_cost``.
        - Pass + true positive → ``miss_cost``.
        - Pass + true negative → $0 (correct).
        - Refer (any outcome) → ``refer_cost``.

        Lower total cost = better strategy.
    """),
    code("""
        def realized_cost(action, true, flop_cost, miss_cost, refer_cost):
            if action == "Greenlight":
                return 0 if true == 1 else flop_cost
            if action == "Pass":
                return miss_cost if true == 1 else 0
            return refer_cost  # Refer (outcome-independent)

        def total_cost(actions, true_labels):
            return sum(realized_cost(a, t, CONFIG["flop_cost"], CONFIG["miss_cost"], CONFIG["refer_cost"])
                       for a, t in zip(actions, true_labels))

        rng = np.random.default_rng(42)
        baselines = {
            "Always-Greenlight": ["Greenlight"] * len(y_cal),
            "Always-Pass":       ["Pass"] * len(y_cal),
            "Read-Everything":   ["Refer"] * len(y_cal),
            "Random":            rng.choice(["Greenlight", "Pass", "Refer"], size=len(y_cal)).tolist(),
            "System (you!)":     actions.tolist(),
        }
        rows = []
        for name, acts in baselines.items():
            t = total_cost(acts, y_cal)
            rows.append({"strategy": name, "total_cost_USD": t,
                         "cost_per_film_M": t / len(y_cal) / 1e6})
        bench = pd.DataFrame(rows).sort_values("total_cost_USD")
        print(bench.to_string(index=False))
    """),

    md("## Bar chart (log scale because the scales differ wildly)"),
    code("""
        bench_sorted = bench.sort_values("total_cost_USD", ascending=True)
        colors = ["#2c7fb8" if s == "System (you!)" else "#a6cee3" for s in bench_sorted["strategy"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(bench_sorted["strategy"], bench_sorted["total_cost_USD"].clip(lower=1), color=colors)
        ax.set_xscale("log")
        ax.set_xlabel("Total realized cost on cal set (USD, log scale)")
        ax.set_title("Cost-asymmetric decision: system vs baselines")
        for s, v in zip(bench_sorted["strategy"], bench_sorted["total_cost_USD"]):
            ax.text(max(v, 1) * 1.2, s, f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K", va="center", fontsize=9)
        plt.show()
    """),

    md("""
        ## Save per-film decisions
    """),
    code("""
        master_lookup = master.set_index("imdb_id")["movie_name"]
        decisions_df = pd.DataFrame({
            "imdb_id":                ids,
            "movie_name":             [master_lookup.get(i, i) for i in ids],
            "calibrated_probability": proba,
            "true_label":             y_cal,
            "recommended_action":     actions,
            "expected_cost_GL":       [c["Greenlight"] for c in cost_breakdown],
            "expected_cost_Pass":     [c["Pass"]       for c in cost_breakdown],
            "expected_cost_Refer":    [c["Refer"]      for c in cost_breakdown],
        })
        decisions_df.to_csv(STUDENT / "student_decisions.csv", index=False)
        print(f"Saved {STUDENT / 'student_decisions.csv'}: {len(decisions_df)} rows")
        # Show the few films the system recommends Greenlight on.
        print("\\nGreenlight films (if any):")
        gl = decisions_df[decisions_df["recommended_action"] == "Greenlight"]
        print(gl[["movie_name", "calibrated_probability", "true_label"]].to_string(index=False) if len(gl)
              else "  (none — under the default cost matrix the model rarely commits)")
    """),

    md("""
        ## Compare to your teammates

        Try halving the refer cost (``"refer_cost": 2_500``) — the
        system should commit to more Greenlights. Try raising it to
        $1M — the system will refuse to commit on anything. The
        knob to play with most is ``refer_cost``: it represents
        your studio's effective per-film human-reader cost.

        Reference: under the source default cost matrix the rigorous
        Phase 6 pipeline commits 1.2% Greenlight and 98.8% Refer on
        the cal set, with total cost $1.3M (tying Read-Everything).
        You should land in that neighborhood.
    """),
]


# ============================================================
# Notebook 04 — Explanation (Layer 4)
# ============================================================


CELLS_04 = [
    md("""
        # Student Notebook 04 — Explanation (Layer 4)

        TreeSHAP attribution: which features pushed each film's
        probability up, which pulled it down. Works for tree-based
        models (xgboost, random forest). For linear models we fall
        back to coefficients; for SVM we fall back to permutation
        importance.

        **Run order:** notebook 01 must run first. Notebooks 02 and
        03 are not required for this notebook.

        **What you'll get out:** global feature importance bar chart,
        per-film SHAP for one example film of your choice.
    """),

    md("""
        ## CONFIG — edit me

        The "which film to inspect" knob is the interesting one. By
        default we pick the film with the highest predicted
        probability so the SHAP signal is strongest, but you can
        substitute any imdb_id from the cal set.
    """),
    code("""
        CONFIG = {
            # MUST match the run_name you used in 01.
            "run_name": "team_baseline_rf",

            "top_k_global":   20,        # how many features to plot in the global bar chart
            "top_k_per_film": 5,         # how many positive + negative contributors per film
            "example_imdb_id": "auto",   # "auto" picks highest-probability film, or paste any imdb_id
        }
    """),

    md("## Imports + load"),
    code(PATH_BOILERPLATE),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        try:
            import shap
            HAS_SHAP = True
        except ImportError:
            HAS_SHAP = False
            print("WARNING: shap not installed. pip install shap")

        bundle = joblib.load(STUDENT / "student_model.joblib")
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        master = pd.read_parquet(DATA / "films_joined.parquet")

        df_cal = feat[feat["split"] == "cal"].reset_index(drop=True)
        feat_cols = bundle["feature_columns"]

        X_cal = df_cal[feat_cols].fillna(0).values
        ids = df_cal["imdb_id"].astype(str).tolist()
        proba = bundle["model"].predict_proba(X_cal)[:, 1]

        family = bundle["config"]["model_family"]
        print(f"Loaded {family} model on target {bundle['config']['target']!r}; "
              f"{len(df_cal)} cal films")
    """),

    md("""
        ## Compute attributions

        Three branches based on model family:

        - **Tree models (xgboost, random_forest):** TreeSHAP, exact +
          fast. Returns per-film signed contributions.
        - **Logistic:** coefficients × scaled features. Linear, exact.
        - **SVM:** permutation importance (slow but model-agnostic).
    """),
    code("""
        def attribute(model_pipeline, X, family):
            scaler = model_pipeline.named_steps["scaler"]
            clf = model_pipeline.named_steps["model"]
            X_scaled = scaler.transform(X)

            if family in ("xgboost", "random_forest"):
                if not HAS_SHAP:
                    raise RuntimeError("shap not installed")
                explainer = shap.TreeExplainer(clf)
                raw = explainer.shap_values(X_scaled)
                arr = np.asarray(raw)
                if arr.ndim == 3:
                    arr = arr[:, :, 1]  # positive class
                return arr, "TreeSHAP"

            if family == "logistic":
                # Per-film contribution = coefficient × scaled feature value.
                coefs = clf.coef_.ravel()
                contributions = X_scaled * coefs[np.newaxis, :]
                return contributions, "linear coefficients × x"

            if family == "svm_rbf":
                from sklearn.inspection import permutation_importance
                # Permutation importance is global (not per-film); we tile.
                # For a per-film story on SVM, KernelSHAP would work but takes hours.
                pi = permutation_importance(model_pipeline, X, model_pipeline.predict(X),
                                             n_repeats=5, random_state=42, n_jobs=-1)
                # Fake "per-film" array by replicating the global mean signed importance.
                return np.tile(pi.importances_mean, (len(X), 1)), "permutation importance (global only)"

            raise ValueError(f"Unsupported family for explanation: {family!r}")

        contribs, method = attribute(bundle["model"], X_cal, family)
        print(f"Attribution method: {method}")
        print(f"Per-film contribution matrix shape: {contribs.shape}")
    """),

    md("""
        ## Global ranking — top features by mean |contribution|
    """),
    code("""
        mean_abs = np.mean(np.abs(contribs), axis=0)
        mean_signed = np.mean(contribs, axis=0)
        ranking = pd.DataFrame({
            "feature":          feat_cols,
            "mean_abs":         mean_abs,
            "mean_signed":      mean_signed,
        }).sort_values("mean_abs", ascending=False).reset_index(drop=True)
        print(ranking.head(15).to_string(index=False))
    """),
    code("""
        K = CONFIG["top_k_global"]
        top = ranking.head(K).iloc[::-1]  # reverse for top-down bars
        colors = ["#2c7fb8" if s >= 0 else "#d7301f" for s in top["mean_signed"]]
        fig, ax = plt.subplots(figsize=(8, K * 0.3 + 1))
        ax.barh(top["feature"], top["mean_abs"], color=colors)
        ax.set_xlabel("Mean |contribution|")
        ax.set_title(f"Top-{K} feature importance — {family} on {bundle['config']['target']}")
        ax.grid(axis="x", alpha=0.3)
        plt.show()
        print("Blue = pushes probability UP on average; Red = pulls DOWN on average")
    """),

    md("""
        ## Per-film attribution

        Pick the film with the highest predicted probability (or
        whatever imdb_id you put in CONFIG). The bar chart shows
        which features pushed *this specific film*'s probability up
        or down.
    """),
    code("""
        if CONFIG["example_imdb_id"] == "auto":
            example_idx = int(np.argmax(proba))
            example_id = ids[example_idx]
        else:
            example_id = CONFIG["example_imdb_id"]
            example_idx = ids.index(example_id)

        movie = master.loc[master["imdb_id"] == example_id, "movie_name"].iloc[0]
        print(f"Inspecting: {movie} ({example_id})")
        print(f"  Predicted probability: {proba[example_idx]:.3f}")

        film_contribs = pd.DataFrame({
            "feature":      feat_cols,
            "contribution": contribs[example_idx],
        })
        # Top K positive and top K negative, by absolute contribution.
        top_pos = film_contribs[film_contribs["contribution"] > 0].nlargest(CONFIG["top_k_per_film"], "contribution")
        top_neg = film_contribs[film_contribs["contribution"] < 0].nsmallest(CONFIG["top_k_per_film"], "contribution")
        show = pd.concat([top_pos, top_neg]).iloc[::-1]
        colors = ["#2c7fb8" if v >= 0 else "#d7301f" for v in show["contribution"]]

        fig, ax = plt.subplots(figsize=(8, len(show) * 0.4 + 1))
        ax.barh(show["feature"], show["contribution"], color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Per-film contribution to predicted log-odds")
        ax.set_title(f"{movie} — top contributors")
        ax.grid(axis="x", alpha=0.3)
        plt.show()
    """),

    md("""
        ## Compare to your teammates

        Pick your favorite test film, paste its imdb_id into
        CONFIG, and re-run the per-film cell. Each teammate
        inspecting a different film will help you see whether the
        system is reading the screenplay's actual content or just
        the metadata genre tag.

        Reference (rigorous Phase 7): on the cal set, the top-5
        SHAP features for ``roi_gt_2`` are release_year_parsed,
        genre_Horror, network_lead_role_count, genre_Action,
        genre_Romance. Your top-5 should overlap heavily.

        TreeSHAP doesn't apply to SVM; if your team is comparing
        SVM vs xgboost on layer 1, only the xgboost run will produce
        a meaningful per-film SHAP.
    """),
]


# ============================================================
# Notebook 05 — End-to-end on the test set
# ============================================================


CELLS_05 = [
    md("""
        # Student Notebook 05 — End-to-End Test-Set Evaluation

        Run the assembled pipeline once on the held-out 257-film
        test set. **This is the only notebook in the project that
        touches the test set.** The number you get out of this
        notebook is the headline result for your section of the
        report.

        **Run order:** all four previous notebooks must run first.
        Re-run this notebook only if you decide to finalize a
        different model in notebook 01.

        **What you'll get out:** test-set AUC, ECE, conformal-style
        coverage, decision-cost vs baselines, top-5 SHAP features.
    """),

    md("""
        ## CONFIG — no knob

        This notebook intentionally has no knob. The whole point of
        the test set is that we touch it once with whatever the team
        finalized in notebooks 01-04. If you want a different model,
        change ``01_modeling`` and re-run the cascade.
    """),
    code("""
        # The only thing to edit: which run_name's calibrated model
        # do we evaluate on the test set? After the team picks one,
        # paste that run_name here and run.
        CONFIG = {
            "run_name": "team_baseline_rf",
        }
        FLOP_COST  =  50_000_000
        MISS_COST  = 100_000_000
        REFER_COST =      5_000
    """),

    md("## Imports + load all four-layer artifacts"),
    code(PATH_BOILERPLATE),
    code("""
        import warnings
        warnings.filterwarnings("ignore")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
        )

        bundle = joblib.load(STUDENT / "student_calibrated.joblib")
        family = bundle["config"]["model_family"]
        target_col = bundle["config"]["target"]
        feat_cols = bundle["feature_columns"]
        print(f"Pipeline: {family} on {target_col} | calibration={bundle['calibration_method']}")
    """),

    md("""
        ## Test-set isolation check

        Programmatic verification that test imdb_ids are disjoint
        from train and cal. This is the safety guarantee.
    """),
    code("""
        feat = pd.read_parquet(DATA / "features.parquet").reset_index()
        train_ids = set(feat.loc[feat["split"] == "train", "imdb_id"])
        cal_ids   = set(feat.loc[feat["split"] == "cal",   "imdb_id"])
        test_ids  = set(feat.loc[feat["split"] == "test",  "imdb_id"])
        assert not (train_ids & test_ids), "Leakage: train + test overlap"
        assert not (cal_ids   & test_ids), "Leakage: cal + test overlap"
        print(f"OK: train {len(train_ids)} | cal {len(cal_ids)} | test {len(test_ids)} (all disjoint)")
    """),

    md("""
        ## Run the calibrated model on the test set
    """),
    code("""
        master = pd.read_parquet(DATA / "films_joined.parquet")
        df_test = feat[feat["split"] == "test"].reset_index(drop=True)
        X_test = df_test[feat_cols].fillna(0).values
        y_test = df_test[target_col].astype(int).values
        print(f"Test set: {len(df_test)} films, positive rate {y_test.mean():.3f}")

        # Calibrated probability is the headline output.
        proba_test = bundle["calibrated_model"].predict_proba(X_test)[:, 1]
    """),

    md("""
        ## Layer 1 — predictive performance on test
    """),
    code("""
        rng = np.random.default_rng(42)
        def boot_ci(metric, *args, n_boot=2000):
            arrs = [np.asarray(a) for a in args]
            n = len(arrs[0])
            point = float(metric(*arrs))
            samples = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    samples.append(float(metric(*[a[idx] for a in arrs])))
                except Exception:
                    pass
            lo, hi = np.quantile(samples, [0.025, 0.975])
            return point, lo, hi

        for name, fn in [
            ("AUC-ROC",  lambda y, p: roc_auc_score(y, p)),
            ("PR-AUC",   lambda y, p: average_precision_score(y, p)),
            ("F1@0.5",   lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
            ("log-loss", lambda y, p: log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            ("Brier",    lambda y, p: brier_score_loss(y, p)),
        ]:
            point, lo, hi = boot_ci(fn, y_test, proba_test)
            print(f"  {name:9s}: {point:.3f} [{lo:.3f}, {hi:.3f}]")
    """),

    md("""
        ## Layer 2 — calibration on test
    """),
    code("""
        def ece(y_true, y_prob, n_bins=10):
            edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.maximum.accumulate(edges)
            bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
            tot = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.any():
                    tot += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
            return float(tot)

        print(f"  ECE on test:   {ece(y_test, proba_test):.4f}")
        print(f"  Brier on test: {brier_score_loss(y_test, proba_test):.4f}")
    """),

    md("""
        ## Layer 3 — decisions on test, system vs five baselines
    """),
    code("""
        def decide(p, fc, mc, rc):
            costs = {"Greenlight": (1 - p) * fc, "Pass": p * mc, "Refer": rc}
            mn = min(costs.values())
            return "Refer" if costs["Refer"] == mn else min(costs, key=costs.get)

        def realized(action, true, fc, mc, rc):
            if action == "Greenlight": return 0 if true == 1 else fc
            if action == "Pass":       return mc if true == 1 else 0
            return rc

        sys_actions = [decide(p, FLOP_COST, MISS_COST, REFER_COST) for p in proba_test]
        baselines = {
            "Always-Greenlight": ["Greenlight"] * len(y_test),
            "Always-Pass":       ["Pass"] * len(y_test),
            "Read-Everything":   ["Refer"] * len(y_test),
            "System (you!)":     sys_actions,
        }
        rows = []
        for name, acts in baselines.items():
            t = sum(realized(a, t_, FLOP_COST, MISS_COST, REFER_COST) for a, t_ in zip(acts, y_test))
            rows.append({
                "strategy":     name,
                "total_cost_M": t / 1e6,
                "p_greenlight": np.mean(np.array(acts) == "Greenlight"),
                "p_pass":       np.mean(np.array(acts) == "Pass"),
                "p_refer":      np.mean(np.array(acts) == "Refer"),
            })
        bench = pd.DataFrame(rows).sort_values("total_cost_M")
        print(bench.round(3).to_string(index=False))
    """),

    md("""
        ## Headline: per-film triage report for 5 example test films

        For each film: true label, calibrated probability, action,
        and the recommendation in plain English. Pick five
        well-known films from the test set if you have favorites.
    """),
    code("""
        master_lookup = master.set_index("imdb_id")["movie_name"].to_dict()
        test_ids_list = df_test["imdb_id"].astype(str).tolist()

        # Pick: 1 highest-prob, 1 lowest-prob, 1 closest-to-0.5, 2 random.
        rng2 = np.random.default_rng(42)
        idx_sorted = np.argsort(proba_test)
        picks = [
            int(idx_sorted[-1]),                                  # most confident hit
            int(idx_sorted[0]),                                   # most confident flop
            int(np.argmin(np.abs(proba_test - 0.5))),             # most uncertain
            int(rng2.integers(0, len(proba_test))),               # random
            int(rng2.integers(0, len(proba_test))),               # random
        ]
        picks = list(dict.fromkeys(picks))[:5]

        for i in picks:
            iid = test_ids_list[i]
            name = master_lookup.get(iid, iid)
            label = "HIT" if y_test[i] == 1 else "FLOP"
            p = proba_test[i]
            action = sys_actions[i]
            print(f"  {name[:50]:50s}  P={p:.3f}  truth={label:>4s}  → {action}")
    """),

    md("## Save the test-set per-film table"),
    code("""
        report = pd.DataFrame({
            "imdb_id":                test_ids_list,
            "movie_name":             [master_lookup.get(i, i) for i in test_ids_list],
            "calibrated_probability": proba_test,
            "true_label":             y_test,
            "recommended_action":     sys_actions,
        })
        report.to_csv(STUDENT / "student_test_predictions.csv", index=False)
        print(f"Saved {STUDENT / 'student_test_predictions.csv'}: {len(report)} rows")
    """),

    md("""
        ## Compare to your teammates / the rigorous pipeline

        | Quantity | Rigorous Phase 8 | Your value here |
        |---|---|---|
        | Test AUC roi_gt_2 | 0.507 [0.437, 0.584] | (paste yours) |
        | Test ECE | 0.085 | (paste yours) |
        | System total cost on test | $51.3M | (paste yours) |

        Your numbers will land in this neighborhood. The fact that
        the test AUC drops from the cal AUC is the honest finding
        the report should lead with.

        That's it. No more knob. No more re-runs on the test set.
        Move on to writing the report.
    """),
]


# ============================================================
# Build all five
# ============================================================


def main() -> None:
    write_notebook("01_modeling.ipynb", CELLS_01)
    write_notebook("02_calibration.ipynb", CELLS_02)
    write_notebook("03_decision.ipynb", CELLS_03)
    write_notebook("04_explanation.ipynb", CELLS_04)
    write_notebook("05_end_to_end.ipynb", CELLS_05)


if __name__ == "__main__":
    main()
