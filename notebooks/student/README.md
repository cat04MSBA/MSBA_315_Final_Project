# Student Notebooks — Simplified, One Knob Per Layer

These five notebooks are the team-friendly version of Phases 4-8.
Each notebook is one layer, one CONFIG cell at the top, and short
code that anyone can read and modify in ten minutes. Swap the
config values, re-run the notebook, compare your results to your
teammates'.

The full pipeline (Phases 4-8 with pre-registration, repeated CV,
Bayesian comparison, conformal, sensitivity sweeps) lives in
`notebooks/phase_4.ipynb` through `notebooks/phase_8.ipynb`.
This `student/` folder is the "play with it" version.

## What to run first (one-time setup)

1. **Install dependencies.** Run `pip install -r requirements.txt`
   from the project root. The student notebooks need: `pandas`,
   `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `xgboost`,
   `shap`. They do *not* need `mapie` or `baycomp`.

2. **Confirm the data is built.** All five notebooks read
   `data/processed/features.parquet`, `films_joined.parquet`, and
   `split_assignments.parquet`. If those files are missing,
   regenerate them with `python -m src.experiments.run_phase4_benchmark`
   or by following the Phase 1-3 notebooks. (In this checkout
   they are already built.)

3. **From the project root, start Jupyter:** `jupyter lab`
   (or open the notebooks in VS Code).

## Run order

Run the notebooks in this order. Each one writes a small artifact
that the next one reads.

```
01_modeling.ipynb       → data/processed/student/student_model.joblib
02_calibration.ipynb    → data/processed/student/student_calibrated.joblib
03_decision.ipynb       → data/processed/student/student_decisions.csv
04_explanation.ipynb    → reads the model from 01; no save
05_end_to_end.ipynb     → reads everything; runs on the test set once
```

If you re-run `01_modeling.ipynb` with a different model, the
later notebooks will pick up your new choice automatically.

## What each notebook does

| Notebook | What it does | Knob |
|---|---|---|
| `01_modeling.ipynb` | Train one model on dialogue features. 5-fold CV AUC + held-out evaluation. | model family (logistic / random forest / xgboost / svm), feature subset, target |
| `02_calibration.ipynb` | Wrap the model so its probability outputs are honest. ECE before vs after; reliability diagram. | calibration method (sigmoid / isotonic) |
| `03_decision.ipynb` | Apply a cost matrix. Recommend Greenlight / Pass / Refer. Compare to baselines. | flop cost, miss cost, refer cost |
| `04_explanation.ipynb` | TreeSHAP global ranking + one example film. | which film to inspect |
| `05_end_to_end.ipynb` | Run the assembled pipeline on the held-out test set. **Touch the test set once.** | none (this is the honest final number) |

## How to test different models as a team

Pick one notebook each. Each person changes one knob and runs
the notebook. Share results in a single table:

| Person | Notebook | Knob change | Result |
|---|---|---|---|
| Alice | 01_modeling | `model_family = "xgboost"` | CV AUC 0.65 |
| Bob | 01_modeling | `model_family = "random_forest"` | CV AUC 0.62 |
| Chloe | 01_modeling | `model_family = "logistic"` | CV AUC 0.59 |
| Dan | 02_calibration | `method = "isotonic"` | ECE 0.10 → 0.08 |
| Eve | 03_decision | `flop_cost = 25_000_000` (half default) | system cost $0.8M |

The whole point of the knobs is to make this swap trivial: change
one line, re-run the notebook, copy the headline number into the
table.

## Important — the test set rule

`05_end_to_end.ipynb` is the **only** notebook allowed to read the
held-out test split. Do not load it anywhere else; do not tune
hyperparameters using test-set numbers; do not pick the "best
model" by test AUC. The test set gets touched exactly once at the
end of the project, and that number is what goes in the report.

The other four notebooks operate on the train + cal splits only.
Within those splits, evaluation uses cross-validation or the cal
holdout — both safe.

## Questions

If something breaks: check that you ran the previous notebook
(the artifacts cascade). If the imports fail: re-run `pip install
-r requirements.txt`. If the numbers look weird: confirm your
CONFIG cell didn't typo a model name (the notebook will print
the recognized options).

To regenerate these notebooks after editing the build script:

```
python -m notebooks.student._build_student_notebooks
```
