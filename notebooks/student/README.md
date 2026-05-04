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

## How the team coordinates — the `run_name` knob

Each notebook's CONFIG cell starts with a `run_name` field. **You
set it to a unique label for yourself**, e.g. `"alice_xgboost"`,
`"bob_random_forest"`, `"chloe_logistic"`. Convention is
`<your-name>_<what-you-tested>`.

The notebook saves your artifacts to
`data/processed/student/<your-run-name>/` so that you and your
teammates do not overwrite each other's models. Multiple teammates
can run notebook 01 in parallel with different model choices and
end up with their own independent results.

```
data/processed/student/
├── alice_xgboost/                    ← Alice's run
│   ├── student_model.joblib
│   ├── student_calibrated.joblib
│   └── student_decisions.csv
├── bob_random_forest/                ← Bob's run
│   ├── student_model.joblib
│   └── ...
└── chloe_logistic/                   ← Chloe's run
    └── ...
```

When you move from notebook 01 to 02 to 03 to 04, **keep the same
`run_name` in the CONFIG cell** so each notebook reads the right
upstream artifact.

## What to run first (one-time setup)

1. **Install dependencies.** Run `pip install -r requirements.txt`
   from the project root. The student notebooks need: `pandas`,
   `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `xgboost`,
   `shap`. They do *not* need `mapie` or `baycomp`.

2. **Confirm the data is built.** All five notebooks read
   `data/processed/features.parquet` and `films_joined.parquet`.
   In this checkout they are already built (Phases 1-3 produced
   them). If they are missing, regenerate via
   `python -m src.experiments.run_phase4_benchmark`.

3. **Start Jupyter from the project root** (`jupyter lab` or
   open the notebooks in VS Code). The notebooks auto-detect the
   project root, so they work regardless of which directory you
   open Jupyter from, but starting at the project root is the
   simplest path.

## Run order

Run the notebooks in this order. Each one writes a small artifact
that the next one reads. **Keep the same `run_name` in every
notebook's CONFIG.**

```
01_modeling.ipynb       → data/processed/student/<run_name>/student_model.joblib
02_calibration.ipynb    → data/processed/student/<run_name>/student_calibrated.joblib
03_decision.ipynb       → data/processed/student/<run_name>/student_decisions.csv
04_explanation.ipynb    → reads the model from 01; no save
05_end_to_end.ipynb     → reads everything; runs on the test set once
```

If you re-run `01_modeling.ipynb` with a different model under the
same `run_name`, the later notebooks pick up your new choice
automatically (they all read from `<run_name>/`).

## What each notebook does

| Notebook | What it does | Knob |
|---|---|---|
| `01_modeling.ipynb` | Train one model on dialogue features. 5-fold CV AUC + held-out evaluation. | model family (logistic / random forest / xgboost / svm), feature subset, target |
| `02_calibration.ipynb` | Wrap the model so its probability outputs are honest. ECE before vs after; reliability diagram. | calibration method (sigmoid / isotonic) |
| `03_decision.ipynb` | Apply a cost matrix. Recommend Greenlight / Pass / Refer. Compare to baselines. | flop cost, miss cost, refer cost |
| `04_explanation.ipynb` | TreeSHAP global ranking + one example film. | which film to inspect |
| `05_end_to_end.ipynb` | Run the assembled pipeline on the held-out test set. **Touch the test set once.** | none (this is the honest final number) |

## How to test different models as a team

Each teammate picks a unique `run_name` and a model family, runs
notebooks 01-04, and shares the headline numbers. Suggested
team workflow for one meeting:

| Person | run_name | model_family | Notebook 01 CV AUC | Notebook 02 ECE after | Notebook 03 system cost |
|---|---|---|---|---|---|
| Alice | `alice_xgboost` | xgboost | 0.60 | 0.00 | $1.3M |
| Bob | `bob_random_forest` | random_forest | 0.62 | 0.05 | $1.3M |
| Chloe | `chloe_logistic` | logistic | 0.59 | 0.04 | $1.3M |
| Dan | `dan_svm` | svm_rbf | 0.65 | 0.06 | $1.3M |

After comparing, the team picks one finalist. That teammate's
`run_name` goes into `05_end_to_end.ipynb` and gets evaluated on
the test set. **Only one run_name gets tested on the test set.**

## Important — the test set rule

`05_end_to_end.ipynb` is the **only** notebook allowed to read the
held-out test split. Do not load it anywhere else; do not tune
hyperparameters using test-set numbers; do not pick the "best
model" by test AUC. The test set gets touched exactly once at the
end of the project, and that number is what goes in the report.

The other four notebooks operate on the train + cal splits only.
Within those splits, evaluation uses cross-validation or the cal
holdout — both safe.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `FileNotFoundError: features.parquet` | The notebook walks up the filesystem to find the project root via `docs/PROJECT_CONTEXT.md`. Make sure you cloned the full repo and the file exists. |
| `FileNotFoundError: student_model.joblib` | Run notebook 01 first under the same `run_name` you have in the CONFIG of the current notebook. |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` (or use a different `model_family`). |
| `ModuleNotFoundError: shap` | `pip install shap` (only notebook 04 needs it). |
| Got the same numbers as a teammate | You forgot to change `run_name`. Each teammate must use a unique label. |

## Regenerating the notebooks

If you edit the build script, regenerate all five notebooks with:

```
python -m notebooks.student._build_student_notebooks
```
