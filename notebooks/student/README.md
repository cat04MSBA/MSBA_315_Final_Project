# Student Pipeline — One Notebook, One CONFIG Cell

There is exactly one notebook to open: **`student_pipeline.ipynb`**.

It runs the entire four-layer triage system in a single file:

1. Load data
2. Train + tune a model (Layer 1)
3. Calibrate its probability outputs (Layer 2)
4. Apply the cost-asymmetric decision rule (Layer 3)
5. Produce SHAP explanations (Layer 4)
6. (Optional) evaluate on the held-out test set

Edit the CONFIG cell once. Run all cells top-to-bottom.

## What to run first

1. **Install dependencies.** From the project root:
   ```
   pip install -r requirements.txt
   ```
   The notebook needs `pandas`, `numpy`, `scikit-learn`,
   `matplotlib`, `joblib`, `xgboost`, and `shap`.

2. **Confirm the data is there.** The notebook reads
   `data/processed/features.parquet` and
   `data/processed/films_joined.parquet`. They are already built
   in this checkout. If missing, run
   `python -m src.experiments.run_phase4_benchmark` from the
   project root.

3. **Open the notebook** in Jupyter or VS Code and run every cell
   in order. Path detection is automatic — works whether you
   launch Jupyter from the project root or from
   `notebooks/student/`.

## How the team coordinates

Every teammate edits the **same notebook file** but sets a unique
`run_name` in CONFIG so artifacts don't clobber each other:

```python
CONFIG = {
    "run_name": "alice_xgboost",   # YOUR unique label
    "model_family": "xgboost",     # YOUR model choice
    ...
}
```

Each run lands in its own folder:

```
data/processed/student/
├── alice_xgboost/
│   ├── student_decisions.csv
│   ├── student_full_pipeline.joblib
│   └── student_test_predictions.csv     (only if EVALUATE_ON_TEST=True)
├── bob_random_forest/
│   └── ...
└── chloe_logistic/
    └── ...
```

## The CONFIG cell — every knob

| Knob | What it does | Default |
|---|---|---|
| `run_name` | Unique folder name for your artifacts | `"team_baseline_rf"` |
| `target` | Which outcome to predict | `"roi_gt_2"` |
| `model_family` | Which classifier to train | `"random_forest"` |
| `feature_set` | Which feature group to use | `"all"` |
| `use_grid_search` | Tune hyperparameters | `True` |
| `use_repeated_cv` | 5-fold × 3-repeat CV | `True` |
| `use_class_balance` | Class weighting / scale_pos_weight | `True` |
| `calibration_method` | Probability calibration | `"isotonic"` |
| `flop_cost` / `miss_cost` / `refer_cost` | Cost matrix | $50M / $100M / $5K |
| `shap_*` | SHAP plotting parameters | sensible defaults |
| **`EVALUATE_ON_TEST`** | **Touches the test set — leave False while iterating** | **`False`** |

## The test-set rule

`EVALUATE_ON_TEST` is False by default. **Keep it False** while
experimenting — the notebook will produce all train/cal numbers
without touching the test set.

When the team agrees on the final configuration, ONE teammate
flips it to True, runs the notebook once, and that's the headline
number for the report. Don't keep flipping it; that's not how
held-out test sets work.

## Team comparison table

After everyone runs the notebook, fill in:

| Person | run_name | model | Tuned CV AUC | Cal AUC | ECE after | System cost (cal) |
|---|---|---|---|---|---|---|
| Alice | alice_xgboost | xgboost | 0.60 | 0.57 | 0.05 | $1.3M |
| Bob | bob_random_forest | random_forest | 0.61 | 0.61 | 0.004 | $1.3M |
| Chloe | chloe_logistic | logistic | 0.58 | 0.57 | 0.04 | $1.3M |
| Dan | dan_svm | svm_rbf | 0.57 | 0.53 | 0.06 | $1.3M |

Pick a finalist; that teammate flips `EVALUATE_ON_TEST=True` and
re-runs to produce the headline test-set number for the report.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: features.parquet` | Make sure you cloned the full repo. The notebook walks up the filesystem to find `docs/PROJECT_CONTEXT.md`; that file must exist. |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` (or pick a different `model_family`). |
| `ModuleNotFoundError: shap` | `pip install shap` (the SHAP section will fail without it). |
| Got the same numbers as a teammate | You forgot to change `run_name`. |
| Want to compare with vs without an improvement | Flip one of the three toggles to False, re-run. |

## Regenerating the notebook

If you edit the build script:

```
python -m notebooks.student._build_student_notebooks
```

That writes `student_pipeline.ipynb` from the build script's cell
list.
