# Experiment Runs

Index of every modelling experiment in the project, sorted newest first.
Each row corresponds to a directory under `runs/<phase>/` produced by
`src.experiments.save_run`. Six files per run directory: `params.json`,
`preprocessing_summary.json`, `features_used.json`, `metrics.json`,
`run.log`, and (when present) `model.joblib`. The `model.joblib` files
are gitignored; everything else is tracked so the run is reproducible
from the recorded git SHA plus the metadata.

New rows are inserted directly above the marker comment near the top of
the table by `RunHandle.append_to_runs_md`. Manual rows can be added in
the same place; keep the newest-first ordering.

| Date | Phase | Run folder | Git SHA | Model | Features group | Key metric | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-03 16:38 | phase_3 | `runs/phase_3/20260503_1637_lexical_first/` | `21a1bb1` | Ridge + LogReg (L2) | structural + lexical | R2 0.037 ; AUC roi_gt_2 0.600 | Phase 3b ablation row 1: lexical (14 features); wordfreq backend |
| 2026-05-03 17:19 | phase_3 | `runs/phase_3/20260503_1719_lexical_multifamily/` | `21a1bb1` | 4-family (linear, histgb, knn, svm) | structural + lexical | linear R2 0.037 / AUC roi_gt_2 0.600; histgb R2 0.060 | Phase 3b row 1: lexical (14 features); 4 families for diagnostic disambiguation |
| 2026-05-03 18:13 | phase_3 | `runs/phase_3/20260503_1812_lexical_multifamily/` | `b624726` | 4-family (linear, histgb, knn, svm) | structural + lexical | linear OOF RMSE 1.350 / AUC roi_gt_2 0.600; histgb RMSE 1.333 | Phase 3b row 1: lexical (14 features); 4 families × 2 eval sets |
| 2026-05-03 19:43 | phase_3 | `runs/phase_3/20260503_1943_sentiment_multifamily/` | `54658d7` | 4-family (linear, histgb, knn, svm) | structural + sentiment | linear OOF RMSE 1.357 / AUC roi_gt_2 0.610; histgb RMSE 1.328 | Phase 3b row 2: sentiment (22 features); 4 families × 2 eval sets |
| 2026-05-03 20:16 | phase_3 | `runs/phase_3/20260503_2015_topic_multifamily/` | `54658d7` | 4-family (linear, histgb, knn, svm) | structural + topic | linear OOF RMSE 1.354 / AUC roi_gt_2 0.613; histgb RMSE 1.336 | Phase 3b row 3: topic (22 features, K=20 LDA); 4 families × 2 eval sets |
| 2026-05-03 20:26 | phase_3 | `runs/phase_3/20260503_2025_character_network_multifamily/` | `54658d7` | 4-family (linear, histgb, knn, svm) | structural + character_network | linear OOF RMSE 1.348 / AUC roi_gt_2 0.618; histgb RMSE 1.328 | Phase 3b row 4: character network (12 features); 4 families × 2 eval sets |
| 2026-05-03 20:48 | phase_3 | `runs/phase_3/20260503_2047_embedding_multifamily/` | `ff0281c` | 4-family (linear, histgb, knn, svm) | structural + embedding | linear OOF RMSE 1.331 / AUC roi_gt_2 0.607; histgb RMSE 1.318; PCA-32 cumvar 0.739 | Phase 3b row 5: embedding (32 PCA of MiniLM pooled); 4 families × 2 eval sets |
<!-- new rows above this line -->
| (none yet) | | | | | | | First lexical run will be the first entry |
