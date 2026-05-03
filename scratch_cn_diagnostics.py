"""Diagnostic checks for the character-network ablation (proposal v1 Section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.baseline_features import (
    BaselineFeatureConfig, build_baseline_features,
)
from src.features.character_network import (
    CHARACTER_NETWORK_FEATURE_COLUMNS, DIAGNOSTIC_ONLY_COLUMNS,
)
from src.features.targets import LOG_ROI_COL, ROI_GT_1_COL, ROI_GT_2_COL, add_targets
from src.utils import paths


def main() -> None:
    cn = pd.read_parquet(paths.DATA_PROCESSED_DIR / "features_character_network.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = set(splits.loc[splits["split"] == "train", "imdb_id"])
    cn_train = cn.loc[cn.index.isin(train_ids)]

    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    df = add_targets(df)
    df_train = df[df["imdb_id"].isin(train_ids)].set_index("imdb_id")
    common = cn_train.index.intersection(df_train.index)
    cn_train = cn_train.loc[common]
    df_train = df_train.loc[common]
    print(f"Train-split character-network matrix: {cn_train.shape}")

    model_cols = [c for c in cn_train.columns if c not in DIAGNOSTIC_ONLY_COLUMNS]

    print("\n=== 1. Significant-character count distribution ===")
    n_sig = cn_train["network_n_significant_characters"]
    print(f"Median: {n_sig.median():.1f}, mean: {n_sig.mean():.1f}, "
          f"p25: {n_sig.quantile(0.25):.1f}, p75: {n_sig.quantile(0.75):.1f}")

    print("\n=== 2/3. Cross-correlations with structural baseline + within-CN ===")
    base_cfg = BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
    )
    base = build_baseline_features(df_train.reset_index(), base_cfg).loc[common]
    base_numeric = [c for c in base.columns if not c.startswith("genre_")]
    pairs = []
    for c in model_cols:
        for bc in base_numeric:
            r = cn_train[c].corr(base[bc])
            if pd.notna(r):
                pairs.append((abs(r), c, bc, round(r, 3)))
    pairs.sort(reverse=True)
    above = [p for p in pairs if p[0] > 0.85]
    print(f"Pairs |r| > 0.85 with structural baseline: {len(above)}")
    for _, c, bc, r in above:
        print(f"  {c} ↔ {bc}: r = {r}")
    if not above:
        print("Top 5 absolute correlations with structural baseline:")
        for _, c, bc, r in pairs[:5]:
            print(f"  {c} ↔ {bc}: r = {r}")

    cn_pairs = []
    cols = list(model_cols)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            r = cn_train[c1].corr(cn_train[c2])
            if pd.notna(r):
                cn_pairs.append((abs(r), c1, c2, round(r, 3)))
    cn_pairs.sort(reverse=True)
    cn_above = [p for p in cn_pairs if p[0] > 0.85]
    print(f"\nWithin-CN pairs |r| > 0.85: {len(cn_above)}")
    for _, c1, c2, r in cn_above:
        print(f"  {c1} ↔ {c2}: r = {r}")
    if not cn_above:
        print("Top 5 within-CN absolute correlations:")
        for _, c1, c2, r in cn_pairs[:5]:
            print(f"  {c1} ↔ {c2}: r = {r}")

    print("\n=== 5. data_quality_flag films vs unflagged ===")
    flagged = df_train["data_quality_flag"].astype(bool)
    n_flag_train = int(flagged.sum())
    print(f"Flagged films in train: {n_flag_train}")
    nans_on_flagged = cn_train.loc[flagged.values, model_cols].isna().all(axis=1).sum()
    print(f"Of those, {int(nans_on_flagged)} are all-NaN on the 12 model columns "
          f"(should be {n_flag_train} — every flagged film should be NaN by construction).")

    print("\n=== 6. Empty-graph rate ===")
    n_under_2 = int((cn_train["network_n_significant_characters"] < 2).sum())
    print(f"Films with fewer than 2 significant characters: {n_under_2} of {len(cn_train)}")

    print("\n=== 7. Univariate target correlations (|r| > 0.10 reported) ===")
    targets = {
        "log_roi": df_train[LOG_ROI_COL].astype(float),
        "roi_gt_1": df_train[ROI_GT_1_COL].astype(float),
        "roi_gt_2": df_train[ROI_GT_2_COL].astype(float),
    }
    found = False
    for tname, t in targets.items():
        for c in model_cols:
            r = cn_train[c].corr(t)
            if pd.notna(r) and abs(r) > 0.10:
                print(f"  {c} ↔ {tname}: r = {r:+.3f}")
                found = True
    if not found:
        all_pairs = []
        for tname, t in targets.items():
            for c in model_cols:
                r = cn_train[c].corr(t)
                if pd.notna(r):
                    all_pairs.append((abs(r), c, tname, r))
        all_pairs.sort(reverse=True)
        print("No feature exceeds |r| = 0.10 against any target.")
        print("Top 5 absolute correlations:")
        for _, c, tname, r in all_pairs[:5]:
            print(f"  {c} ↔ {tname}: r = {r:+.3f}")


if __name__ == "__main__":
    main()
