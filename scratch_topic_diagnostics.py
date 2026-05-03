"""One-off diagnostic script for the topic ablation Section 8 checks.

Run::

    python3 scratch_topic_diagnostics.py

Produces a console summary of the eight post-implementation
diagnostic checks listed in proposal v1 Section 8. Output is
captured into the topic handoff. This file is a one-off and can
be deleted after the handoff is written.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.features.baseline_features import (
    BaselineFeatureConfig, build_baseline_features,
)
from src.features.targets import LOG_ROI_COL, ROI_GT_1_COL, ROI_GT_2_COL, add_targets
from src.utils import paths


def main() -> None:
    top = pd.read_parquet(paths.DATA_PROCESSED_DIR / "features_topic.parquet")
    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = set(splits.loc[splits["split"] == "train", "imdb_id"])
    top_train = top.loc[top.index.isin(train_ids)]

    df = pd.read_parquet(paths.DATA_PROCESSED_DIR / "films_joined.parquet")
    df = add_targets(df)
    df_train = df[df["imdb_id"].isin(train_ids)].set_index("imdb_id")

    common = top_train.index.intersection(df_train.index)
    top_train = top_train.loc[common]
    df_train = df_train.loc[common]
    print(f"Train-split topic matrix: {top_train.shape}")

    model_cols = list(top_train.columns)

    # ---- Check 1: topic coherence (UMass on the train fold) ----
    # We approximate UMass coherence using the fitted vectorizer +
    # train-fold doc-term matrix. UMass coherence for a topic is the
    # mean log of (D(w_i, w_j) + 1) / D(w_j) over the top-N word pairs.
    print("\n=== 1. Topic coherence (UMass) ===")
    artifacts_dir = paths.DATA_PROCESSED_DIR / "topic_model_artifacts"
    vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")
    lda = joblib.load(artifacts_dir / "lda.joblib")
    train_id_arr = np.load(artifacts_dir / "train_ids.npy")
    # Reconstruct the train documents to get the doc-term matrix.
    import pickle
    with (paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl").open("rb") as f:
        parsed_corpus = pickle.load(f)
    from src.features.topic import _film_documents, TopicFeatureConfig
    cfg = TopicFeatureConfig()
    docs, _ = _film_documents(parsed_corpus, list(train_id_arr), cfg)
    X = vectorizer.transform(docs)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_n_words = 10
    eps = 1.0
    coherences = []
    for topic_id, topic_distribution in enumerate(lda.components_):
        top_idx = np.argsort(topic_distribution)[::-1][:top_n_words]
        top_words = feature_names[top_idx]
        # Compute document-frequency of each top word and pairwise
        # co-occurrence in the doc-term matrix.
        doc_freq = np.array((X[:, top_idx] > 0).sum(axis=0)).ravel().astype(float)
        # Pairwise co-occurrence count.
        binary = (X[:, top_idx] > 0).astype(int)
        cooccur = (binary.T @ binary).toarray() if hasattr(binary, "toarray") else (binary.T @ binary)
        coh = 0.0
        n_pairs = 0
        for i in range(top_n_words):
            for j in range(i + 1, top_n_words):
                num = cooccur[j, i] + eps
                denom = doc_freq[i] if doc_freq[i] > 0 else eps
                coh += np.log(num / denom)
                n_pairs += 1
        coh = coh / n_pairs if n_pairs > 0 else float("nan")
        coherences.append(coh)
    print(f"Mean UMass coherence across 20 topics: {np.mean(coherences):.3f}")
    print(f"Median: {np.median(coherences):.3f}")
    print(f"Min:    {np.min(coherences):.3f}")
    print(f"Max:    {np.max(coherences):.3f}")
    print("(UMass is unbounded above and at most 0; values closer to 0 are more coherent.)")

    # ---- Check 2: top-10 words per topic (saved separately) ----
    print("\n=== 2. Top-10 words per topic (first 5 topics; full table at reports/tables/phase3_topic_labels.csv) ===")
    label_table = pd.read_csv(paths.REPORTS_TABLES_DIR / "phase3_topic_labels.csv")
    for _, row in label_table.head(5).iterrows():
        print(f"  topic_{int(row['topic_id']):02d}: {row['top_words']}")

    # ---- Check 3: cross-correlations with structural baseline ----
    print("\n=== 3. Topic ↔ structural baseline cross-correlations ===")
    base_cfg = BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
    )
    base = build_baseline_features(df_train.reset_index(), base_cfg).loc[common]
    base_numeric = [c for c in base.columns if not c.startswith("genre_")]
    pairs = []
    for tc in model_cols:
        for bc in base_numeric:
            r = top_train[tc].corr(base[bc])
            if pd.notna(r):
                pairs.append((abs(r), tc, bc, round(r, 3)))
    pairs.sort(reverse=True)
    above = [p for p in pairs if p[0] > 0.85]
    if above:
        print(f"Pairs with |r| > 0.85: {len(above)}")
        for _, tc, bc, r in above:
            print(f"  {tc} ↔ {bc}: r = {r}")
    else:
        print("All pairs |r| < 0.85; top 5 absolute correlations:")
        for _, tc, bc, r in pairs[:5]:
            print(f"  {tc} ↔ {bc}: r = {r}")

    # ---- Check 4: cross-correlations with genre dummies ----
    print("\n=== 4. Topic ↔ genre-dummy cross-correlations (genre-residual diagnostic) ===")
    genre_cols = [c for c in base.columns if c.startswith("genre_")]
    g_pairs = []
    for tc in model_cols:
        for gc in genre_cols:
            r = top_train[tc].corr(base[gc])
            if pd.notna(r):
                g_pairs.append((abs(r), tc, gc, round(r, 3)))
    g_pairs.sort(reverse=True)
    print("Top 10 absolute topic-genre correlations:")
    for _, tc, gc, r in g_pairs[:10]:
        print(f"  {tc} ↔ {gc}: r = {r}")

    # ---- Check 5: data_quality_flag films vs unflagged ----
    print("\n=== 5. data_quality_flag films vs unflagged ===")
    flagged = df_train["data_quality_flag"].astype(bool)
    n_flag = int(flagged.sum())
    print(f"Flagged films in train: {n_flag}")
    if n_flag > 0:
        print(f"{'feature':>32} | flagged μ  unflagged μ  z_diff")
        for c in model_cols[:6]:  # spot-check first 6
            mu_flag = top_train.loc[flagged.values, c].mean()
            mu_unfl = top_train.loc[~flagged.values, c].mean()
            sd_unfl = top_train.loc[~flagged.values, c].std(ddof=0)
            z_diff = (mu_flag - mu_unfl) / sd_unfl if sd_unfl > 0 else 0.0
            print(f"  {c:>30} | {mu_flag:+.4f}   {mu_unfl:+.4f}   {z_diff:+.2f}")

    # ---- Check 6: topic-distribution dominance ----
    print("\n=== 6. Topic-distribution dominance ===")
    proportion_cols = [c for c in model_cols if c.endswith("_proportion")]
    max_topic_share = top_train[proportion_cols].max(axis=1)
    print(f"Mean top-topic share: {max_topic_share.mean():.3f}")
    print(f"Median:               {max_topic_share.median():.3f}")
    print(f"Films with top-topic share > 0.5: "
          f"{int((max_topic_share > 0.5).sum())} of {len(max_topic_share)}")

    # ---- Check 7: univariate target correlations ----
    print("\n=== 7. Univariate target correlations (|r| > 0.10 reported) ===")
    targets = {
        "log_roi": df_train[LOG_ROI_COL].astype(float),
        "roi_gt_1": df_train[ROI_GT_1_COL].astype(float),
        "roi_gt_2": df_train[ROI_GT_2_COL].astype(float),
    }
    found = False
    for tname, t in targets.items():
        for c in model_cols:
            r = top_train[c].corr(t)
            if pd.notna(r) and abs(r) > 0.10:
                print(f"  {c} ↔ {tname}: r = {r:+.3f}")
                found = True
    if not found:
        all_pairs = []
        for tname, t in targets.items():
            for c in model_cols:
                r = top_train[c].corr(t)
                if pd.notna(r):
                    all_pairs.append((abs(r), c, tname, r))
        all_pairs.sort(reverse=True)
        print("No feature exceeds |r| = 0.10 against any target.")
        print("Top 5 absolute correlations:")
        for _, c, tname, r in all_pairs[:5]:
            print(f"  {c} ↔ {tname}: r = {r:+.3f}")


if __name__ == "__main__":
    main()
