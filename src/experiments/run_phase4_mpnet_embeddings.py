"""Phase 4 Tier-A3 step: re-encode screenplays with mpnet-base.

The Phase 3 embedding group used MiniLM-L6 (384-dim, 23 MB). The
Phase 4 model benchmark hit a corpus ceiling around 0.63 OOF AUC on
``roi_gt_2`` across four very different model families. One Tier-A
hypothesis from the Phase 4 escalation discussion: a stronger
sentence encoder may extract more signal from the dialogue text. The
mpnet-base model (768-dim, 110 MB) typically adds 0.5 to 1.5 AUC
points on text-classification tasks vs MiniLM.

This script reuses the existing Phase 3 embedding pipeline with two
overrides: encoder name swapped to mpnet-base and cache path renamed
to avoid collision with the MiniLM cache. The PCA dimensionality is
held at 32 for a fair comparison; the embedding feature group's
column-naming convention (``embed_pc_NN``) is preserved so that the
existing matrix builders and model pipelines pick the new features
up without code changes when configured to do so.

Outputs:
    data/processed/embeddings_mpnet_pooled.parquet   (raw 768-dim cache)
    data/processed/embedding_pca_mpnet.joblib        (train-fitted PCA)
    data/processed/features_embedding_mpnet.parquet  (32 PCA features)

Run from project root::

    python -m src.experiments.run_phase4_mpnet_embeddings

Idempotent. Re-uses any existing cache for the encoder pass; safe to
re-run if downstream steps need to refit PCA on a different train
split.
"""

from __future__ import annotations

# Allow running this script by file path; no-op under ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd

from src.features.embedding import (
    EmbeddingFeatureConfig,
    compute_embedding_features,
    embedding_feature_columns,
    extract_pooled_embeddings,
    fit_embedding_pca,
)
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


def main() -> None:
    set_log_level("INFO")
    paths.ensure_dirs()

    cfg = EmbeddingFeatureConfig(
        encoder_name="sentence-transformers/all-mpnet-base-v2",
        n_pca_components=32,
        cache_path=paths.DATA_PROCESSED_DIR / "embeddings_mpnet_pooled.parquet",
    )

    logger.info("Loading parsed screenplays")
    with open(paths.DATA_PROCESSED_DIR / "screenplays_parsed.pkl", "rb") as f:
        parsed_corpus = pickle.load(f)
    logger.info("Loaded %d parsed screenplays", len(parsed_corpus))

    pooled = extract_pooled_embeddings(parsed_corpus, cfg)
    logger.info(
        "Pooled embeddings: shape=%s dim=%d", pooled.shape, pooled.shape[1],
    )

    splits = pd.read_parquet(paths.DATA_PROCESSED_DIR / "split_assignments.parquet")
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"].tolist()
    logger.info("Fitting PCA on %d train-split films", len(train_ids))
    fitted = fit_embedding_pca(pooled, train_ids=train_ids, cfg=cfg)
    logger.info(
        "Train-fitted PCA: %d components, cumulative variance %.4f",
        cfg.n_pca_components,
        float(fitted.pca.explained_variance_ratio_.cumsum()[-1]),
    )

    pca_path = paths.DATA_PROCESSED_DIR / "embedding_pca_mpnet.joblib"
    joblib.dump(fitted, pca_path)
    logger.info("Saved PCA artifact to %s", pca_path)

    features = compute_embedding_features(pooled, fitted)
    out = paths.DATA_PROCESSED_DIR / "features_embedding_mpnet.parquet"
    features.to_parquet(out)
    logger.info("Saved %d-column feature matrix to %s", features.shape[1], out)


if __name__ == "__main__":
    main()
