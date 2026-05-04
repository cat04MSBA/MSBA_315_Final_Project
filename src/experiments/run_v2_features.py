"""Extract Phase 3 features on the v2 corpus.

Mirrors the per-group Phase 3 ablation runs but writes outputs to
``data/processed/v2/`` paths so v1 artifacts stay untouched. For
embeddings we reuse the v1 caches as much as possible — only the new
~373 v2 survivor films need encoder passes.

Steps:

1. Lexical / Sentiment / Character-network: per-film deterministic
   functions, recomputed on the full v2 corpus (cheap).
2. Topic (LDA): re-fit on v2 train fold, transform all v2 films.
3. Embedding (MiniLM and mpnet): merge v1's pooled-embedding caches
   with newly-encoded rows for the v2 survivors only, refit PCA on
   v2 train fold, transform all v2 films.
4. Consolidated ``features_v2.parquet``: union of all groups + targets
   + split-assignment.

Run from project root::

    python -m src.experiments.run_v2_features

Idempotent: per-group parquets are skipped if they exist.
"""

from __future__ import annotations

# Allow running by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.parse_screenplay import ParsedScreenplay
from src.features.baseline_features import (
    BaselineFeatureConfig,
    build_baseline_features,
)
from src.features.character_network import (
    CharacterNetworkConfig,
    compute_character_network_features,
)
from src.features.embedding import (
    EmbeddingFeatureConfig,
    compute_embedding_features,
    extract_pooled_embeddings,
    fit_embedding_pca,
)
from src.features.lexical import compute_lexical_features
from src.features.sentiment import compute_sentiment_features
from src.features.targets import add_targets
from src.features.topic import (
    TopicFeatureConfig,
    compute_topic_features,
    fit_topic_model,
)
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

V2_DIR = paths.DATA_PROCESSED_DIR / "v2"


def _load_v2_inputs() -> tuple[pd.DataFrame, dict[str, ParsedScreenplay], pd.DataFrame]:
    """Load the v2 master films table, parses, and split assignments."""
    df = pd.read_parquet(V2_DIR / "films_joined_v2.parquet")
    with (V2_DIR / "screenplays_parsed_v2.pkl").open("rb") as f:
        parsed = pickle.load(f)
    splits = pd.read_parquet(V2_DIR / "split_assignments_v2.parquet")
    return df, parsed, splits


def _maybe_skip(path: Path, label: str) -> bool:
    if path.is_file():
        logger.info("%s already exists at %s — skipping", label, path)
        return True
    return False


def _extract_per_film_groups(parsed: dict, films: pd.DataFrame) -> None:
    """Lexical, sentiment, character-network — re-extract on full v2 corpus."""
    out_lex = V2_DIR / "features_lexical_v2.parquet"
    if not _maybe_skip(out_lex, "lexical"):
        df = compute_lexical_features(parsed)
        df.to_parquet(out_lex)
        logger.info("Wrote lexical_v2: %d rows × %d cols", *df.shape)

    out_sent = V2_DIR / "features_sentiment_v2.parquet"
    if not _maybe_skip(out_sent, "sentiment"):
        df = compute_sentiment_features(parsed)
        df.to_parquet(out_sent)
        logger.info("Wrote sentiment_v2: %d rows × %d cols", *df.shape)

    out_cn = V2_DIR / "features_character_network_v2.parquet"
    if not _maybe_skip(out_cn, "character_network"):
        flags = films.set_index("imdb_id")["data_quality_flag"]
        df = compute_character_network_features(parsed, flags, CharacterNetworkConfig())
        df.to_parquet(out_cn)
        logger.info("Wrote character_network_v2: %d rows × %d cols", *df.shape)


def _extract_topic(parsed: dict, splits: pd.DataFrame) -> None:
    out = V2_DIR / "features_topic_v2.parquet"
    if _maybe_skip(out, "topic"):
        return
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"].tolist()
    cfg = TopicFeatureConfig()
    fitted = fit_topic_model(parsed, train_ids, cfg)
    df = compute_topic_features(parsed, fitted)
    df.to_parquet(out)
    logger.info("Wrote topic_v2: %d rows × %d cols", *df.shape)
    # Persist artifacts so the run is reproducible without re-fitting.
    art_dir = V2_DIR / "topic_model_artifacts_v2"
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted.vectorizer, art_dir / "vectorizer.joblib")
    joblib.dump(fitted.lda, art_dir / "lda.joblib")
    pd.Series(list(fitted.train_ids), name="imdb_id").to_csv(
        art_dir / "train_ids.csv", index=False
    )


def _build_pooled_cache(
    parsed: dict[str, ParsedScreenplay],
    v1_cache_path: Path,
    v2_cache_path: Path,
    encoder_cfg: EmbeddingFeatureConfig,
) -> pd.DataFrame:
    """Build a v2 pooled-embedding cache by reusing v1 + encoding new films.

    The encoder cost is only paid for IDs that aren't in the v1 cache.
    """
    if v2_cache_path.is_file():
        cached = pd.read_parquet(v2_cache_path)
        if set(cached.index) == set(parsed.keys()):
            logger.info(
                "Reusing v2 pooled-embedding cache at %s (%d films)",
                v2_cache_path, len(cached),
            )
            return cached

    if v1_cache_path.is_file():
        v1_cache = pd.read_parquet(v1_cache_path)
        overlap_ids = sorted(set(parsed.keys()) & set(v1_cache.index))
        new_ids = sorted(set(parsed.keys()) - set(v1_cache.index))
        logger.info(
            "Pooled cache: %d v1-overlap (reuse) + %d new (encode)",
            len(overlap_ids), len(new_ids),
        )
    else:
        logger.warning("No v1 pooled cache at %s — encoding entire v2 corpus", v1_cache_path)
        v1_cache = None
        overlap_ids = []
        new_ids = sorted(parsed.keys())

    # Encode only the new films, but use a temporary cache path to keep
    # the existing v1 cache untouched.
    if new_ids:
        new_parsed = {i: parsed[i] for i in new_ids}
        # Use a v2 sub-cache so extract_pooled_embeddings doesn't bypass.
        sub_cache = V2_DIR / f"_partial_{v2_cache_path.stem}.parquet"
        sub_cfg = replace(encoder_cfg, cache_path=sub_cache)
        new_emb = extract_pooled_embeddings(new_parsed, sub_cfg)
    else:
        new_emb = pd.DataFrame()

    pieces = []
    if v1_cache is not None and overlap_ids:
        pieces.append(v1_cache.loc[overlap_ids])
    if not new_emb.empty:
        pieces.append(new_emb)
    combined = pd.concat(pieces) if pieces else pd.DataFrame()
    combined.index.name = "imdb_id"
    combined.to_parquet(v2_cache_path)
    logger.info("Wrote v2 pooled cache to %s (%d rows × %d cols)",
                v2_cache_path, *combined.shape)
    return combined


def _extract_embedding(
    parsed: dict[str, ParsedScreenplay],
    splits: pd.DataFrame,
    encoder_name: str,
    v1_cache: Path,
    v2_cache: Path,
    pca_path: Path,
    out_path: Path,
) -> None:
    if _maybe_skip(out_path, f"embedding ({encoder_name})"):
        return
    cfg = EmbeddingFeatureConfig(
        encoder_name=encoder_name,
        cache_path=v2_cache,
    )
    pooled = _build_pooled_cache(parsed, v1_cache, v2_cache, cfg)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"].tolist()
    fitted = fit_embedding_pca(pooled, train_ids, cfg)
    df = compute_embedding_features(pooled, fitted)
    df.to_parquet(out_path)
    joblib.dump(fitted.pca, pca_path)
    logger.info("Wrote %s features to %s", encoder_name, out_path)


def _build_consolidated(
    df: pd.DataFrame, splits: pd.DataFrame, encoder_tag: str
) -> Path:
    """Build features_v2.parquet (or features_v2_mpnet.parquet) with targets + split."""
    embedding_path = (
        V2_DIR / "features_embedding_mpnet_v2.parquet"
        if encoder_tag == "mpnet"
        else V2_DIR / "features_embedding_v2.parquet"
    )
    cfg = BaselineFeatureConfig(
        log_transform_structural=True,
        include_log_runtime=True,
        include_lexical=True,
        lexical_features_path=V2_DIR / "features_lexical_v2.parquet",
        include_sentiment=True,
        sentiment_features_path=V2_DIR / "features_sentiment_v2.parquet",
        include_topic=True,
        topic_features_path=V2_DIR / "features_topic_v2.parquet",
        include_character_network=True,
        character_network_features_path=V2_DIR / "features_character_network_v2.parquet",
        include_embedding=True,
        embedding_features_path=embedding_path,
    )
    feat = build_baseline_features(df, cfg)
    targets = add_targets(df).set_index("imdb_id")[["log_roi", "roi_gt_1", "roi_gt_2"]]
    split_col = splits.set_index("imdb_id")["split"]
    consolidated = feat.join(targets).join(split_col)
    out = V2_DIR / (
        "features_v2_mpnet.parquet" if encoder_tag == "mpnet" else "features_v2.parquet"
    )
    consolidated.to_parquet(out)
    logger.info("Consolidated %s: %d × %d → %s", encoder_tag, *consolidated.shape, out)
    return out


def main() -> None:
    paths.ensure_dirs()
    df, parsed, splits = _load_v2_inputs()
    logger.info("v2 inputs: %d films, %d parses, %d splits",
                len(df), len(parsed), len(splits))

    _extract_per_film_groups(parsed, df)
    _extract_topic(parsed, splits)

    _extract_embedding(
        parsed, splits,
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        v1_cache=paths.DATA_PROCESSED_DIR / "embeddings_minilm_pooled.parquet",
        v2_cache=V2_DIR / "embeddings_minilm_pooled_v2.parquet",
        pca_path=V2_DIR / "embedding_pca_v2.joblib",
        out_path=V2_DIR / "features_embedding_v2.parquet",
    )

    _extract_embedding(
        parsed, splits,
        encoder_name="sentence-transformers/all-mpnet-base-v2",
        v1_cache=paths.DATA_PROCESSED_DIR / "embeddings_mpnet_pooled.parquet",
        v2_cache=V2_DIR / "embeddings_mpnet_pooled_v2.parquet",
        pca_path=V2_DIR / "embedding_pca_mpnet_v2.joblib",
        out_path=V2_DIR / "features_embedding_mpnet_v2.parquet",
    )

    # Two consolidated tables: one with MiniLM embeddings, one with mpnet.
    _build_consolidated(df, splits, encoder_tag="minilm")
    _build_consolidated(df, splits, encoder_tag="mpnet")

    print("\n=== v2 features build done ===")
    for p in sorted(V2_DIR.glob("features_*.parquet")):
        try:
            d = pd.read_parquet(p)
            print(f"  {p.name}: {d.shape[0]} rows × {d.shape[1]} cols")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name}: ERR {e}")


if __name__ == "__main__":
    main()
