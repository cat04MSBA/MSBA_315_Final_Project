"""Phase 3b embedding feature group.

Computes 32 features per film: the leading 32 principal components
of mean-pooled MiniLM sentence embeddings of dialogue text. The
pipeline is two stages.

* **Stage 1 (extract_pooled_embeddings)** runs the
  ``sentence-transformers/all-MiniLM-L6-v2`` encoder over every
  non-empty dialogue line of every film, mean-pools per film to a
  384-dim vector, and caches the resulting (1,713 × 384) matrix to
  ``data/processed/embeddings_minilm_pooled.parquet``. This is the
  expensive step; subsequent runs reuse the cache.
* **Stage 2 (fit_embedding_pca / compute_embedding_features)** fits
  PCA on the train-fold pooled embeddings only (the no-leakage
  discipline) and projects all 1,713 films to the 32-dim feature
  matrix.

The feature design follows the planning-conversation-approved v1
proposal at ``docs/proposals/phase3_embedding_proposal.md``.

No-leakage discipline (CRITICAL)
--------------------------------

* The MiniLM encoder is pre-trained and applied uniformly to all
  films; this does not leak training-fold information.
* PCA is fit on the **training-fold pooled embeddings only**
  (1,199 films from ``split_assignments.parquet``). The fitted
  components and explained-variance ratios are saved alongside the
  feature matrix so downstream phases can reload without refitting.
* The cal/test films contribute zero information to the PCA fit.

Dependencies
------------

* ``sentence-transformers`` (transitively pulls torch, transformers,
  tokenizers).
* ``scikit-learn`` PCA (already installed).
* Apple Silicon MPS or CUDA acceleration auto-detected; falls back
  to CPU if neither is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data.parse_screenplay import ParsedScreenplay
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingFeatureConfig:
    """Knobs for the embedding feature pipeline.

    Defaults reproduce proposal v1 Section 7.3.
    """

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    n_pca_components: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    batch_size: int = 64
    random_state: int = 42

    # Cache path for the per-film mean-pooled raw embeddings.
    # Encoder-name-derived suffix prevents accidental reuse across
    # encoder swaps.
    cache_path: Path = field(
        default_factory=lambda: paths.DATA_PROCESSED_DIR / "embeddings_minilm_pooled.parquet"
    )


def embedding_feature_columns(n_pca_components: int) -> tuple[str, ...]:
    """Return the ordered PCA-component column names."""
    return tuple(f"embed_pc_{i:02d}" for i in range(n_pca_components))


# Diagnostic columns are written to a separate diagnostic table; no
# leading-underscore feature columns are produced by this module.
DIAGNOSTIC_ONLY_COLUMNS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    """Resolve ``"auto"`` to the best available device.

    Apple Silicon MPS is preferred over CPU when available; CUDA is
    preferred over MPS when both are present.
    """
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Stage 1: extract pooled embeddings
# ---------------------------------------------------------------------------


def _dialogue_lines(parsed: ParsedScreenplay) -> list[str]:
    """Return non-empty dialogue lines, applying the Phase 2 Tier 1.3 filter."""
    out: list[str] = []
    for scene in parsed.scenes:
        for _char, text in scene.dialogue_units:
            cleaned = text.strip() if text else ""
            if cleaned:
                out.append(cleaned)
    return out


def extract_pooled_embeddings(
    parsed_corpus: dict[str, ParsedScreenplay],
    cfg: EmbeddingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Encode every non-empty dialogue line, mean-pool per film.

    Returns a DataFrame indexed by ``imdb_id`` with 384 columns
    (the encoder's output dimension). Cached to
    ``cfg.cache_path``; subsequent calls reuse the cache when its
    row count matches the input corpus.
    """
    cfg = cfg or EmbeddingFeatureConfig()

    if cfg.cache_path.is_file():
        cached = pd.read_parquet(cfg.cache_path)
        if len(cached) == len(parsed_corpus) and set(cached.index) == set(parsed_corpus.keys()):
            logger.info(
                "Reusing cached pooled embeddings at %s (%d films)",
                cfg.cache_path, len(cached),
            )
            return cached

    from sentence_transformers import SentenceTransformer
    device = _resolve_device(cfg.device)
    logger.info("Loading encoder %s on device=%s", cfg.encoder_name, device)
    model = SentenceTransformer(cfg.encoder_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()

    pooled: dict[str, np.ndarray] = {}
    total = len(parsed_corpus)
    for i, (imdb_id, parsed) in enumerate(parsed_corpus.items(), start=1):
        lines = _dialogue_lines(parsed)
        if not lines:
            pooled[imdb_id] = np.zeros(embed_dim, dtype=np.float32)
            continue
        embeddings = model.encode(
            lines,
            batch_size=cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        pooled[imdb_id] = embeddings.mean(axis=0).astype(np.float32)
        if i % 50 == 0 or i == total:
            logger.info("Encoded %d / %d films", i, total)

    df = pd.DataFrame.from_dict(pooled, orient="index")
    df.columns = [f"emb_{j:03d}" for j in range(embed_dim)]
    df.index.name = "imdb_id"
    df = df.astype(np.float32)
    df.to_parquet(cfg.cache_path)
    logger.info("Saved pooled embeddings to %s", cfg.cache_path)
    return df


# ---------------------------------------------------------------------------
# Stage 2: PCA fit and project
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedEmbeddingPCA:
    """Bundle of train-fold-fit PCA artifacts."""
    pca: PCA
    train_ids: tuple[str, ...]
    config: EmbeddingFeatureConfig


def fit_embedding_pca(
    pooled_embeddings: pd.DataFrame,
    train_ids: Sequence[str],
    cfg: EmbeddingFeatureConfig | None = None,
) -> FittedEmbeddingPCA:
    """Fit PCA on the training-fold pooled embeddings only.

    Parameters
    ----------
    pooled_embeddings
        DataFrame of (n_films, 384) raw mean-pooled embeddings.
    train_ids
        IMDb IDs of the training-fold films.
    cfg
        Feature configuration; ``None`` uses defaults.

    Returns
    -------
    FittedEmbeddingPCA
        The fitted PCA estimator, the training IDs, and the
        configuration.
    """
    cfg = cfg or EmbeddingFeatureConfig()
    train_ids_present = [i for i in train_ids if i in pooled_embeddings.index]
    missing = set(train_ids) - set(train_ids_present)
    if missing:
        logger.warning(
            "%d train_ids missing from pooled_embeddings: %s",
            len(missing), sorted(missing)[:5],
        )

    X_train = pooled_embeddings.loc[train_ids_present].values.astype(np.float64)
    pca = PCA(n_components=cfg.n_pca_components, random_state=cfg.random_state)
    pca.fit(X_train)
    cum_var = pca.explained_variance_ratio_.cumsum()[-1]
    logger.info(
        "Fitted PCA: K=%d, cumulative variance explained = %.3f",
        cfg.n_pca_components, cum_var,
    )
    return FittedEmbeddingPCA(
        pca=pca,
        train_ids=tuple(train_ids_present),
        config=cfg,
    )


def compute_embedding_features(
    pooled_embeddings: pd.DataFrame,
    fitted: FittedEmbeddingPCA,
) -> pd.DataFrame:
    """Project all films' pooled embeddings to the PCA-component matrix.

    Returns a DataFrame indexed by ``imdb_id`` with
    :func:`embedding_feature_columns` as columns.
    """
    cfg = fitted.config
    X = pooled_embeddings.values.astype(np.float64)
    Z = fitted.pca.transform(X)
    cols = list(embedding_feature_columns(cfg.n_pca_components))
    df = pd.DataFrame(Z, index=pooled_embeddings.index, columns=cols)
    df.index.name = "imdb_id"
    df = df.astype(np.float64)
    logger.info(
        "Embedding feature matrix complete: %d films x %d columns",
        df.shape[0], df.shape[1],
    )
    return df
