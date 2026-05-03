"""Phase 3b topic feature group.

Computes 22 topic features per film via Latent Dirichlet Allocation
(Blei, Ng, and Jordan 2003) over per-film concatenated dialogue
text. The 22 features split into three sub-blocks:

* 20 topic proportions (one per LDA topic; the per-film posterior
  topic distribution).
* 1 topic-distribution concentration measure: the entropy of the
  topic distribution rescaled to ``[0, 1]``.
* 1 dominant-topic identifier: the index of the topic with the
  highest proportion.

The feature design follows the planning-conversation-approved v1
proposal at ``docs/proposals/phase3_topic_proposal.md``. See that
document for per-feature rationale and pre-registered lift bands.

No-leakage discipline (CRITICAL)
--------------------------------

LDA is the first Phase 3b feature group whose feature computation
depends on the data distribution. Per ``PROJECT_CONTEXT.md`` Section
6 and the proposal Section 6.2, the discipline is:

* :class:`CountVectorizer` and
  :class:`LatentDirichletAllocation` are fit on **training-fold
  tokens only**.
* The fitted vocabulary, document-term matrix, and LDA model are
  saved to ``data/processed/topic_model_artifacts/`` for downstream
  inference.
* The 1,713-film topic-distribution matrix is computed by applying
  ``transform`` on every film's tokens, using the train-fitted
  vocabulary and topic-word distributions.
* Cal and test films contribute zero information to the fitted
  vocabulary or topic-word distributions.

Dependencies
------------

* ``scikit-learn`` for ``CountVectorizer`` and
  ``LatentDirichletAllocation`` (already a project dependency).
* ``nltk`` for the English stopword list (already a project
  dependency).

References
----------

* Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). "Latent
  Dirichlet Allocation." *Journal of Machine Learning Research*, 3,
  993-1022.
* Eliashberg, J., Hui, S. K., and Zhang, Z. J. (2014). Used K = 19
  topics on a screenplay corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.data.parse_screenplay import ParsedScreenplay
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicFeatureConfig:
    """Knobs for the topic feature pipeline.

    Defaults reproduce proposal v1 Section 7.3.
    """

    n_topics: int = 20
    min_df: int = 5
    max_df: float = 0.5
    n_lda_iterations: int = 10
    learning_method: str = "batch"
    random_state: int = 42
    remove_stopwords: bool = True
    min_token_length: int = 3
    # Optional override for the stopword list. None means NLTK English.
    stopword_set: frozenset[str] | None = None


# ---------------------------------------------------------------------------
# Token-stream construction
# ---------------------------------------------------------------------------


_WORD_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def _english_stopwords() -> frozenset[str]:
    """Return the cached NLTK English stopword set.

    Imports the resource lazily so callers that override the stopword
    set via ``TopicFeatureConfig.stopword_set`` do not pay the
    download / import cost.
    """
    from nltk.corpus import stopwords as _stopwords
    return frozenset(_stopwords.words("english"))


def _ensure_nltk_resources() -> None:
    """Download stopwords + punkt if missing."""
    import nltk
    for category, resource in (
        ("corpora", "stopwords"),
        ("tokenizers", "punkt"),
        ("tokenizers", "punkt_tab"),
    ):
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def _dialogue_tokens(
    parsed: ParsedScreenplay,
    cfg: TopicFeatureConfig,
    stopwords: frozenset[str],
) -> list[str]:
    """Return the per-film whole-screenplay token stream for LDA input.

    Concatenates non-empty dialogue lines, lowercases, drops tokens
    shorter than ``cfg.min_token_length``, drops tokens that are not
    purely alphabetic, and (if configured) drops English stopwords.
    The empty-text filter (Phase 2 Tier 1.3) is applied defensively
    before tokenization.
    """
    import nltk
    pieces: list[str] = []
    for scene in parsed.scenes:
        for _char, text in scene.dialogue_units:
            cleaned = text.strip() if text else ""
            if cleaned:
                pieces.append(cleaned)
    if not pieces:
        return []
    raw = nltk.word_tokenize(" ".join(pieces))
    out: list[str] = []
    for token in raw:
        if not _WORD_TOKEN_RE.match(token):
            continue
        token = token.lower()
        if len(token) < cfg.min_token_length:
            continue
        if cfg.remove_stopwords and token in stopwords:
            continue
        out.append(token)
    return out


def _film_documents(
    parsed_corpus: dict[str, ParsedScreenplay],
    ids: Sequence[str],
    cfg: TopicFeatureConfig,
) -> tuple[list[str], list[str]]:
    """Build (joined-text-per-film, ordered-imdb-ids) for the given IDs.

    The joined text per film is the space-joined token stream after
    filtering. ``CountVectorizer`` will re-split on whitespace, so
    pre-tokenizing here keeps the filter centralized.
    """
    if cfg.stopword_set is not None:
        stopwords = cfg.stopword_set
    else:
        _ensure_nltk_resources()
        stopwords = _english_stopwords()

    docs: list[str] = []
    out_ids: list[str] = []
    for imdb_id in ids:
        if imdb_id not in parsed_corpus:
            continue
        tokens = _dialogue_tokens(parsed_corpus[imdb_id], cfg, stopwords)
        docs.append(" ".join(tokens))
        out_ids.append(imdb_id)
    return docs, out_ids


# ---------------------------------------------------------------------------
# Fitted-model bundle and feature column names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedTopicModel:
    """Bundle of train-fold-fit artifacts that drive feature extraction.

    The vectorizer holds the train-fold vocabulary; the LDA holds the
    train-fold topic-word distributions. Both are reused unchanged
    when computing features for cal and test films.
    """
    vectorizer: CountVectorizer
    lda: LatentDirichletAllocation
    train_ids: tuple[str, ...]
    config: TopicFeatureConfig


def _topic_proportion_columns(n_topics: int) -> tuple[str, ...]:
    """Return the ordered topic-proportion column names."""
    return tuple(f"topic_{i:02d}_proportion" for i in range(n_topics))


def topic_feature_columns(n_topics: int) -> tuple[str, ...]:
    """Return all model-feature column names in canonical order."""
    return (
        *_topic_proportion_columns(n_topics),
        "topic_concentration_entropy",
        "topic_dominant_id",
    )


# Diagnostic columns are written to a separate table; no leading-
# underscore feature columns are produced by this module. The trainer
# treats every column from ``compute_topic_features`` as model input.
DIAGNOSTIC_ONLY_COLUMNS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Fit and transform
# ---------------------------------------------------------------------------


def fit_topic_model(
    parsed_corpus: dict[str, ParsedScreenplay],
    train_ids: Sequence[str],
    cfg: TopicFeatureConfig | None = None,
) -> FittedTopicModel:
    """Fit the vectorizer and LDA on training-fold dialogue tokens.

    Parameters
    ----------
    parsed_corpus
        Full mapping of ``imdb_id`` to :class:`ParsedScreenplay`. Only
        the entries listed in ``train_ids`` are used for fitting.
    train_ids
        IMDb IDs of the training-fold films. Must all be present in
        ``parsed_corpus``.
    cfg
        Feature configuration; ``None`` uses defaults.

    Returns
    -------
    FittedTopicModel
        The vectorizer, the LDA estimator, the training IDs, and the
        configuration. Pass to :func:`compute_topic_features` to
        produce the per-film feature matrix.
    """
    cfg = cfg or TopicFeatureConfig()
    docs, used_ids = _film_documents(parsed_corpus, train_ids, cfg)
    if len(used_ids) < len(train_ids):
        missing = set(train_ids) - set(used_ids)
        raise ValueError(
            f"{len(missing)} train_ids missing from parsed_corpus: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if not docs:
        raise ValueError("Empty training corpus; cannot fit LDA.")

    vectorizer = CountVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        token_pattern=r"(?u)\b\w\w\w+\b",
    )
    X_train = vectorizer.fit_transform(docs)
    logger.info(
        "Vectorized %d training documents; vocab size %d (min_df=%d, max_df=%.2f)",
        len(docs), len(vectorizer.vocabulary_), cfg.min_df, cfg.max_df,
    )

    lda = LatentDirichletAllocation(
        n_components=cfg.n_topics,
        max_iter=cfg.n_lda_iterations,
        learning_method=cfg.learning_method,
        random_state=cfg.random_state,
    )
    lda.fit(X_train)
    logger.info(
        "Fitted LDA: K=%d, max_iter=%d, learning_method=%s, perplexity (train) %.2f",
        cfg.n_topics, cfg.n_lda_iterations, cfg.learning_method,
        lda.perplexity(X_train),
    )

    return FittedTopicModel(
        vectorizer=vectorizer,
        lda=lda,
        train_ids=tuple(used_ids),
        config=cfg,
    )


def _topic_proportions_for_documents(
    docs: Sequence[str], fitted: FittedTopicModel,
) -> np.ndarray:
    """Return an (n_films, n_topics) array of topic proportions."""
    X = fitted.vectorizer.transform(docs)
    return fitted.lda.transform(X)


def compute_topic_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    fitted: FittedTopicModel,
) -> pd.DataFrame:
    """Apply the fitted model to every film and return the feature matrix.

    Parameters
    ----------
    parsed_corpus
        Mapping of ``imdb_id`` to :class:`ParsedScreenplay`. Every
        film in this mapping receives a row in the output.
    fitted
        The :class:`FittedTopicModel` returned by
        :func:`fit_topic_model`.

    Returns
    -------
    pd.DataFrame
        One row per film, indexed by ``imdb_id``, with columns
        matching :func:`topic_feature_columns` (22 features for the
        default K=20).
    """
    cfg = fitted.config
    ids = list(parsed_corpus.keys())
    docs, used_ids = _film_documents(parsed_corpus, ids, cfg)
    if not docs:
        raise ValueError("Empty corpus passed to compute_topic_features")

    proportions = _topic_proportions_for_documents(docs, fitted)
    # Topic-concentration entropy: H = -sum(p log p), normalized to [0, 1]
    # by dividing by log(K). Films whose dominant topic captures all
    # mass produce H ≈ 0; uniform distributions produce H = 1.
    eps = 1e-12
    entropy = -np.sum(
        proportions * np.log(proportions + eps), axis=1,
    ) / np.log(cfg.n_topics)
    dominant = np.argmax(proportions, axis=1).astype(float)

    cols = topic_feature_columns(cfg.n_topics)
    data: dict[str, np.ndarray] = {
        col: proportions[:, i] for i, col in enumerate(_topic_proportion_columns(cfg.n_topics))
    }
    data["topic_concentration_entropy"] = entropy
    data["topic_dominant_id"] = dominant

    df = pd.DataFrame(data, index=pd.Index(used_ids, name="imdb_id"))
    df = df[list(cols)].astype(float)
    logger.info(
        "Topic feature matrix complete: %d films x %d columns",
        df.shape[0], df.shape[1],
    )
    return df


def topic_label_table(
    fitted: FittedTopicModel, top_n_words: int = 10,
) -> pd.DataFrame:
    """Return a per-topic table of the ``top_n_words`` most representative words.

    Saved to ``reports/tables/phase3_topic_labels.csv`` by the runner;
    the columns are ``topic_id`` and ``top_words``. A
    ``human_label`` column is appended (empty by default) for
    manual annotation in the report.
    """
    feature_names = np.array(fitted.vectorizer.get_feature_names_out())
    rows: list[dict] = []
    for topic_id, topic_distribution in enumerate(fitted.lda.components_):
        top_idx = np.argsort(topic_distribution)[::-1][:top_n_words]
        top_words = [str(w) for w in feature_names[top_idx]]
        rows.append({
            "topic_id": topic_id,
            "top_words": ", ".join(top_words),
            "human_label": "",
        })
    return pd.DataFrame(rows)
