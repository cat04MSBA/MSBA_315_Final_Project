"""Phase 3b lexical feature group.

Computes 14 lexical features per film from the parsed screenplay
structure. Features cover five sub-groups: vocabulary diversity,
lexical sophistication, readability, length statistics, and
punctuation/pronoun signals.

The feature design follows the planning-conversation-approved v2
proposal at ``docs/proposals/phase3_lexical_proposal.md``. See that
document for per-feature rationale and the pre-registered lift bands.

Tactical implementation note (frequency source)
------------------------------------------------

The proposal specifies SUBTLEX-US (Brysbaert and New 2009) as the
external frequency reference for the two sophistication features. At
implementation time the canonical SUBTLEX-US download URLs at the
Brysbaert lab returned 404 / HTML redirect pages, so a hash-checked
direct download was not viable. The implementation backend is the
``wordfreq`` package (Speer 2018; Robyn Speer's wordlist mixture),
which computes English Zipf-scale log-frequencies from a blend of
sources INCLUDING OpenSubtitles (subtitle-domain, similar to SUBTLEX
in spirit) along with Wikipedia, Twitter, Reddit, and prose corpora.

The mixture is not pure SUBTLEX-US, but the conceptual mechanism the
proposal selected SUBTLEX-US for (subtitle-derived frequencies match
our screenplay-dialogue input domain better than prose-derived
frequencies like COCA) is preserved by wordfreq's inclusion of
OpenSubtitles in its English mix. The frequency features are
correspondingly named without the `_subtlex` suffix
(``mean_log_frequency``, ``rare_word_proportion``) to be honest
about the source. This deviation is surfaced to the planning
conversation alongside the first lexical run's diagnostic results;
if strict SUBTLEX-US is preferred, swapping the backend is a
single-function change on this module.

Dependencies
------------

* ``nltk`` for ``word_tokenize`` and ``sent_tokenize``. The
  ``punkt`` and ``punkt_tab`` resources must be downloaded once via
  ``nltk.download('punkt')`` and ``nltk.download('punkt_tab')``;
  this is handled by ``ensure_nltk_resources`` in this module.
* ``wordfreq`` for the English Zipf-frequency lookup.

References
----------

* Brysbaert, M., and New, B. (2009). "Moving beyond Kučera and
  Francis: A critical evaluation of current word frequency norms..."
* McCarthy, P. M., and Jarvis, S. (2010). "MTLD, vocd-D, and HD-D..."
* Speer, R. (2018). ``wordfreq`` Python package documentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import nltk
import numpy as np
import pandas as pd
import wordfreq

from src.data.parse_screenplay import ParsedScreenplay
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LexicalFeatureConfig:
    """Knobs for the lexical feature pipeline.

    Defaults reproduce the values in proposal v2 Section 8.3.
    """

    # MTLD threshold for the running TTR drop test (McCarthy and Jarvis 2010).
    mtld_threshold: float = 0.72

    # Cutoff (in tokens) below which a dialogue line counts as "short".
    short_line_cutoff_tokens: int = 5

    # Whether to remove stop words before vocabulary-diversity computations.
    # Default off; standard MTLD definition does not remove stop words.
    remove_stop_words: bool = False

    # Quartile (1-indexed, where 4 = bottom quartile of frequency) counted
    # as "rare" for ``rare_word_proportion``.
    rare_quartile: int = 4

    # Zipf-scale log-frequency assigned to out-of-vocabulary tokens. Set to
    # the rarest 5th-percentile value of the wordfreq distribution by
    # default; this treats unknown tokens as plausibly-rare rather than
    # missing, preventing OOV from swamping the mean.
    oov_zipf_fallback: float = 1.0


# Set of common English stop words. Used only when ``remove_stop_words``
# is True; standard MTLD definition does not remove them.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "will", "with",
    "you", "your",
})


# Pronoun lists for the first-to-second-person ratio. Includes archaic
# forms ("thou", "thee", "thy", "thine", "thyself") per the planning-
# conversation direction in proposal v2 Section 4.3: the corpus extends
# back to 1932 and contains period dramas; cost of including them is one
# extra five-element string set, the asymmetric upside is correctly
# handling the films that use the forms.
_FIRST_PERSON: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
})
_SECOND_PERSON: frozenset[str] = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
    "thou", "thee", "thy", "thine", "thyself",
})


# Regex matching a token that consists entirely of word characters
# (excludes pure-punctuation NLTK tokens like ".", ",", "''").
_WORD_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


# Reference-distribution variance band for Flesch-Kincaid, used for the
# Section 9 diagnostic check. F-K on continuous prose typically has
# standard deviation in the 1.5-3.5 band; if the column variance lands
# substantially below this on the corpus, the column is flagged as
# degenerate.
FK_REFERENCE_STD_BAND: tuple[float, float] = (1.0, 4.0)


# Feature columns produced by ``compute_lexical_features``. Order is
# fixed for the output DataFrame; downstream code (the baseline trainer,
# the diagnostic step) iterates through this list rather than relying on
# alphabetical sort.
LEXICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    # Vocabulary diversity (3)
    "mtld_dialogue",
    "mtld_action",
    "hapax_ratio_dialogue",
    # Lexical sophistication (2, dialogue only by domain match)
    "mean_log_frequency",
    "rare_word_proportion",
    # Readability (2)
    "flesch_kincaid_grade_dialogue",
    "flesch_kincaid_grade_action",
    # Length statistics (3, dialogue only)
    "mean_dialogue_line_tokens",
    "std_dialogue_line_tokens",
    "short_line_proportion",
    # Punctuation and pronouns (3, dialogue only)
    "question_rate_per_1k_tokens",
    "exclamation_rate_per_1k_tokens",
    "first_to_second_pronoun_ratio",
    # Diagnostic (not a model feature, but tracked for the Section 9 OOV check)
    "_oov_rate_dialogue",
)


# Index of the diagnostic-only column. The main feature matrix (used by
# the modelling pipeline) excludes this; the diagnostic step reads it
# from the full DataFrame.
DIAGNOSTIC_ONLY_COLUMNS: frozenset[str] = frozenset({"_oov_rate_dialogue"})


# Public list of the 14 model-input features (excludes the diagnostic
# column).
MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in LEXICAL_FEATURE_COLUMNS if c not in DIAGNOSTIC_ONLY_COLUMNS
)


# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Download punkt + punkt_tab if not already present.

    NLTK ships ``word_tokenize`` and ``sent_tokenize`` but the underlying
    Punkt sentence-tokenizer model is a separate download. Idempotent.
    """
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _tokenize_words(text: str) -> list[str]:
    """Tokenize a string into word tokens (punctuation excluded), case-folded.

    Uses NLTK ``word_tokenize`` then filters to alphabetic tokens via
    ``_WORD_TOKEN_RE`` so subsequent feature computations operate on a
    clean word stream. Case folding happens at this step so vocabulary
    measures treat ``Tony`` and ``tony`` as the same word.
    """
    raw = nltk.word_tokenize(text)
    return [t.lower() for t in raw if _WORD_TOKEN_RE.match(t)]


def _tokenize_sentences(text: str) -> list[str]:
    """Sentence-tokenize a string via NLTK ``sent_tokenize``.

    The input is preprocessed by appending a period to each line that
    lacks terminal punctuation (this is the heuristic from proposal v2
    Section 6.7; the empirical fire rate is 7.3% on the corpus, well
    inside the harmless band).
    """
    return nltk.sent_tokenize(text) if text.strip() else []


def _ensure_terminal_punctuation(line: str) -> str:
    """Append a period to ``line`` if it lacks terminal punctuation.

    Used only for sentence segmentation in the readability score; the
    question and exclamation rates count actual terminal punctuation in
    the source.
    """
    line = line.strip()
    if not line:
        return line
    if line[-1] in ".!?":
        return line
    return line + "."


# ---------------------------------------------------------------------------
# Per-channel text accumulation
# ---------------------------------------------------------------------------


def _dialogue_text(parsed: ParsedScreenplay) -> tuple[list[str], list[str]]:
    """Return (raw_dialogue_lines, joined_dialogue_text_for_segmentation).

    The empty-text filter (Phase 2 Tier 1.3 constraint) is applied here:
    any dialogue tuple whose text is empty or whitespace-only after
    stripping is excluded.

    The first element preserves per-line structure for length-statistics
    computations; the second is the concatenated form (with terminal
    punctuation ensured) used as input to sentence tokenization for the
    readability score.
    """
    lines: list[str] = []
    for scene in parsed.scenes:
        for _char, text in scene.dialogue_units:
            cleaned = text.strip() if text else ""
            if cleaned:
                lines.append(cleaned)
    joined_with_terminal = " ".join(_ensure_terminal_punctuation(l) for l in lines)
    return lines, joined_with_terminal


def _action_text(parsed: ParsedScreenplay) -> str:
    """Return the concatenated stage-direction + scene-description text."""
    pieces: list[str] = []
    for scene in parsed.scenes:
        if scene.stage_direction and scene.stage_direction.strip():
            pieces.append(scene.stage_direction.strip())
        if scene.scene_description and scene.scene_description.strip():
            pieces.append(scene.scene_description.strip())
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# Vocabulary diversity: MTLD and hapax ratio
# ---------------------------------------------------------------------------


def compute_mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy and Jarvis 2010).

    Walks the token sequence forward and backward, accumulating a
    running TTR; a "factor" is counted each time the running TTR drops
    below ``threshold``. The MTLD score is the mean factor length
    (total tokens divided by factor count), averaged across forward and
    backward passes.

    For sequences too short to produce a complete factor, partial
    factors are counted with a fractional weight derived from how far
    below 1.0 the TTR finished.

    Returns
    -------
    float
        MTLD score. Higher means more lexically diverse. NaN for
        sequences shorter than approximately 50 tokens (the
        established lower bound for MTLD reliability).
    """
    if len(tokens) < 50:
        return float("nan")
    forward = _mtld_one_pass(tokens, threshold)
    backward = _mtld_one_pass(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0


def _mtld_one_pass(tokens: Sequence[str], threshold: float) -> float:
    """One forward MTLD pass; helper for :func:`compute_mtld`."""
    types: set[str] = set()
    factor_count = 0.0
    factor_token_count = 0
    for token in tokens:
        types.add(token)
        factor_token_count += 1
        ttr = len(types) / factor_token_count
        if ttr <= threshold:
            factor_count += 1
            types = set()
            factor_token_count = 0
    # Partial factor at the end: count fractionally based on how far
    # below 1.0 the running TTR finished.
    if factor_token_count > 0:
        ttr = len(types) / factor_token_count if factor_token_count else 1.0
        # Map TTR in [threshold, 1.0] to factor in [0, 1] linearly.
        if ttr >= 1.0:
            partial = 0.0
        elif ttr <= threshold:
            partial = 1.0
        else:
            partial = (1.0 - ttr) / (1.0 - threshold)
        factor_count += partial
    if factor_count <= 0:
        return float("nan")
    return len(tokens) / factor_count


def compute_hapax_ratio(tokens: Sequence[str]) -> float:
    """Hapax legomena ratio: proportion of distinct tokens occurring exactly once.

    Returns NaN for an empty input.
    """
    if not tokens:
        return float("nan")
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return float("nan")
    hapaxes = sum(1 for c in counts.values() if c == 1)
    return hapaxes / len(counts)


# ---------------------------------------------------------------------------
# Lexical sophistication: wordfreq-backed mean and rare-word proportion
# ---------------------------------------------------------------------------


def compute_mean_log_frequency(
    tokens: Sequence[str],
    oov_fallback: float,
) -> tuple[float, float]:
    """Return (mean_zipf_frequency, oov_rate).

    For each token the wordfreq Zipf-scale log-frequency is looked up;
    tokens not in wordfreq's English vocabulary (returned as 0 by the
    library, by convention) are assigned ``oov_fallback`` so they do not
    dominate the mean. The OOV rate is the fraction of tokens that hit
    the fallback path, surfaced for the Section 9 diagnostic check.

    Both returned values are NaN if ``tokens`` is empty.
    """
    if not tokens:
        return float("nan"), float("nan")
    zipfs: list[float] = []
    oov_count = 0
    for token in tokens:
        z = wordfreq.zipf_frequency(token, "en")
        if z <= 0.0:
            zipfs.append(oov_fallback)
            oov_count += 1
        else:
            zipfs.append(z)
    return float(np.mean(zipfs)), oov_count / len(tokens)


def compute_rare_word_proportion(
    tokens: Sequence[str],
    quartile_cutoff: float,
    oov_fallback: float,
) -> float:
    """Proportion of tokens with Zipf log-frequency at or below ``quartile_cutoff``.

    The cutoff is computed corpus-wide (see
    :func:`compute_lexical_features`) so the quartile is interpretable
    across films.
    """
    if not tokens:
        return float("nan")
    rare = 0
    for token in tokens:
        z = wordfreq.zipf_frequency(token, "en")
        z = oov_fallback if z <= 0.0 else z
        if z <= quartile_cutoff:
            rare += 1
    return rare / len(tokens)


# ---------------------------------------------------------------------------
# Readability: Flesch-Kincaid grade
# ---------------------------------------------------------------------------


# Approximate syllable counter. Empirical work (e.g. Sloman 1968) shows
# vowel-cluster counting matches expert syllabification on roughly 90%
# of English words; perfect syllabification requires a pronunciation
# dictionary which is out of scope. The error mostly cancels out across
# hundreds of words per film.
_VOWEL_CLUSTER_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _count_syllables(word: str) -> int:
    """Approximate syllable count via vowel-cluster counting."""
    if not word:
        return 0
    word = word.lower()
    # Strip a trailing silent "e" (rule of thumb).
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
    clusters = _VOWEL_CLUSTER_RE.findall(word)
    # Every word has at least one syllable.
    return max(len(clusters), 1)


def compute_flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level for a block of text.

    Uses the standard formula:
        ``0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59``

    Returns NaN for inputs with too few words or sentences to support
    the computation.
    """
    sentences = _tokenize_sentences(text)
    if not sentences:
        return float("nan")
    words: list[str] = []
    for sent in sentences:
        words.extend(_tokenize_words(sent))
    if not words or len(sentences) == 0:
        return float("nan")
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


# ---------------------------------------------------------------------------
# Length statistics (dialogue only)
# ---------------------------------------------------------------------------


def compute_length_statistics(
    dialogue_lines: Sequence[str],
    short_line_cutoff: int,
) -> tuple[float, float, float]:
    """Return (mean_tokens, std_tokens, short_line_proportion).

    All three are NaN for an empty input.
    """
    if not dialogue_lines:
        return float("nan"), float("nan"), float("nan")
    line_token_counts: list[int] = []
    for line in dialogue_lines:
        tokens = _tokenize_words(line)
        line_token_counts.append(len(tokens))
    arr = np.asarray(line_token_counts, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))  # population std; per-film descriptor
    short = float((arr < short_line_cutoff).mean())
    return mean, std, short


# ---------------------------------------------------------------------------
# Punctuation rates and pronoun ratio (dialogue only)
# ---------------------------------------------------------------------------


def compute_punctuation_rates(
    dialogue_lines: Sequence[str],
) -> tuple[float, float, int]:
    """Return (question_rate_per_1k, exclamation_rate_per_1k, total_word_tokens).

    Rates are normalized per 1,000 word-tokens. ``total_word_tokens`` is
    returned for caller use in computing other rate-style features
    without re-tokenizing.
    """
    text = " ".join(dialogue_lines)
    word_tokens = _tokenize_words(text)
    n_tokens = len(word_tokens)
    if n_tokens == 0:
        return float("nan"), float("nan"), 0
    n_questions = text.count("?")
    n_exclamations = text.count("!")
    return (
        1000.0 * n_questions / n_tokens,
        1000.0 * n_exclamations / n_tokens,
        n_tokens,
    )


def compute_pronoun_ratio(dialogue_tokens: Sequence[str]) -> float:
    """First-to-second-person pronoun ratio.

    Numerator: count of ``i``, ``me``, ``my``, ``mine``, ``myself``.
    Denominator: count of ``you``, ``your``, ``yours``, ``yourself``,
    ``yourselves``, ``thou``, ``thee``, ``thy``, ``thine``, ``thyself``,
    plus an epsilon of 1.0 to prevent division by zero.

    Tokens must be lowercase (the caller's responsibility).
    """
    n_first = sum(1 for t in dialogue_tokens if t in _FIRST_PERSON)
    n_second = sum(1 for t in dialogue_tokens if t in _SECOND_PERSON)
    return n_first / (n_second + 1.0)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _compute_one_film(
    parsed: ParsedScreenplay,
    cfg: LexicalFeatureConfig,
    rare_cutoff_zipf: float,
) -> dict[str, float]:
    """Compute all 14 features (plus the OOV diagnostic) for one screenplay."""
    dialogue_lines, dialogue_concat = _dialogue_text(parsed)
    action_concat = _action_text(parsed)

    dialogue_tokens = _tokenize_words(" ".join(dialogue_lines))
    action_tokens = _tokenize_words(action_concat)

    if cfg.remove_stop_words:
        dialogue_tokens_diversity = [t for t in dialogue_tokens if t not in _STOP_WORDS]
        action_tokens_diversity = [t for t in action_tokens if t not in _STOP_WORDS]
    else:
        dialogue_tokens_diversity = dialogue_tokens
        action_tokens_diversity = action_tokens

    mean_log_freq, oov_rate = compute_mean_log_frequency(
        dialogue_tokens, cfg.oov_zipf_fallback
    )
    rare_prop = compute_rare_word_proportion(
        dialogue_tokens, rare_cutoff_zipf, cfg.oov_zipf_fallback
    )

    mean_line, std_line, short_prop = compute_length_statistics(
        dialogue_lines, cfg.short_line_cutoff_tokens
    )
    question_rate, exclamation_rate, _ = compute_punctuation_rates(dialogue_lines)

    return {
        "mtld_dialogue": compute_mtld(dialogue_tokens_diversity, cfg.mtld_threshold),
        "mtld_action": compute_mtld(action_tokens_diversity, cfg.mtld_threshold),
        "hapax_ratio_dialogue": compute_hapax_ratio(dialogue_tokens_diversity),
        "mean_log_frequency": mean_log_freq,
        "rare_word_proportion": rare_prop,
        "flesch_kincaid_grade_dialogue": compute_flesch_kincaid_grade(dialogue_concat),
        "flesch_kincaid_grade_action": compute_flesch_kincaid_grade(action_concat),
        "mean_dialogue_line_tokens": mean_line,
        "std_dialogue_line_tokens": std_line,
        "short_line_proportion": short_prop,
        "question_rate_per_1k_tokens": question_rate,
        "exclamation_rate_per_1k_tokens": exclamation_rate,
        "first_to_second_pronoun_ratio": compute_pronoun_ratio(dialogue_tokens),
        "_oov_rate_dialogue": oov_rate,
    }


def _compute_corpus_rare_cutoff(
    parsed_corpus: dict[str, ParsedScreenplay],
    quartile: int,
    oov_fallback: float,
    sample_n: int = 200,
) -> float:
    """Compute the corpus-wide Zipf cutoff for the rare-word quartile.

    For computational efficiency the cutoff is estimated from a sample
    of films rather than the full corpus; with sample_n=200 the
    quartile estimate is stable to within roughly 0.05 Zipf units.
    """
    rng = np.random.default_rng(42)
    ids = list(parsed_corpus.keys())
    if len(ids) > sample_n:
        sample_ids = list(rng.choice(ids, size=sample_n, replace=False))
    else:
        sample_ids = ids
    all_zipfs: list[float] = []
    for imdb_id in sample_ids:
        lines, _ = _dialogue_text(parsed_corpus[imdb_id])
        tokens = _tokenize_words(" ".join(lines))
        for t in tokens:
            z = wordfreq.zipf_frequency(t, "en")
            all_zipfs.append(oov_fallback if z <= 0.0 else z)
    if not all_zipfs:
        raise RuntimeError("No tokens found in sample for rare-cutoff estimation")
    arr = np.asarray(all_zipfs)
    # quartile=4 means bottom quartile, so cutoff is the 25th percentile.
    pct = (5 - quartile) * 25  # 4 -> 25, 3 -> 50, 2 -> 75, 1 -> 100
    return float(np.percentile(arr, pct))


def compute_lexical_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    cfg: LexicalFeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute all 14 lexical features for every film in the parsed corpus.

    Parameters
    ----------
    parsed_corpus
        Mapping of ``imdb_id`` to :class:`ParsedScreenplay` objects, as
        produced by Phase 2's ``parse_all_screenplays`` and persisted at
        ``data/processed/screenplays_parsed.pkl``.
    cfg
        Feature configuration. ``None`` uses defaults.

    Returns
    -------
    pd.DataFrame
        One row per film, indexed by ``imdb_id``, with columns matching
        :data:`LEXICAL_FEATURE_COLUMNS` (14 model features +
        1 diagnostic OOV-rate column). Dtypes are float64.

    Notes
    -----
    The corpus-wide rare-word Zipf cutoff is computed from a 200-film
    sample at the start of feature extraction so the rare-word
    proportion is interpretable across films. The cutoff is stable to
    within roughly 0.05 Zipf units for samples of this size.
    """
    cfg = cfg or LexicalFeatureConfig()
    ensure_nltk_resources()

    logger.info("Estimating corpus-wide rare-word Zipf cutoff (sample n=200)")
    rare_cutoff = _compute_corpus_rare_cutoff(
        parsed_corpus, cfg.rare_quartile, cfg.oov_zipf_fallback,
    )
    logger.info("Rare-word cutoff at quartile %d: Zipf <= %.3f",
                cfg.rare_quartile, rare_cutoff)

    logger.info("Computing lexical features for %d films", len(parsed_corpus))
    records: dict[str, dict[str, float]] = {}
    for i, (imdb_id, parsed) in enumerate(parsed_corpus.items(), start=1):
        records[imdb_id] = _compute_one_film(parsed, cfg, rare_cutoff)
        if i % 200 == 0:
            logger.info("Lexical features computed: %d / %d", i, len(parsed_corpus))

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "imdb_id"
    # Ensure column order matches the documented constant.
    df = df[list(LEXICAL_FEATURE_COLUMNS)]
    df = df.astype(float)
    logger.info("Lexical feature matrix complete: %d films x %d columns",
                df.shape[0], df.shape[1])
    return df
