"""Phase 3b sentiment feature group.

Computes 22 sentiment features per film at three levels of pooling
abstraction:

* **Whole-screenplay aggregates (11 features).** Three VADER compound-
  score statistics over dialogue lines, plus eight NRC-EmoLex emotion
  proportions over non-stopword dialogue tokens.
* **Scene-windowed (5 features).** Four within-quartile compound-score
  means and one cross-quartile volatility-concentration measure, with
  quartiles defined on the dialogue-line index (robust to films with
  collapsed scene structure).
* **Arc-clustered (6 features).** Cosine similarities between each
  film's per-line compound trajectory and six hand-coded Reagan et al.
  (2016) archetype templates.

The feature design follows the planning-conversation-approved v2
proposal at ``docs/proposals/phase3_sentiment_proposal.md``. See that
document for per-feature rationale and pre-registered lift bands.

Tactical implementation note (NRC-EmoLex source)
------------------------------------------------

The proposal specifies the canonical NRC Word-Emotion Association
Lexicon (Mohammad and Turney 2013; saifmohammad.com). The canonical
distribution is form-gated (the user submits name and affiliation and
receives the zip by email), which blocks an automated hash-checked
download into ``data/external/``. The implementation backend is the
``nrclex`` Python package, which ships the same word-emotion mappings
under the author's research-use license. The bundled lexicon contains
6,468 word entries (filtered to those with at least one emotion tag),
which matches the proposal's expected mechanism — surface-form
matching for the eight emotion categories anger, anticipation,
disgust, fear, joy, sadness, surprise, trust. The ``nrclex`` lexicon
preserves separate entries for related surface forms (``happy`` vs
``happiness``, ``fear`` vs ``fearful``), which the proposal's Section
3.2 specifically called for.

This deviation matches the wordfreq-vs-SUBTLEX-US precedent set in
the lexical group: when the canonical distribution is friction-loaded
for reproducibility, use a stable Python package wrapper that
preserves the conceptual mechanism, name the deviation in the run's
preprocessing metadata, and document it in the handoff. The
``SentimentFeatureConfig.lexicon_path`` knob is exposed so a future
strict-canonical run can swap in a manually-downloaded
``data/external/nrc_emolex.tsv`` without code changes.

Dependencies
------------

* ``nltk`` for ``word_tokenize``, ``sent_tokenize``, the VADER
  sentiment analyzer, and the English stopword list. Required NLTK
  resources are downloaded on demand by ``ensure_nltk_resources``.
* ``nrclex`` for the NRC Word-Emotion Association Lexicon. The
  bundled lexicon ships with the package; no external download.

References
----------

* Hutto, C., and Gilbert, E. (2014). VADER.
* Mohammad, S. M., and Turney, P. D. (2013). NRC EmoLex.
* Reagan, A. J., et al. (2016). The emotional arcs of stories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nltk
import numpy as np
import pandas as pd
from nrclex import NRCLex

from src.data.parse_screenplay import ParsedScreenplay
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentFeatureConfig:
    """Knobs for the sentiment feature pipeline.

    Defaults reproduce the values in proposal v2 Section 8.3.
    """

    # Number of contiguous dialogue-line-index windows for the scene-
    # windowed pooling block. The proposal commits to four (quartiles).
    n_quartile_windows: int = 4

    # Length to which each film's per-line compound trajectory is
    # interpolated before cosine similarity against archetype templates.
    arc_template_length: int = 100

    # Whether to remove English stopwords from dialogue tokens before
    # NRC matching. Default True per proposal Section 3.2: NLTK English
    # stopwords skew per-emotion proportions toward function words that
    # NRC tags weakly or not at all.
    remove_stopwords_for_nrc: bool = True

    # Identifier of the archetype set used for the arc-clustered block.
    # The implementation supports only ``"reagan_six"`` at present;
    # exposed as a knob so a future SVD-derived set can be swapped in.
    archetype_set: str = "reagan_six"

    # Optional path to an externally-downloaded NRC EmoLex JSON file
    # (matching the schema of nrclex's bundled ``nrc_en.json``). Default
    # ``None`` uses the ``nrclex`` package's bundled lexicon. Reserved
    # for a future strict-canonical run that replaces the package source
    # with the form-gated direct download from saifmohammad.com.
    lexicon_path: Path | None = None


# Eight NRC emotion categories used as features. The bundled lexicon
# also carries binary ``positive`` and ``negative`` tags; per the
# proposal these are intentionally excluded from the feature set
# because the eight-dimensional representation already encodes the
# polarity information without the over-counting bias documented in
# the proposal Section 3.2.
NRC_EMOTIONS: tuple[str, ...] = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)


# Names of the six Reagan et al. (2016) archetype templates.
REAGAN_ARCHETYPES: tuple[str, ...] = (
    "rags_to_riches",
    "tragedy",
    "man_in_a_hole",
    "icarus",
    "cinderella",
    "oedipus",
)


# Feature columns produced by ``compute_sentiment_features``. Order is
# fixed for the output DataFrame; downstream code (the baseline trainer,
# the diagnostic step) iterates through this list rather than relying
# on alphabetical sort.
SENTIMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    # VADER aggregates (3, whole-screenplay)
    "vader_compound_mean",
    "vader_compound_std",
    "vader_compound_range",
    # NRC emotion proportions (8, whole-screenplay)
    "nrc_anger_proportion",
    "nrc_anticipation_proportion",
    "nrc_disgust_proportion",
    "nrc_fear_proportion",
    "nrc_joy_proportion",
    "nrc_sadness_proportion",
    "nrc_surprise_proportion",
    "nrc_trust_proportion",
    # Quartile-windowed sentiment trajectory (5, scene-windowed)
    "sentiment_q1_compound_mean",
    "sentiment_q2_compound_mean",
    "sentiment_q3_compound_mean",
    "sentiment_q4_compound_mean",
    "sentiment_volatility_concentration",
    # Reagan arc archetype similarities (6, arc-clustered)
    "arc_similarity_rags_to_riches",
    "arc_similarity_tragedy",
    "arc_similarity_man_in_a_hole",
    "arc_similarity_icarus",
    "arc_similarity_cinderella",
    "arc_similarity_oedipus",
    # Diagnostics (Section 9 of proposal v2; not model features)
    "_nrc_oov_rate_dialogue",
    "_vader_zero_compound_rate",
)


# Diagnostic-only columns excluded from the model-input matrix.
DIAGNOSTIC_ONLY_COLUMNS: frozenset[str] = frozenset({
    "_nrc_oov_rate_dialogue",
    "_vader_zero_compound_rate",
})


# Public list of the 22 model-input features.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in SENTIMENT_FEATURE_COLUMNS if c not in DIAGNOSTIC_ONLY_COLUMNS
)


# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Download punkt + punkt_tab + vader_lexicon + stopwords if missing.

    Idempotent. Lexical and sentiment groups share punkt and punkt_tab;
    the vader_lexicon and stopwords resources are added here and stay
    cached after first invocation.
    """
    tokenizer_resources = (
        ("tokenizers", "punkt"),
        ("tokenizers", "punkt_tab"),
        ("sentiment", "vader_lexicon"),
        ("corpora", "stopwords"),
    )
    for category, resource in tokenizer_resources:
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def _english_stopwords() -> frozenset[str]:
    """Return the cached NLTK English stopword set."""
    from nltk.corpus import stopwords as _stopwords
    return frozenset(_stopwords.words("english"))


# ---------------------------------------------------------------------------
# Per-channel text accumulation (mirrors src.features.lexical)
# ---------------------------------------------------------------------------


def _dialogue_lines(parsed: ParsedScreenplay) -> list[str]:
    """Return the dialogue lines of a screenplay after the empty-text filter.

    Phase 2 Tier 1.3 filter: any dialogue tuple whose text is empty or
    whitespace-only after stripping is excluded. Applied defensively
    here even though Phase 2 already filters at parse time.
    """
    out: list[str] = []
    for scene in parsed.scenes:
        for _char, text in scene.dialogue_units:
            cleaned = text.strip() if text else ""
            if cleaned:
                out.append(cleaned)
    return out


def _tokenize_dialogue_to_lower(lines: Sequence[str]) -> list[str]:
    """Concatenate ``lines`` and tokenize to lowercase alphabetic tokens.

    Uses NLTK ``word_tokenize`` and filters to tokens consisting purely
    of letters (matches the lexical module's word-token regex).
    """
    import re
    word_re = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
    text = " ".join(lines)
    raw = nltk.word_tokenize(text)
    return [t.lower() for t in raw if word_re.match(t)]


# ---------------------------------------------------------------------------
# VADER aggregates (whole-screenplay)
# ---------------------------------------------------------------------------


def compute_vader_aggregates(
    lines: Sequence[str],
    sia,
) -> tuple[float, float, float, float]:
    """Compute (mean, std, range, zero_compound_rate) over dialogue lines.

    VADER's compound score is in ``[-1, 1]``. The zero-compound rate
    is the fraction of lines that produced exactly 0.0 (i.e. VADER
    found no sentiment-bearing tokens in the line); surfaced as a
    diagnostic per proposal Section 9 #5.

    Returns
    -------
    (mean, std, value_range, zero_compound_rate)
        All NaN for empty input. ``value_range`` is max − min.
    """
    if not lines:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    compounds: list[float] = []
    n_zero = 0
    for line in lines:
        score = sia.polarity_scores(line)["compound"]
        compounds.append(float(score))
        if score == 0.0:
            n_zero += 1
    arr = np.asarray(compounds, dtype=float)
    return (
        float(arr.mean()),
        float(arr.std(ddof=0)),
        float(arr.max() - arr.min()),
        n_zero / len(lines),
    )


# ---------------------------------------------------------------------------
# NRC emotion proportions (whole-screenplay)
# ---------------------------------------------------------------------------


def compute_nrc_proportions(
    tokens: Sequence[str],
    *,
    remove_stopwords: bool,
    stopwords: frozenset[str],
    lexicon_path: Path | None = None,
) -> tuple[dict[str, float], float]:
    """Per-emotion proportion over (non-stopword) dialogue tokens.

    For each token in the input, NRCLex looks up the surface form in
    the bundled NRC EmoLex; tokens with at least one emotion tag
    contribute their tag(s) to the per-emotion counts (multiple tags
    per token allowed). Per-emotion proportion is the per-emotion
    count divided by the size of the (possibly stopword-filtered) input.
    The OOV rate is the fraction of tokens not present in the NRC
    lexicon.

    Returns
    -------
    ({emotion: proportion}, oov_rate)
        ``emotion`` key set is :data:`NRC_EMOTIONS`. NaN for empty input.
    """
    if not tokens:
        nan_dict = {e: float("nan") for e in NRC_EMOTIONS}
        return nan_dict, float("nan")
    if remove_stopwords:
        filtered = [t for t in tokens if t not in stopwords]
    else:
        filtered = list(tokens)
    if not filtered:
        nan_dict = {e: float("nan") for e in NRC_EMOTIONS}
        return nan_dict, float("nan")

    nrc = NRCLex(lexicon_file=lexicon_path) if lexicon_path else NRCLex()
    nrc.load_token_list(filtered)
    counts = nrc.raw_emotion_scores
    proportions = {
        e: float(counts.get(e, 0)) / len(filtered) for e in NRC_EMOTIONS
    }

    # OOV rate: tokens with NO entry in the lexicon (no tag at all).
    # NRCLex's ``affect_dict`` keys are matched-token forms.
    matched = set(nrc.affect_dict.keys())
    n_oov = sum(1 for t in filtered if t not in matched)
    oov_rate = n_oov / len(filtered)
    return proportions, oov_rate


# ---------------------------------------------------------------------------
# Quartile-windowed compound trajectory (scene-windowed)
# ---------------------------------------------------------------------------


def compute_quartile_features(
    per_line_compounds: Sequence[float],
    n_windows: int,
) -> tuple[float, float, float, float, float]:
    """Return (q1, q2, q3, q4, volatility_concentration).

    The four within-quartile means are the per-window means of the
    VADER compound trajectory. Volatility concentration is
    ``max - min`` across the four per-quartile standard deviations.
    Quartiles are defined on the dialogue-line index, not scene
    boundaries, so films with collapsed scene structure
    (``data_quality_flag``) still produce well-defined values.

    All five returned values are NaN if the input has fewer than
    ``n_windows`` lines.
    """
    n = len(per_line_compounds)
    if n < n_windows:
        return (float("nan"),) * 5
    if n_windows != 4:
        # The feature schema is hard-coded to 4 quartile means + 1
        # volatility column; rewriting for arbitrary window counts is
        # out of scope. Surfaces as a clear error rather than a silent
        # mismatch.
        raise ValueError(
            f"compute_quartile_features expects n_windows=4; got {n_windows}"
        )
    arr = np.asarray(per_line_compounds, dtype=float)
    # ``np.array_split`` divides into n_windows chunks of as-equal-as-
    # possible size; with n=880 and n_windows=4 this is 220 lines each.
    chunks = np.array_split(arr, n_windows)
    means = [float(c.mean()) for c in chunks]
    stds = [float(c.std(ddof=0)) for c in chunks]
    volatility_concentration = float(max(stds) - min(stds))
    return (means[0], means[1], means[2], means[3], volatility_concentration)


# ---------------------------------------------------------------------------
# Reagan arc-archetype templates and similarity (arc-clustered)
# ---------------------------------------------------------------------------


def build_reagan_templates(length: int) -> dict[str, np.ndarray]:
    """Build the six Reagan archetype templates at the given length.

    Mathematical forms per proposal Section 3.4:

    * Rags to Riches: linear ramp from -1 to +1.
    * Tragedy: linear ramp from +1 to -1 (reflection of above).
    * Man in a Hole: cosine wave with a trough at the midpoint
      (fall-rise shape; +1 at the ends, -1 in the middle).
    * Icarus: reflection of Man in a Hole (peak at midpoint).
    * Cinderella: three-cycle cosine pattern with maxima near the
      ends and middle, troughs near the quartile points (rise, fall,
      rise, fall, rise).
    * Oedipus: reflection of Cinderella.

    All six are returned z-score normalized (mean 0, std 1) so cosine
    similarity against a film's z-score-normalized trajectory is
    invariant to the trajectory's amplitude.

    Returns a mapping with keys :data:`REAGAN_ARCHETYPES`.
    """
    if length < 4:
        raise ValueError(f"arc template length must be >= 4; got {length}")
    t = np.arange(length, dtype=float)

    # Linear shapes.
    rr = -1.0 + 2.0 * t / (length - 1)
    tr = -rr

    # Single-cycle cosine over t=0..length-1 evaluates to +1 at both
    # endpoints and -1 at the midpoint, exactly the Man-in-a-Hole
    # fall-rise shape. Icarus is the reflection.
    mh = np.cos(2.0 * np.pi * t / (length - 1))
    ic = -mh

    # Two-cycle cosine evaluates to +1 at t=0, -1 at t=(L-1)/4, +1 at
    # t=(L-1)/2, -1 at t=3(L-1)/4, +1 at t=L-1. Three maxima and two
    # troughs (Cinderella). Oedipus is the reflection.
    ci = np.cos(4.0 * np.pi * t / (length - 1))
    oe = -ci

    raw = {
        "rags_to_riches": rr,
        "tragedy": tr,
        "man_in_a_hole": mh,
        "icarus": ic,
        "cinderella": ci,
        "oedipus": oe,
    }
    # Z-score normalize so cosine similarity is amplitude-invariant.
    out: dict[str, np.ndarray] = {}
    for name, vec in raw.items():
        mu = vec.mean()
        sd = vec.std(ddof=0)
        if sd == 0.0:
            # Degenerate: should not happen for the six shapes above,
            # but defensive.
            out[name] = vec - mu
        else:
            out[name] = (vec - mu) / sd
    return out


def _interpolate_to_length(values: Sequence[float], length: int) -> np.ndarray:
    """Linearly interpolate ``values`` onto a uniform grid of size ``length``."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == length:
        return arr.copy()
    if n < 2:
        # Cannot interpolate from a single point; return constant.
        return np.full(length, arr[0] if n == 1 else np.nan, dtype=float)
    # Map original indices [0, n-1] to target indices [0, length-1].
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, arr)


def _zscore_or_nan(values: np.ndarray) -> np.ndarray | None:
    """Z-score normalize; return None if std is zero (degenerate trajectory)."""
    sd = values.std(ddof=0)
    if sd == 0.0 or not np.isfinite(sd):
        return None
    return (values - values.mean()) / sd


def compute_arc_similarities(
    per_line_compounds: Sequence[float],
    templates: dict[str, np.ndarray],
    target_length: int,
) -> dict[str, float]:
    """Cosine similarity of a film's compound trajectory to each archetype.

    The trajectory is interpolated to ``target_length``, z-score
    normalized, then compared to each (already z-normalized) template
    via cosine similarity. Films with degenerate (constant) trajectories
    receive NaN for every archetype.

    Returns
    -------
    dict[str, float]
        Keys match :data:`REAGAN_ARCHETYPES`; values in ``[-1, 1]`` or
        NaN.
    """
    if len(per_line_compounds) < 2:
        return {a: float("nan") for a in REAGAN_ARCHETYPES}
    interp = _interpolate_to_length(per_line_compounds, target_length)
    z = _zscore_or_nan(interp)
    if z is None:
        return {a: float("nan") for a in REAGAN_ARCHETYPES}
    out: dict[str, float] = {}
    z_norm = float(np.linalg.norm(z))
    if z_norm == 0.0:
        return {a: float("nan") for a in REAGAN_ARCHETYPES}
    for name in REAGAN_ARCHETYPES:
        tpl = templates[name]
        tpl_norm = float(np.linalg.norm(tpl))
        if tpl_norm == 0.0:
            out[name] = float("nan")
            continue
        out[name] = float(np.dot(z, tpl) / (z_norm * tpl_norm))
    return out


# ---------------------------------------------------------------------------
# Per-film orchestration
# ---------------------------------------------------------------------------


def _compute_one_film(
    parsed: ParsedScreenplay,
    cfg: SentimentFeatureConfig,
    sia,
    stopwords: frozenset[str],
    archetype_templates: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute all 22 features (plus 2 diagnostics) for one screenplay."""
    lines = _dialogue_lines(parsed)

    # VADER per-line compound scores (used by aggregates, quartiles, arcs).
    per_line: list[float] = [
        float(sia.polarity_scores(line)["compound"]) for line in lines
    ]

    # Whole-screenplay aggregates: VADER + NRC.
    if per_line:
        arr = np.asarray(per_line, dtype=float)
        vader_mean = float(arr.mean())
        vader_std = float(arr.std(ddof=0))
        vader_range = float(arr.max() - arr.min())
        n_zero = int(np.sum(arr == 0.0))
        zero_rate = n_zero / len(per_line)
    else:
        vader_mean = vader_std = vader_range = float("nan")
        zero_rate = float("nan")

    tokens = _tokenize_dialogue_to_lower(lines)
    proportions, oov_rate = compute_nrc_proportions(
        tokens,
        remove_stopwords=cfg.remove_stopwords_for_nrc,
        stopwords=stopwords,
        lexicon_path=cfg.lexicon_path,
    )

    # Scene-windowed (quartile) trajectory features.
    q1, q2, q3, q4, vol_conc = compute_quartile_features(
        per_line, cfg.n_quartile_windows,
    )

    # Arc-clustered similarity features.
    arc_sims = compute_arc_similarities(
        per_line, archetype_templates, cfg.arc_template_length,
    )

    return {
        "vader_compound_mean": vader_mean,
        "vader_compound_std": vader_std,
        "vader_compound_range": vader_range,
        "nrc_anger_proportion": proportions["anger"],
        "nrc_anticipation_proportion": proportions["anticipation"],
        "nrc_disgust_proportion": proportions["disgust"],
        "nrc_fear_proportion": proportions["fear"],
        "nrc_joy_proportion": proportions["joy"],
        "nrc_sadness_proportion": proportions["sadness"],
        "nrc_surprise_proportion": proportions["surprise"],
        "nrc_trust_proportion": proportions["trust"],
        "sentiment_q1_compound_mean": q1,
        "sentiment_q2_compound_mean": q2,
        "sentiment_q3_compound_mean": q3,
        "sentiment_q4_compound_mean": q4,
        "sentiment_volatility_concentration": vol_conc,
        "arc_similarity_rags_to_riches": arc_sims["rags_to_riches"],
        "arc_similarity_tragedy": arc_sims["tragedy"],
        "arc_similarity_man_in_a_hole": arc_sims["man_in_a_hole"],
        "arc_similarity_icarus": arc_sims["icarus"],
        "arc_similarity_cinderella": arc_sims["cinderella"],
        "arc_similarity_oedipus": arc_sims["oedipus"],
        "_nrc_oov_rate_dialogue": oov_rate,
        "_vader_zero_compound_rate": zero_rate,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_sentiment_features(
    parsed_corpus: dict[str, ParsedScreenplay],
    cfg: SentimentFeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute all 22 sentiment features for every film in the parsed corpus.

    Parameters
    ----------
    parsed_corpus
        Mapping of ``imdb_id`` to :class:`ParsedScreenplay` objects, as
        produced by Phase 2's ``parse_all_screenplays`` and persisted at
        ``data/processed/screenplays_parsed.pkl``.
    cfg
        Feature configuration; ``None`` uses defaults.

    Returns
    -------
    pd.DataFrame
        One row per film, indexed by ``imdb_id``, with columns matching
        :data:`SENTIMENT_FEATURE_COLUMNS` (22 model features +
        2 diagnostic columns). Dtypes are float64.
    """
    cfg = cfg or SentimentFeatureConfig()
    ensure_nltk_resources()

    # Lazily import VADER's class-based analyzer so the lexicon download
    # in ``ensure_nltk_resources`` runs first.
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    stopwords = _english_stopwords()

    if cfg.archetype_set != "reagan_six":
        raise ValueError(
            f"Unsupported archetype_set {cfg.archetype_set!r}; "
            "the implementation supports only 'reagan_six' at present"
        )
    archetype_templates = build_reagan_templates(cfg.arc_template_length)

    logger.info(
        "Computing sentiment features for %d films "
        "(remove_stopwords_for_nrc=%s, n_quartile_windows=%d, "
        "arc_template_length=%d)",
        len(parsed_corpus),
        cfg.remove_stopwords_for_nrc,
        cfg.n_quartile_windows,
        cfg.arc_template_length,
    )
    records: dict[str, dict[str, float]] = {}
    for i, (imdb_id, parsed) in enumerate(parsed_corpus.items(), start=1):
        records[imdb_id] = _compute_one_film(
            parsed, cfg, sia, stopwords, archetype_templates,
        )
        if i % 200 == 0:
            logger.info("Sentiment features computed: %d / %d", i, len(parsed_corpus))

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "imdb_id"
    df = df[list(SENTIMENT_FEATURE_COLUMNS)]
    df = df.astype(float)
    logger.info(
        "Sentiment feature matrix complete: %d films x %d columns",
        df.shape[0], df.shape[1],
    )
    return df
