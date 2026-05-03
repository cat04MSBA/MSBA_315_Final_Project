"""MovieSum screenplay XML parser.

Phase 1 confirmed MovieSum's XML structure:

::

    <script>
      <scene>
        <stage_direction>EXT. PARIS - DAY</stage_direction>
        <scene_description>The Eiffel Tower looms in the distance...</scene_description>
        <character>ALICE</character>
        <dialogue>Bonjour, Bob.</dialogue>
        <character>BOB</character>
        <dialogue>Bonjour.</dialogue>
        ...
      </scene>
      ...
    </script>

A scene is a flat sequence of `stage_direction`, `scene_description`,
`character`, and `dialogue` elements. Character/dialogue pairs alternate
to encode "Alice says X, then Bob says Y." This parser reconstructs the
implied (character, dialogue) pairs by walking the scene's elements in
document order.

Output is a frozen dataclass tree (``Scene`` and ``ParsedScreenplay``)
plus a set of structural metrics computed at parse time. The parser is
deterministic: same input always produces the same output.

Edge cases (per Phase 2 brief Task 2):

* Malformed XML — returns a degenerate ``ParsedScreenplay`` with empty
  scenes and the error captured in ``parse_warnings``. Does not raise.
* Wrong root tag (not ``<script>``) — emits a warning to
  ``parse_warnings`` and continues parsing scene children if any.
* Character-name normalization (Tier 1.1) — trailing parenthetical
  variant suffixes are stripped from `<character>` text:
  ``"TONY (CONT'D)"``, ``"TONY (V.O.)"``, ``"TONY (O.S.)"`` all
  normalize to ``"TONY"``. If stripping would leave the empty string
  (rare; e.g. ``"(WAITER)"``), the original is preserved.
* Implausible character names (Tier 1.2 conservative filter) — strings
  that contain ``©``/``®``/``™``, start with a 4-digit year, or
  contain studio-attribution substrings (``STUDIOS``,
  ``PICTURES INC``, ``PRODUCTIONS LLC``) are rejected as not-real
  characters. The tag is treated as a flow break: a warning is
  recorded, ``last_speaker`` is reset, and the next ``<dialogue>``
  lands on the orphan path (Case 11) rather than being attributed to
  the implausible name.
* Unique-character counting (Tier 1.3) — a character only counts in
  ``n_unique_characters`` if they delivered at least one non-empty,
  non-whitespace dialogue line. Empty-text placeholders (left in
  ``dialogue_units`` for traceability when Cases 5-8 fire) do not
  contribute.
* ``<parenthetical>`` elements (e.g. ``"(softly)"``, ``"(beat)"``).
  These are a real, frequent screenplay element; we recognize them
  silently and use them as "continuation markers" so dialogue that
  follows a parenthetical without an intervening ``<character>`` is
  correctly attributed to the same speaker.
* Dialogue with no preceding ``<character>`` but the same speaker is
  continuing (i.e. a stage_direction or scene_description has not
  intervened) — attributed to the previous speaker, no warning.
* Dialogue genuinely orphaned (no preceding character AND a
  stage_direction or scene_description has reset the dialogue flow) —
  paired with character ``""`` and a warning recorded.
* Scenes with no dialogue — ``dialogue_units`` is an empty tuple.
* Empty ``<stage_direction>`` or ``<scene_description>`` — stored as
  empty strings rather than dropped.

A ratio note: the Phase 2 brief defines
``dialogue_to_action_ratio = total_dialogue_chars / (total_dialogue_chars
+ total_stage_direction_chars)``. In MovieSum, ``stage_direction`` is
typically just the scene slugline ("INT. KITCHEN - DAY"), so this
literal formula yields values close to 1 for most scripts. We expose
the literal formula alongside a more informative
``dialogue_to_total_text_ratio`` that includes ``scene_description``
chars too, plus the raw character counts for both, so downstream phases
can pick whichever is appropriate.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable

from src.utils.logging import get_logger

logger = get_logger(__name__)


# --- Character-name normalization (Tier 1.1) ---------------------------------
# Trailing parenthetical suffixes mark variants of the same speaker:
# "TONY (CONT'D)", "TONY (V.O.)", "TONY (O.S.)", "TONY (PRELAP)", etc.
# We normalize them all to "TONY" so the unique-character count reflects
# the speaking entities, not the formatting variants.
_VARIANT_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")

# --- Implausible-character-name filter (Tier 1.2 conservative) ----------------
# Strings that match any of these patterns are rejected as not-a-character.
# The set is deliberately narrow to keep false-positive risk low.
_IMPLAUSIBLE_NAME_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"[©®™]"),                # © ® ™
    re.compile(r"^\d{4}\b"),                            # starts with a 4-digit year
    re.compile(r"\bSTUDIOS\b", re.IGNORECASE),
    re.compile(r"\bPICTURES\s+INC\b", re.IGNORECASE),
    re.compile(r"\bPRODUCTIONS\s+LLC\b", re.IGNORECASE),
)


def _normalize_character_name(raw: str) -> str:
    """Strip trailing parenthetical variant markers (CONT'D, V.O., etc.).

    If stripping leaves an empty string (the raw name was *only* a
    parenthetical, as occasionally happens with unnamed characters like
    ``(WAITER)``), the original is preserved so the subsequent
    plausibility check has something to evaluate.
    """
    if not raw:
        return raw
    stripped = _VARIANT_SUFFIX_RE.sub("", raw).strip()
    return stripped if stripped else raw


def _is_plausible_character_name(name: str) -> bool:
    """Conservative reject filter for `<character>` content that isn't a real name.

    Returns False on copyright/registered/trademark symbols, year-prefixed
    strings, and well-known studio-attribution substrings. Plain names
    (regardless of length, casing, or punctuation) pass.
    """
    if not name:
        return False
    return not any(p.search(name) for p in _IMPLAUSIBLE_NAME_PATTERNS)


@dataclass(frozen=True)
class Scene:
    """One scene within a screenplay.

    Attributes
    ----------
    scene_number
        1-indexed position in the screenplay.
    stage_direction
        Concatenated text of all ``<stage_direction>`` elements in the
        scene. Usually a single slugline like ``"INT. KITCHEN - DAY"``.
    scene_description
        Concatenated text of all ``<scene_description>`` elements in
        the scene. Usually one or more paragraphs of action.
    dialogue_units
        Ordered tuple of ``(character_name, dialogue_text)`` pairs.
        Character names with no following dialogue, or dialogue with
        no preceding character, are included with the missing side as
        an empty string.
    """
    scene_number: int
    stage_direction: str
    scene_description: str
    dialogue_units: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ParsedScreenplay:
    """Parsed structure for one MovieSum screenplay.

    The structural metric fields are denormalized onto the master
    DataFrame in `src.data.build_corpus`; the full ``scenes`` list is
    saved separately to ``data/processed/screenplays_parsed.pkl``.
    """
    imdb_id: str
    scenes: tuple[Scene, ...]
    parse_warnings: tuple[str, ...]
    # Structural metrics (denormalized onto the master parquet).
    n_scenes: int
    n_unique_characters: int
    n_dialogue_lines: int
    total_dialogue_chars: int
    total_stage_direction_chars: int
    total_scene_description_chars: int
    total_action_chars: int  # stage_direction + scene_description
    # Two ratios. ``dialogue_to_action_ratio`` follows the brief's literal
    # formula and tends to be ~0.99 because stage_direction is usually
    # just a slugline. ``dialogue_to_total_text_ratio`` is more
    # informative for downstream features.
    dialogue_to_action_ratio: float
    dialogue_to_total_text_ratio: float


def _stripped(text: str | None) -> str:
    """Return ``text.strip()`` or ``""`` if ``text`` is ``None``."""
    return text.strip() if text else ""


def _parse_one_scene(
    scene_element: ET.Element,
    scene_number: int,
    warnings: list[str],
) -> Scene:
    """Walk a ``<scene>`` element's children in order and build a Scene.

    Reconstructs ``(character, dialogue)`` pairs from the alternating
    sequence. If the structure breaks (e.g., two characters in a row,
    or dialogue with no preceding character), records a warning and
    keeps going with empty-string fillers so downstream code can still
    iterate.
    """
    stage_directions: list[str] = []
    scene_descriptions: list[str] = []
    dialogue_units: list[tuple[str, str]] = []
    # ``pending_character`` is set after a <character> tag, awaiting its
    # paired <dialogue>. ``last_speaker`` is the most recent character
    # who actually spoke; used to attribute continuation dialogue
    # (e.g., second <dialogue> after a <parenthetical>) to the same
    # speaker. ``last_speaker`` resets when a stage_direction or
    # scene_description breaks the dialogue flow.
    pending_character: str | None = None
    last_speaker: str | None = None

    for child in scene_element:
        tag = child.tag
        text = _stripped(child.text)
        if tag == "stage_direction":
            if pending_character is not None:
                # Character without dialogue before this stage_direction.
                dialogue_units.append((pending_character, ""))
                warnings.append(
                    f"scene {scene_number}: character {pending_character!r} "
                    "had no following <dialogue>"
                )
                pending_character = None
            last_speaker = None  # action breaks the dialogue flow
            stage_directions.append(text)
        elif tag == "scene_description":
            if pending_character is not None:
                dialogue_units.append((pending_character, ""))
                warnings.append(
                    f"scene {scene_number}: character {pending_character!r} "
                    "had no following <dialogue>"
                )
                pending_character = None
            last_speaker = None  # action breaks the dialogue flow
            scene_descriptions.append(text)
        elif tag == "character":
            # Flush any dangling pending_character first (Case 7 path).
            if pending_character is not None:
                dialogue_units.append((pending_character, ""))
                warnings.append(
                    f"scene {scene_number}: character {pending_character!r} "
                    "followed by another <character>"
                )
                pending_character = None

            # Tier 1.1: normalize trailing variant suffixes.
            normalized = _normalize_character_name(text)

            # Tier 1.2 (conservative): reject implausible character names
            # (copyright headers, year-prefixed strings, studio attributions).
            # When rejected, treat the tag as a flow break so any subsequent
            # dialogue lands on the orphan path (Case 11) rather than being
            # attributed to the implausible "character" or carried over by
            # last_speaker.
            if not _is_plausible_character_name(normalized):
                warnings.append(
                    f"scene {scene_number}: rejected implausible character name "
                    f"{text!r}"
                )
                last_speaker = None  # force Case 11 for the next <dialogue>
                continue

            pending_character = normalized
        elif tag == "parenthetical":
            # Standard screenplay element; not a flow break. Don't reset
            # pending_character or last_speaker. Don't store the
            # parenthetical text (the brief's dialogue_units schema is
            # 2-tuples; parentheticals would expand it).
            pass
        elif tag == "dialogue":
            if pending_character is not None:
                speaker = pending_character
                pending_character = None
            elif last_speaker is not None:
                # Continuation: same speaker, with a parenthetical or
                # similar non-flow-breaking element in between.
                speaker = last_speaker
            else:
                # Genuinely orphaned: no <character> before this and a
                # stage_direction or scene_description has reset the
                # dialogue flow.
                speaker = ""
                warnings.append(
                    f"scene {scene_number}: <dialogue> with no attributable speaker"
                )
            dialogue_units.append((speaker, text))
            last_speaker = speaker if speaker else last_speaker
        else:
            # An unknown tag we haven't seen in the corpus survey.
            # Record but don't fail.
            warnings.append(f"scene {scene_number}: unexpected tag {tag!r}")

    # If the scene ended on a dangling <character>, flush it.
    if pending_character is not None:
        dialogue_units.append((pending_character, ""))
        warnings.append(
            f"scene {scene_number}: scene ended with character {pending_character!r} "
            "and no following <dialogue>"
        )

    return Scene(
        scene_number=scene_number,
        stage_direction=" ".join(s for s in stage_directions if s),
        scene_description=" ".join(s for s in scene_descriptions if s),
        dialogue_units=tuple(dialogue_units),
    )


def _empty_screenplay(imdb_id: str, warning: str) -> ParsedScreenplay:
    """Build a degenerate ParsedScreenplay used when parsing fails."""
    return ParsedScreenplay(
        imdb_id=imdb_id,
        scenes=(),
        parse_warnings=(warning,),
        n_scenes=0,
        n_unique_characters=0,
        n_dialogue_lines=0,
        total_dialogue_chars=0,
        total_stage_direction_chars=0,
        total_scene_description_chars=0,
        total_action_chars=0,
        dialogue_to_action_ratio=0.0,
        dialogue_to_total_text_ratio=0.0,
    )


def parse_screenplay(xml_string: str, imdb_id: str) -> ParsedScreenplay:
    """Parse a MovieSum XML screenplay into a ``ParsedScreenplay``.

    Parameters
    ----------
    xml_string
        The full screenplay XML, as stored in MovieSum's ``script``
        field.
    imdb_id
        The IMDb ID for this screenplay; included in the output for
        traceability.

    Returns
    -------
    ParsedScreenplay
        Frozen dataclass with the parsed scenes, structural metrics,
        and any warnings. Never raises on bad input — malformed XML
        produces a degenerate object with the error in
        ``parse_warnings``.
    """
    if not xml_string or not xml_string.strip():
        return _empty_screenplay(imdb_id, "empty XML string")

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        logger.warning("XML parse failed for %s: %s", imdb_id, exc)
        return _empty_screenplay(imdb_id, f"XML parse error: {exc}")

    warnings: list[str] = []
    if root.tag != "script":
        logger.warning("Root tag for %s is %r, expected 'script'", imdb_id, root.tag)
        # Persist the anomaly on the dataclass so downstream audits can
        # see it (consistency with the empty-XML and parse-error paths,
        # which both record their issue in parse_warnings). Continue
        # parsing because the mislabeled root may still contain valid
        # <scene> children.
        warnings.append(f"unexpected root tag {root.tag!r}")

    scenes: list[Scene] = []
    for i, scene_el in enumerate(root.findall("scene"), start=1):
        scenes.append(_parse_one_scene(scene_el, scene_number=i, warnings=warnings))

    return _summarize(imdb_id, scenes, warnings)


def _summarize(
    imdb_id: str, scenes: list[Scene], warnings: list[str]
) -> ParsedScreenplay:
    """Compute the structural metrics from a list of parsed scenes."""
    n_scenes = len(scenes)
    n_dialogue_lines = sum(len(s.dialogue_units) for s in scenes)
    total_dialogue_chars = sum(
        len(text) for s in scenes for _, text in s.dialogue_units
    )
    total_stage_direction_chars = sum(len(s.stage_direction) for s in scenes)
    total_scene_description_chars = sum(len(s.scene_description) for s in scenes)
    total_action_chars = total_stage_direction_chars + total_scene_description_chars

    # Tier 1.3: a character only counts as "unique" if they delivered at
    # least one non-empty, non-whitespace dialogue line. Empty-text
    # placeholders (Cases 5, 6, 7, 8) are excluded from the count even
    # though they remain in dialogue_units for traceability.
    unique_characters = {
        name for s in scenes for name, text in s.dialogue_units
        if name and text.strip()
    }
    n_unique_characters = len(unique_characters)

    # Brief's literal formula: dialogue / (dialogue + stage_direction).
    if total_dialogue_chars + total_stage_direction_chars > 0:
        dialogue_to_action_ratio = total_dialogue_chars / (
            total_dialogue_chars + total_stage_direction_chars
        )
    else:
        dialogue_to_action_ratio = 0.0

    # More informative variant: dialogue / (dialogue + all non-dialogue text).
    if total_dialogue_chars + total_action_chars > 0:
        dialogue_to_total_text_ratio = total_dialogue_chars / (
            total_dialogue_chars + total_action_chars
        )
    else:
        dialogue_to_total_text_ratio = 0.0

    return ParsedScreenplay(
        imdb_id=imdb_id,
        scenes=tuple(scenes),
        parse_warnings=tuple(warnings),
        n_scenes=n_scenes,
        n_unique_characters=n_unique_characters,
        n_dialogue_lines=n_dialogue_lines,
        total_dialogue_chars=total_dialogue_chars,
        total_stage_direction_chars=total_stage_direction_chars,
        total_scene_description_chars=total_scene_description_chars,
        total_action_chars=total_action_chars,
        dialogue_to_action_ratio=dialogue_to_action_ratio,
        dialogue_to_total_text_ratio=dialogue_to_total_text_ratio,
    )


def parse_many(
    items: Iterable[tuple[str, str]],
) -> dict[str, ParsedScreenplay]:
    """Parse a batch of screenplays.

    Parameters
    ----------
    items
        Iterable of ``(imdb_id, xml_string)`` pairs.

    Returns
    -------
    dict[str, ParsedScreenplay]
        Keyed by ``imdb_id``. If the same ID appears twice in
        ``items`` the second one wins (caller should dedupe upstream).
    """
    out: dict[str, ParsedScreenplay] = {}
    for imdb_id, xml in items:
        out[imdb_id] = parse_screenplay(xml, imdb_id)
    n_with_warnings = sum(1 for v in out.values() if v.parse_warnings)
    logger.info(
        "Parsed %s screenplays (%s with warnings)",
        f"{len(out):,}", f"{n_with_warnings:,}",
    )
    return out
