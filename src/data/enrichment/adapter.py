"""Adapter: unified_scripts.jsonl record → ParsedScreenplay.

Maps the user's flat scenes/elements format (produced by
``MovieScripts.ipynb``) onto the v1 ``ParsedScreenplay`` dataclass so
the rest of the pipeline (Phase 3 features, Phase 4 models) keeps
working unchanged.

Mapping per scene:

* ``scene.heading`` → ``Scene.stage_direction`` (the slugline,
  matches v1's MovieSum convention).
* All ``elements[type='action'].text`` concatenated → ``Scene.scene_description``.
* Walk the elements list in order, tracking the current speaker via
  ``character`` elements (with parenthetical as continuation marker),
  to reconstruct ``Scene.dialogue_units = [(speaker, text), ...]``.

The same Tier 1.1 character-name normalization and Tier 1.2
implausible-name filter as v1's parser are applied via the helpers in
``src.data.parse_screenplay``. Structural metrics are computed by the
same ``_summarize`` function v1 uses, so v1 vs v2 metrics are
comparable on the overlap.

Element ``type`` values seen in the v1 corpus survey:
``action``, ``character``, ``dialogue``, ``parenthetical``. Any other
value is recorded as a parse warning but does not break parsing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.data.parse_screenplay import (
    ParsedScreenplay,
    Scene,
    _is_plausible_character_name,
    _normalize_character_name,
    _stripped,
    _summarize,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Element ``type`` values we know how to handle. Anything else gets
# logged as a warning and ignored.
_KNOWN_TYPES = frozenset({"action", "character", "dialogue", "parenthetical"})


def _convert_one_scene(
    scene: dict, scene_number: int, warnings: list[str]
) -> Scene:
    """Walk one unified-format scene's elements, build a v1 Scene."""
    heading = _stripped(scene.get("heading"))
    elements = scene.get("elements") or []

    scene_description_parts: list[str] = []
    dialogue_units: list[tuple[str, str]] = []
    pending_character: str | None = None
    last_speaker: str | None = None

    for el in elements:
        et = el.get("type")
        if et not in _KNOWN_TYPES:
            warnings.append(f"scene {scene_number}: unknown element type {et!r}")
            continue

        if et == "action":
            text = _stripped(el.get("text"))
            if pending_character is not None:
                dialogue_units.append((pending_character, ""))
                warnings.append(
                    f"scene {scene_number}: character {pending_character!r} "
                    "had no following dialogue"
                )
                pending_character = None
            last_speaker = None  # action breaks the dialogue flow
            scene_description_parts.append(text)

        elif et == "character":
            # Flush dangling pending_character (Case 7 path).
            if pending_character is not None:
                dialogue_units.append((pending_character, ""))
                warnings.append(
                    f"scene {scene_number}: character {pending_character!r} "
                    "followed by another character"
                )
                pending_character = None

            raw_name = _stripped(el.get("name"))
            normalized = _normalize_character_name(raw_name)

            if not _is_plausible_character_name(normalized):
                warnings.append(
                    f"scene {scene_number}: rejected implausible character name "
                    f"{raw_name!r}"
                )
                last_speaker = None  # force orphan path on next dialogue
                continue
            pending_character = normalized

        elif et == "parenthetical":
            # Continuation marker; do not break dialogue flow, do not
            # store text (matching v1's 2-tuple dialogue schema).
            pass

        elif et == "dialogue":
            text = _stripped(el.get("text"))
            if pending_character is not None:
                speaker = pending_character
                pending_character = None
            elif last_speaker is not None:
                speaker = last_speaker
            else:
                speaker = ""
                warnings.append(
                    f"scene {scene_number}: dialogue with no attributable speaker"
                )
            dialogue_units.append((speaker, text))
            last_speaker = speaker if speaker else last_speaker

    # Flush a dangling pending_character at scene end.
    if pending_character is not None:
        dialogue_units.append((pending_character, ""))
        warnings.append(
            f"scene {scene_number}: scene ended with character "
            f"{pending_character!r} and no following dialogue"
        )

    return Scene(
        scene_number=scene_number,
        stage_direction=heading,
        scene_description=" ".join(s for s in scene_description_parts if s),
        dialogue_units=tuple(dialogue_units),
    )


def convert_to_parsed_screenplay(record: dict) -> ParsedScreenplay:
    """Convert one unified_scripts.jsonl record into a ParsedScreenplay.

    Never raises. Empty / missing scenes produce an empty ParsedScreenplay
    with the issue captured in ``parse_warnings``.
    """
    imdb_id = record["imdb_id"]
    raw_scenes = record.get("scenes") or []
    warnings: list[str] = []

    if not raw_scenes:
        warnings.append("unified record has no scenes")

    scenes: list[Scene] = []
    for i, sc in enumerate(raw_scenes, start=1):
        scenes.append(_convert_one_scene(sc, scene_number=i, warnings=warnings))

    return _summarize(imdb_id, scenes, warnings)


def convert_many(
    records: dict[str, dict],
) -> dict[str, ParsedScreenplay]:
    """Convert a dict of unified records to a dict of ParsedScreenplay."""
    out: dict[str, ParsedScreenplay] = {}
    for imdb_id, rec in records.items():
        out[imdb_id] = convert_to_parsed_screenplay(rec)
    n_warn = sum(1 for p in out.values() if p.parse_warnings)
    logger.info(
        "Converted %s unified records (%s with warnings)",
        f"{len(out):,}", f"{n_warn:,}",
    )
    return out


# ---------------------------------------------------------------------------
# Validation: compare adapter output against v1's parses on the overlap
# ---------------------------------------------------------------------------

def validate_against_v1(
    unified: dict[str, dict],
    v1_parsed_pkl: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the adapter on a sample of v1 IDs, compare structural metrics.

    Returns a DataFrame with one row per sampled film and columns
    ``imdb_id, n_scenes_v1, n_scenes_v2, abs_pct_err_*`` for each
    structural metric. The caller logs the median and 95th-percentile
    abs % error and decides whether to proceed.
    """
    import random

    with v1_parsed_pkl.open("rb") as f:
        v1: dict[str, ParsedScreenplay] = pickle.load(f)

    overlap = sorted(set(v1.keys()) & set(unified.keys()))
    rng = random.Random(seed)
    sample_ids = rng.sample(overlap, min(sample_size, len(overlap)))

    rows: list[dict] = []
    for imdb_id in sample_ids:
        v1p = v1[imdb_id]
        v2p = convert_to_parsed_screenplay(unified[imdb_id])
        row = {"imdb_id": imdb_id}
        for metric in (
            "n_scenes",
            "n_unique_characters",
            "n_dialogue_lines",
            "total_dialogue_chars",
            "total_action_chars",
        ):
            v1v = getattr(v1p, metric)
            v2v = getattr(v2p, metric)
            row[f"{metric}_v1"] = v1v
            row[f"{metric}_v2"] = v2v
            row[f"{metric}_abs_pct_err"] = (
                100.0 * abs(v2v - v1v) / max(v1v, 1)
            )
        rows.append(row)

    return pd.DataFrame(rows)
