from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

from src.data.load_moviesum import load_moviesum
from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# 🔥 Increased preview size for deeper inspection
PREVIEW_CHARS = 5000
TAIL_CHARS = 5000
SUMMARY_HEAD_CHARS = 1000

# 🔥 Focus only on specific duplicate IMDb IDs
FOCUS_IDS = ["tt0175526", "tt1022603", "tt5536736"]


def _script_chunk(xml_string: str, n: int, mode: str = "head") -> str:
    if not xml_string:
        return ""
    chunk = xml_string[:n] if mode == "head" else xml_string[-n:]
    return chunk.replace("\n", " ⏎ ").strip()


def _structural_counts(xml_string: str) -> dict[str, int]:
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        logger.warning("XML parse failed: %s", exc)
        return {"_parse_error": 1}

    counts: Counter[str] = Counter()
    counts["scenes"] = len(root.findall("scene"))
    for tag in ("stage_direction", "scene_description", "character", "dialogue"):
        counts[tag + "s"] = sum(len(s.findall(tag)) for s in root.findall("scene"))
    counts["unique_characters"] = len({
        c.text for s in root.findall("scene")
        for c in s.findall("character") if c.text
    })
    return dict(counts)


def _format_structural_line(counts: dict[str, int]) -> str:
    if "_parse_error" in counts:
        return "structure: <XML parse error>"
    return (
        f"structure: {counts.get('scenes', 0)} scenes, "
        f"{counts.get('unique_characters', 0)} unique characters, "
        f"{counts.get('dialogues', 0)} dialogue lines, "
        f"{counts.get('stage_directions', 0)} stage directions"
    )


def main() -> pd.DataFrame:
    paths.ensure_dirs()
    df = load_moviesum(include_script=True)

    dups = df[df.duplicated(subset="imdb_id", keep=False)].copy()

    # 🔥 Apply filter here
    dups = dups[dups["imdb_id"].isin(FOCUS_IDS)].copy()

    dups = dups.sort_values(
        ["imdb_id", "script_char_len"], ascending=[True, False]
    ).reset_index(drop=True)

    if dups.empty:
        print("No matching duplicate IMDb IDs found.")
        return dups

    print(f"Reviewing {dups['imdb_id'].nunique()} IMDb IDs "
          f"({len(dups)} rows total).\n")

    for imdb_id, group in dups.groupby("imdb_id", sort=False):
        print("=" * 90)
        print(f"IMDb ID: {imdb_id}    ({len(group)} rows)")
        print("=" * 90)

        for i, (_, row) in enumerate(group.iterrows(), start=1):
            counts = _structural_counts(row["script"] or "")

            print(
                f"\n  [{i}] {row['movie_name']!r}\n"
                f"      year={row['year_in_title']}, split={row['origin_split']}, "
                f"length={row['script_char_len']:,} chars"
            )
            print(f"      {_format_structural_line(counts)}")

            summary_head = (row["summary"] or "")[:SUMMARY_HEAD_CHARS].replace("\n", " ⏎ ").strip()
            if summary_head:
                print(f"\n      summary head:")
                print(f"        {summary_head} ...")

            print(f"\n      script head:")
            print(f"        {_script_chunk(row['script'], PREVIEW_CHARS, 'head')} ...")

            print(f"\n      script tail:")
            print(f"        ... {_script_chunk(row['script'], TAIL_CHARS, 'tail')}")
            print()

    return dups


if __name__ == "__main__":
    # 🔥 Redirect output directly to Downloads
    output_path = Path.home() / "Downloads" / "focused_duplicates_review.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        main()
        sys.stdout = sys.__stdout__

    print(f"Saved detailed review to: {output_path}")