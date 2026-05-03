"""Project path resolution.

Resolves the project root by walking up from this file's location until a
marker is found (the ``docs/`` directory containing ``PROJECT_CONTEXT.*``),
then exposes well-known subdirectories as ``pathlib.Path`` objects.

All other modules should import paths from here rather than constructing
them ad-hoc, so the project remains runnable from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# Filenames accepted as the project-root marker. The foundation docs live
# in ``docs/`` and are saved as ``.txt`` in this checkout, but their internal
# references use ``.md``; accept either so path resolution survives a rename.
_ROOT_MARKERS: tuple[str, ...] = (
    "docs/PROJECT_CONTEXT.md",
    "docs/PROJECT_CONTEXT.txt",
)


def _find_project_root(start: Path) -> Path:
    """Walk upward from ``start`` until a marker file is found.

    Parameters
    ----------
    start
        Directory to start searching from. Typically ``Path(__file__).parent``.

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    RuntimeError
        If no marker is found before reaching the filesystem root.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).is_file() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(
        f"Could not locate project root from {start!s}; "
        f"none of {_ROOT_MARKERS} were found in any parent directory."
    )


PROJECT_ROOT: Path = _find_project_root(Path(__file__).parent)

# Top-level project directories.
SRC_DIR: Path = PROJECT_ROOT / "src"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
DOCS_SUMMARIES_DIR: Path = DOCS_DIR / "summaries"
DOCS_BRIEFS_DIR: Path = DOCS_DIR / "briefs"

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR: Path = REPORTS_DIR / "figures"
REPORTS_TABLES_DIR: Path = REPORTS_DIR / "tables"

# Per-run experiment tracking. ``runs/<phase>/<YYYYMMDD_HHMM>_<name>/`` per
# run, populated by ``src.experiments.save_run``. The phase subdirectory and
# the per-run directory are created on demand by ``save_run``; this constant
# is the parent that holds the ``RUNS.md`` index.
RUNS_DIR: Path = PROJECT_ROOT / "runs"


def ensure_dirs() -> None:
    """Create the standard data/report directories if they don't already exist.

    Safe to call repeatedly; existing directories are left untouched. Useful
    when a script is invoked on a fresh checkout where only ``raw`` was
    populated by hand.
    """
    for directory in (
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_TABLES_DIR,
        DOCS_SUMMARIES_DIR,
        RUNS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
