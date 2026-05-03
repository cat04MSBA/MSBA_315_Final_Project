"""Per-run experiment logging for the modelling phases.

Wraps each experiment with three things: (1) a timestamped run directory
under ``runs/<phase>/``, (2) a temporary file handler attached to the
project's root logger so the per-run ``run.log`` captures the full
INFO/DEBUG trace of what happened, (3) a small set of metadata JSON
files written at entry time and an optional ``metrics.json`` and
``model.joblib`` written during the run.

Console output during the run stays at whatever level the user has
currently configured via :func:`src.utils.logging.set_log_level` (default
INFO). The run-log file always captures DEBUG and above regardless of
the console level. On exit the file handler is detached cleanly so
subsequent code does not write into that run's log.

Usage
-----

::

    from src.experiments.save_run import save_run

    with save_run(
        phase="phase_3",
        name="lexical_first",
        params={"alpha_grid": [0.1, 1.0, 10.0], "fit_intercept": True},
        preprocessing={"split": "70_15_15", "log_transform_structural": True},
        features=["log_n_scenes", "mtld_dialogue", "..."],
    ) as run:
        # ... train and evaluate inside the block ...
        run.record_metrics({"r2": 0.07, "mae": 0.93, "auc_roi_gt_2": 0.62})
        # run.save_model(model)  # optional; Phase 4+
        run.append_to_runs_md(
            model_family="Ridge + LogisticRegression",
            features_group="structural + lexical",
            key_metric="R2 0.07, AUC roi_gt_2 0.62",
            notes="First lexical ablation row",
        )

The block writes ``runs/phase_3/<YYYYMMDD_HHMM>_lexical_first/`` with
``params.json``, ``preprocessing_summary.json``, ``features_used.json``,
``metrics.json``, and ``run.log``. ``model.joblib`` is written only if
:meth:`RunHandle.save_model` is called.

Design notes
------------

* The phase-prefix and run-name conventions match the original
  experiment-tracking design in ``CLAUDE_CODE_GUIDELINES.md`` Section 7.
* JSON metadata files are split (rather than one monolithic config) so
  cross-run diffs are easy to read. "What features did the run that
  beat the baseline use?" is just ``diff features_used.json``.
* No MLflow. For this project's scale the directory-based scheme is
  clearer, has zero install overhead, and is git-trackable.
* Nested ``save_run`` blocks are not supported and will silently
  produce confusing logs; the caller is expected to keep one run per
  block.
"""

from __future__ import annotations

import json
import logging
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.utils import paths
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Per-run log format. Matches the project's standard format (defined in
# ``src/utils/logging.py``) so the file output is consistent with stdout.
_RUN_LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_RUN_LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Marker comment in ``runs/RUNS.md`` that ``append_to_runs_md`` inserts
# new rows above. Lets the index stay sorted newest-first without
# re-parsing markdown.
_RUNS_MD_MARKER: str = "<!-- new rows above this line -->"


@dataclass(frozen=True)
class RunConfig:
    """Frozen container for the per-run metadata supplied at block entry.

    Each field is written to a dedicated JSON file in the run directory
    so cross-run diffs are clean.
    """
    phase: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class RunHandle:
    """Live handle yielded by :func:`save_run`.

    Methods on the handle write artifacts to the run directory while the
    block is active. After the block exits the handle is marked closed
    and any further writes raise :class:`RuntimeError` to prevent stray
    output from landing in the wrong run.
    """

    run_dir: Path
    cfg: RunConfig
    _metrics: dict[str, Any] | None = None
    _closed: bool = False

    @property
    def params(self) -> dict[str, Any]:
        """Convenience accessor for the params dict supplied at entry."""
        return dict(self.cfg.params)

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Record metrics for the run.

        Writes ``metrics.json`` immediately so the file survives even
        if the block raises before exit. Subsequent calls overwrite.
        """
        self._check_open("record_metrics")
        self._metrics = dict(metrics)
        _write_json(self.run_dir / "metrics.json", self._metrics)

    def save_model(self, model: Any) -> Path:
        """Persist a fitted model object to ``model.joblib`` via joblib.

        Returns the absolute path the model was saved to. ``joblib`` is
        the preferred serialization for sklearn-style fitted estimators;
        for non-sklearn objects, callers can pickle separately and pass
        the path back through ``record_metrics`` if needed.
        """
        self._check_open("save_model")
        try:
            import joblib  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "joblib is required for save_model; install with "
                "`pip install joblib`"
            ) from exc
        out = self.run_dir / "model.joblib"
        joblib.dump(model, out)
        return out

    def append_to_runs_md(
        self,
        model_family: str,
        features_group: str,
        key_metric: str,
        notes: str = "",
    ) -> None:
        """Append a one-line row to ``runs/RUNS.md``, newest-first.

        The row is inserted directly above the marker comment so the
        index remains sorted by run-time without re-parsing the file.
        """
        self._check_open("append_to_runs_md")
        runs_md = paths.RUNS_DIR / "RUNS.md"
        if not runs_md.is_file():
            raise RuntimeError(
                f"runs/RUNS.md missing at {runs_md}; expected to be created "
                "alongside the runs/ directory"
            )
        text = runs_md.read_text(encoding="utf-8")
        if _RUNS_MD_MARKER not in text:
            raise RuntimeError(
                f"runs/RUNS.md is missing the {_RUNS_MD_MARKER!r} marker; "
                "cannot determine where to insert new rows"
            )

        rel_dir = self.run_dir.relative_to(paths.PROJECT_ROOT).as_posix() + "/"
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        sha = _git_sha_short()
        row = (
            f"| {date} | {self.cfg.phase} | `{rel_dir}` | `{sha}` | "
            f"{model_family} | {features_group} | {key_metric} | {notes} |"
        )
        before, after = text.split(_RUNS_MD_MARKER, 1)
        new_text = before + row + "\n" + _RUNS_MD_MARKER + after
        runs_md.write_text(new_text, encoding="utf-8")

    def _check_open(self, method: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"RunHandle is closed; {method!r} cannot be called outside "
                "the save_run() with-block"
            )


def _write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty-printed JSON to ``path``.

    Uses ``default=str`` so values that aren't natively serializable
    (Path, set, datetime) get a string representation rather than raising.
    """
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _git_sha_short() -> str:
    """Return the short git SHA for ``HEAD``, or ``"<unknown>"`` if unavailable.

    Defensive: a missing ``git`` binary, a non-git checkout, or a
    detached HEAD all return the unknown sentinel rather than raise.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "<unknown>"


@contextmanager
def save_run(
    phase: str,
    name: str,
    params: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    features: list[str] | None = None,
    notes: str = "",
) -> Iterator[RunHandle]:
    """Context manager wrapping one experiment run with logging and metadata.

    On entry: creates ``runs/<phase>/<YYYYMMDD_HHMM>_<name>/``, writes
    ``params.json``, ``preprocessing_summary.json``, and
    ``features_used.json``, and attaches a DEBUG-level FileHandler to
    the root logger so every ``logger.info`` and ``logger.debug`` call
    inside the block is preserved in ``run.log``.

    On exit: detaches the file handler, restores the previous root and
    stdout-handler levels, and marks the handle closed.

    Console output during the block stays at whatever level the user
    has configured (typically INFO). Inside the block, the root logger
    is bumped to DEBUG so DEBUG records reach the file handler; the
    stdout handler's own level is pinned at the prior root level so
    DEBUG records don't also leak to the console.

    Parameters
    ----------
    phase
        Phase identifier (e.g. ``"phase_3"`` or ``"phase_4"``). Used as
        the parent directory under ``runs/``.
    name
        Short slug describing the run (e.g.
        ``"ridge_baseline_with_log_budget"``).
    params
        Hyperparameters of the run. Keys and values must be
        JSON-serializable (or have a useful ``str()`` representation).
    preprocessing
        Pipeline choices upstream of the model (split policy, transforms,
        feature-engineering toggles).
    features
        List of feature column names fed to the model.
    notes
        Optional free-text notes string. Stored in ``params.json`` under
        the reserved ``"_notes"`` key for retrievability.

    Yields
    ------
    RunHandle
        Handle for recording metrics, saving the model, and appending to
        the global runs index. Methods raise :class:`RuntimeError` if
        called after the block exits.
    """
    cfg = RunConfig(
        phase=phase,
        name=name,
        params=dict(params or {}),
        preprocessing=dict(preprocessing or {}),
        features=list(features or []),
        notes=notes,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = paths.RUNS_DIR / phase / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write entry-time metadata. Notes ride along inside params.json under
    # a reserved key so the JSON file count stays at the documented six.
    params_payload = dict(cfg.params)
    if cfg.notes:
        params_payload["_notes"] = cfg.notes
    _write_json(run_dir / "params.json", params_payload)
    _write_json(run_dir / "preprocessing_summary.json", cfg.preprocessing)
    _write_json(run_dir / "features_used.json", cfg.features)

    # Locate and snapshot the existing stdout handler so we can pin its
    # level at the prior root level for the duration of the block, then
    # restore on exit.
    root = logging.getLogger()
    prior_root_level = root.level
    stdout_handler = next(
        (h for h in root.handlers if getattr(h, "_msba315_handler", False)),
        None,
    )
    prior_stdout_level = stdout_handler.level if stdout_handler is not None else None
    if stdout_handler is not None:
        # Pin stdout to the user's prior level so DEBUG records that we're
        # about to start emitting don't also flood the console.
        stdout_handler.setLevel(prior_root_level)

    file_handler = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_RUN_LOG_FORMAT, _RUN_LOG_DATEFMT))
    root.addHandler(file_handler)
    root.setLevel(logging.DEBUG)

    handle = RunHandle(run_dir=run_dir, cfg=cfg)
    rel_dir = run_dir.relative_to(paths.PROJECT_ROOT).as_posix()
    logger.info("Run started: phase=%s name=%s dir=%s", phase, name, rel_dir)

    try:
        yield handle
    finally:
        logger.info("Run finished: phase=%s name=%s dir=%s", phase, name, rel_dir)
        # Restore logger state before any subsequent code runs.
        root.removeHandler(file_handler)
        file_handler.close()
        root.setLevel(prior_root_level)
        if stdout_handler is not None:
            stdout_handler.setLevel(prior_stdout_level if prior_stdout_level is not None else logging.NOTSET)
        handle._closed = True
