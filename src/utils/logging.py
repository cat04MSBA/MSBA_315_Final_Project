"""Project-wide logging configuration.

Modules import the configured logger via ``get_logger(__name__)`` rather
than calling ``logging.getLogger`` directly, so the project's stream
handler and format are reliably installed on first import.

**Where logs go:** to ``sys.stdout`` (not stderr). This deviates from the
conventional library default deliberately so Jupyter renders log lines
as ordinary black-text output rather than red-stderr boxes. **Logs are
NOT written to a file** — there is no file handler installed.

**Default level:** ``INFO``. INFO messages show main pipeline milestones
(loaded X rows, dedup X→Y, joined N films, saved K artifacts). For the
fine-grained per-item trace (which file was saved where, etc.) opt in
to ``DEBUG``. To silence operational chatter, set the level to
``WARNING``. Three ways to change the level:

1. **Environment variable** (before importing): set
   ``MSBA315_LOG_LEVEL=INFO`` in your shell. Requires a fresh Python
   process / fresh Jupyter kernel.
2. **Runtime call** (works in a live Jupyter kernel): from a notebook
   cell, ``from src.utils.logging import set_log_level;
   set_log_level("INFO")``. Takes effect immediately, no restart needed.
3. **Direct**: ``import logging; logging.getLogger().setLevel(logging.INFO)``.

Format string (``_DEFAULT_FORMAT`` below) — the four pipe-separated
fields you see in every log line:

    YYYY-MM-DD HH:MM:SS | LEVEL | module.name | message
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: int | str | None) -> int:
    """Convert a level (str / int / None) into a ``logging`` integer.

    ``None`` reads the ``MSBA315_LOG_LEVEL`` env var (default WARNING).
    Strings are looked up case-insensitively against logging constants.
    """
    if level is None:
        env_level = os.environ.get("MSBA315_LOG_LEVEL", "INFO").upper()
        return getattr(logging, env_level, logging.INFO)
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.WARNING)
    return int(level)


def configure_logging(level: int | str | None = None) -> None:
    """Install the project's stdout handler (idempotent) and apply level.

    Always re-applies the level — so calling this from a notebook (or
    via :func:`set_log_level`) works even after the handler is already
    in place from an earlier import in the same kernel session.

    Parameters
    ----------
    level
        New root-logger level. ``None`` means "read from env var, fall
        back to WARNING".
    """
    root = logging.getLogger()
    if not any(getattr(h, "_msba315_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        handler._msba315_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    # Always re-apply the level. This is what fixes the "I changed the
    # default but my Jupyter kernel still shows INFO" surprise.
    root.setLevel(_resolve_level(level))


def set_log_level(level: int | str) -> None:
    """Change logging verbosity at runtime; works inside a live Jupyter kernel.

    Parameters
    ----------
    level
        ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``,
        or the equivalent ``logging`` integer constant.

    Examples
    --------
    >>> from src.utils.logging import set_log_level
    >>> set_log_level("INFO")    # opt in to per-step progress messages
    >>> set_log_level("WARNING") # silence them again
    """
    configure_logging(level=level)


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """Return a project-configured logger named ``name``.

    Configures the root logger on first call. Subsequent calls return
    the requested named logger; pass ``level`` to set this individual
    logger's threshold (rare; usually you want :func:`set_log_level`).
    """
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_resolve_level(level))
    return logger
