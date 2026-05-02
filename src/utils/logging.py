"""Project-wide logging configuration.

Modules import the configured logger via ``get_logger(__name__)`` rather
than calling ``logging.getLogger`` directly, so that the first import in
any process reliably installs a stream handler with the project's format.

Per ``CLAUDE_CODE_GUIDELINES.txt`` Section 3, no diagnostic ``print()``
calls are used; everything goes through this logger.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger(level: int) -> None:
    """Install a single stderr stream handler on the root logger.

    Idempotent: a sentinel attribute on the handler prevents duplicate
    installation if multiple modules call ``get_logger`` independently.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_msba315_handler", False):
            return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
    handler._msba315_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name
        Usually ``__name__`` of the calling module.
    level
        Optional override for the logger level. If ``None``, the level is
        taken from the ``MSBA315_LOG_LEVEL`` env var (default: ``INFO``).

    Returns
    -------
    logging.Logger
        Logger ready for use; the project root handler is installed on first
        call and re-used thereafter.
    """
    global _configured
    if not _configured:
        env_level = os.environ.get("MSBA315_LOG_LEVEL", "INFO").upper()
        resolved_level = level if level is not None else getattr(logging, env_level, logging.INFO)
        _configure_root_logger(resolved_level)
        _configured = True

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
