"""Structured logging for the project.

Use `get_logger(__name__)` at the top of every module. Level is controlled by
the `LOG_LEVEL` environment variable (default: INFO). Format includes timestamp,
module, level, and message.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def _configure_root_once() -> None:
    """Attach a single stderr StreamHandler to the root logger. Idempotent."""
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Strip any pre-existing handlers (e.g. Jupyter) to avoid duplicate lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Quiet noisy third-party libs.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name; pass `__name__` from the calling module.

    Returns:
        A `logging.Logger` instance using the project-wide format.
    """
    _configure_root_once()
    return logging.getLogger(name)


if __name__ == "__main__":
    log = get_logger("logger.smoketest")
    log.debug("debug example")
    log.info("info example")
    log.warning("warning example")
    log.error("error example")
