"""Logging helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def get_logger(name: str = "fxmaster", level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Return a logger configured to write both to stdout and (optionally) a file."""
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger
