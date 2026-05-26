"""Logging setup utilities."""

from __future__ import annotations

import logging
import sys

from .io import PathLike, ensure_dir

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def setup_logger(
        name: str = "llm_clp",
        output_dir: PathLike | None = None,
        level: int = logging.INFO,
        format_str: str = DEFAULT_LOG_FORMAT,
) -> logging.Logger:
    """Create a console logger and, optionally, a file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_dir is not None:
        log_dir = ensure_dir(output_dir)
        file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


__all__ = ["DEFAULT_LOG_FORMAT", "setup_logger"]
