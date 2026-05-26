"""Repository path constants and output path helpers."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
EVAL_DIR = OUTPUT_DIR / "eval"

_OUTPUT_DIRS = (OUTPUT_DIR, MODEL_DIR, LOG_DIR, EVAL_DIR)


def ensure_output_dirs() -> None:
    """Create the default output directories if they do not already exist."""
    for directory in _OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def model_path(filename: str) -> str:
    """Return a path under ``outputs/models``."""
    ensure_output_dirs()
    return str(MODEL_DIR / filename)


def log_path(filename: str) -> str:
    """Return a path under ``outputs/logs``."""
    ensure_output_dirs()
    return str(LOG_DIR / filename)


def eval_path(filename: str) -> str:
    """Return a path under ``outputs/eval``."""
    ensure_output_dirs()
    return str(EVAL_DIR / filename)


__all__ = [
    "EVAL_DIR",
    "LOG_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "ROOT_DIR",
    "ensure_output_dirs",
    "eval_path",
    "log_path",
    "model_path",
]
