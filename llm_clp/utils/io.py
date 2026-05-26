"""Consistent file-system and tabular-data I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import pandas as pd

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: PathLike) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Mapping[str, Any], path: PathLike, **kwargs: Any) -> None:
    """Write a mapping to a UTF-8 JSON file."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str, **kwargs)


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: PathLike) -> None:
    """Write dictionaries to a UTF-8 JSON Lines file."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_parquet(path: PathLike, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file."""
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: PathLike, **kwargs: Any) -> None:
    """Write a DataFrame to Parquet without an index."""
    target = Path(path)
    ensure_dir(target.parent)
    df.to_parquet(target, index=False, **kwargs)


__all__ = [
    "PathLike",
    "ensure_dir",
    "read_json",
    "read_jsonl",
    "read_parquet",
    "write_json",
    "write_jsonl",
    "write_parquet",
]
