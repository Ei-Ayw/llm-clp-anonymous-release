"""Command-line and YAML configuration helpers.

The public entry points use a small two-pass parser: first read ``--config``
from the command line, then apply YAML values as parser defaults, and finally
parse the full argument list. Explicit CLI arguments still override values in
configuration files because ``argparse`` sees them during the second pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

CONFIG_ALIASES: Mapping[str, str] = {
    "ckpt": "checkpoint",
    "data": "data_dir",
    "input": "input_path",
    "model": "model_name",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and validate that it contains a top-level mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    return dict(data)


def normalize_config_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map common short aliases to the internal argument names."""
    return {CONFIG_ALIASES.get(key, key): value for key, value in config.items()}


def parse_args_with_config(
        parser: argparse.ArgumentParser,
        argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    """Parse arguments after applying defaults from an optional YAML config."""
    args, _ = parse_known_args_with_config(parser, argv)
    return args


def parse_known_args_with_config(
        parser: argparse.ArgumentParser,
        argv: Iterable[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse known arguments and preserve unknown passthrough flags."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    if config_args.config:
        defaults = normalize_config_keys(load_yaml_config(config_args.config))
        parser.set_defaults(**defaults)

    return parser.parse_known_args(argv)


def apply_config_defaults(
        parser: argparse.ArgumentParser,
        argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    """Backward-compatible wrapper around :func:`parse_args_with_config`."""
    return parse_args_with_config(parser, argv)


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return that path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def namespace_to_cli(
        args: argparse.Namespace,
        exclude: Iterable[str] | None = None,
) -> list[str]:
    """Convert selected namespace values into underscore-style CLI flags."""
    excluded = set(exclude or [])
    cli: list[str] = []

    for key, value in vars(args).items():
        if key in excluded or key.startswith("_") or value is None:
            continue

        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            for item in value:
                cli.extend([flag, str(item)])
            continue

        cli.extend([flag, str(value)])

    return cli


__all__ = [
    "apply_config_defaults",
    "ensure_parent",
    "load_yaml_config",
    "namespace_to_cli",
    "normalize_config_keys",
    "parse_args_with_config",
    "parse_known_args_with_config",
]
