#!/usr/bin/env python3
"""Evaluate all LLM-CLP checkpoints under LLM-CF and swap-CF sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

MODELS_DIR = Path("outputs/models")
EVAL_DIR = Path("outputs/eval")
DATA_DIR = Path("data/causal_fair")
DATASETS = {"hatexplain", "toxigen", "dynahate"}


def parse_checkpoint(ckpt_path: Path) -> tuple[str | None, int | str, str]:
    """Return dataset, seed, and model name stored in an LLM-CLP checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    dataset = args.get("dataset")
    seed = args.get("seed", "unknown")
    model_name = args.get("model_name", "microsoft/deberta-v3-base")

    if not dataset:
        parts = ckpt_path.stem.split("_")
        dataset = next((part for part in parts if part in DATASETS), None)
    return dataset, seed, model_name


def run_eval(ckpt: Path, dataset: str, model_name: str, cf_method: str) -> int:
    output_path = EVAL_DIR / f"{ckpt.stem}_{cf_method}_metrics.json"
    if output_path.exists():
        print(f"SKIP existing {output_path.name}")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "llm_clp.validation.validate_llm_clp",
        "--dataset",
        dataset,
        "--checkpoint",
        str(ckpt),
        "--cf_method",
        cf_method,
        "--data_dir",
        str(DATA_DIR),
        "--model_name",
        model_name,
        "--output",
        str(output_path),
    ]
    print(f"EVAL {ckpt.name} | dataset={dataset} | cf={cf_method}")
    return subprocess.run(cmd).returncode


def main() -> None:
    checkpoints = sorted(MODELS_DIR.glob("*.pth"))
    if not checkpoints:
        print(f"No checkpoints found under {MODELS_DIR}.")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for ckpt in checkpoints:
        dataset, _seed, model_name = parse_checkpoint(ckpt)
        if dataset is None:
            print(f"SKIP parse failure: {ckpt}")
            continue
        for cf_method in ("swap", "llm"):
            rc = run_eval(ckpt, dataset, model_name, cf_method)
            if rc != 0:
                errors.append(f"{ckpt.stem}:{cf_method}")

    print("=" * 60)
    if errors:
        print(f"Evaluation completed with {len(errors)} failed jobs:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("All LLM-CLP evaluation jobs completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
