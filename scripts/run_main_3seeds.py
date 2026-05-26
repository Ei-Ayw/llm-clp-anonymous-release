#!/usr/bin/env python3
"""Run the paper-standard LLM-CLP training protocol with three seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SEEDS = [42, 123, 2024]
DATASETS = ["hatexplain", "toxigen", "dynahate"]
BACKBONES = ["microsoft/deberta-v3-base", "roberta-base", "bert-base-uncased"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM-CLP with seeds 42, 123, and 2024.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS, help="Dataset name.")
    parser.add_argument("--data_dir", default="data/causal_fair", help="Directory containing normalized parquet data.")
    parser.add_argument("--cf_path", type=str, default=None, help="LLM counterfactual training parquet path.")
    parser.add_argument("--backbone", default="microsoft/deberta-v3-base", choices=BACKBONES, help="Classifier backbone.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience.")
    parser.add_argument("--lambda_clp", type=float, default=1.0, help="CLP loss weight.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--cf_sample_ratio", type=float, default=1.0, help="Probability of using an available CF pair.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    return parser


def infer_cf_path(data_dir: str, dataset: str, explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    candidate = Path(data_dir) / f"{dataset}_train_cf_llm.parquet"
    return str(candidate) if candidate.exists() else None


def run_training(args: argparse.Namespace, seed: int) -> dict[str, int]:
    cmd = [
        sys.executable,
        "-m",
        "llm_clp.training.train_DebertaV3CausalFair",
        "--dataset",
        args.dataset,
        "--data_dir",
        args.data_dir,
        "--model_name",
        args.backbone,
        "--max_len",
        str(args.max_len),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(seed),
        "--patience",
        str(args.patience),
        "--lambda_clp",
        str(args.lambda_clp),
        "--num_workers",
        str(args.num_workers),
        "--cf_sample_ratio",
        str(args.cf_sample_ratio),
        "--output_dir",
        args.output_dir,
    ]

    cf_path = infer_cf_path(args.data_dir, args.dataset, args.cf_path)
    if cf_path:
        cmd.extend(["--cf_path", cf_path])

    print(f"\n{'=' * 60}")
    print(f"LLM-CLP | backbone={args.backbone} | dataset={args.dataset} | seed={seed}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=False)
    return {"seed": seed, "returncode": result.returncode}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"llm_clp_{args.backbone.split('/')[-1]}_{args.dataset}_3seeds"
    results = []

    print(f"{'=' * 60}")
    print("LLM-CLP three-seed training")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Seeds: {SEEDS}")
    print(f"Experiment: {experiment_name}")
    print(f"{'=' * 60}")

    for seed in SEEDS:
        result = run_training(args, seed)
        results.append(result)
        if result["returncode"] != 0:
            print(f"[ERROR] seed={seed} failed.")
            sys.exit(1)

    output_file = Path(args.output_dir) / f"{experiment_name}_summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment_name,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seeds": SEEDS,
        "args": vars(args),
        "timestamp": timestamp,
        "results": results,
        "all_success": all(r["returncode"] == 0 for r in results),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("LLM-CLP three-seed training completed.")
    print(f"Summary: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
