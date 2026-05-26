#!/usr/bin/env python3
"""Run LLM-CLP across three classifier backbones and three seeds."""

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
        description="Run LLM-CLP for DeBERTa-V3, RoBERTa, and BERT backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES, help="Backbones to run.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="Datasets to run.")
    parser.add_argument("--data_dir", default="data/causal_fair", help="Directory containing normalized parquet data.")
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


def run_training(args: argparse.Namespace, backbone: str, dataset: str, seed: int) -> dict[str, int | str]:
    cf_path = Path(args.data_dir) / f"{dataset}_train_cf_llm.parquet"
    cmd = [
        sys.executable,
        "-m",
        "llm_clp.training.train_DebertaV3CausalFair",
        "--dataset",
        dataset,
        "--data_dir",
        args.data_dir,
        "--model_name",
        backbone,
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
    if cf_path.exists():
        cmd.extend(["--cf_path", str(cf_path)])

    print(f"\n{'=' * 60}")
    print(f"LLM-CLP | backbone={backbone} | dataset={dataset} | seed={seed}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, capture_output=False)
    return {"backbone": backbone, "dataset": dataset, "seed": seed, "returncode": result.returncode}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = "llmclp_3backbones_3seeds"
    results = []
    total_runs = len(args.backbones) * len(args.datasets) * len(SEEDS)
    run_idx = 0

    print(f"{'=' * 60}")
    print("LLM-CLP backbone diagnostic runs")
    print(f"Backbones: {args.backbones}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {total_runs}")
    print(f"{'=' * 60}")

    for backbone in args.backbones:
        for dataset in args.datasets:
            for seed in SEEDS:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}]")
                result = run_training(args, backbone, dataset, seed)
                results.append(result)
                if result["returncode"] != 0:
                    print(f"[ERROR] {backbone}/{dataset}/s{seed} failed; continuing.")

    output_file = Path(args.output_dir) / f"{experiment_name}_summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment_name,
        "backbones": args.backbones,
        "datasets": args.datasets,
        "seeds": SEEDS,
        "args": vars(args),
        "timestamp": timestamp,
        "results": results,
        "all_success": all(r["returncode"] == 0 for r in results),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    failed = [r for r in results if r["returncode"] != 0]
    print(f"\n{'=' * 60}")
    if failed:
        print(f"Completed with {len(failed)} failed runs:")
        for r in failed:
            print(f"  - {r['backbone']}/{r['dataset']}/s{r['seed']}")
    else:
        print("All runs completed successfully.")
    print(f"Summary: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
