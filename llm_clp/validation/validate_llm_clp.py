"""Evaluate an LLM-CLP checkpoint on task, group, and counterfactual metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from ..models.DebertaV3CausalFair import DebertaV3CausalFair
from ..utils.io import ensure_dir, write_json
from ..utils.parse import parse_args_with_config
from ..utils.paths import EVAL_DIR
from ..validation.metrics import evaluate_causal_fairness

DATASETS = ("hatexplain", "toxigen", "dynahate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM-CLP checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=False, default=None, help="Checkpoint path.")
    parser.add_argument("--dataset", default="hatexplain", choices=DATASETS, help="Dataset name.")
    parser.add_argument("--cf_method", default="swap", choices=("swap", "llm"), help="Counterfactual evaluation source.")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", help="HF model name or local path.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive-class decision threshold.")
    parser.add_argument("--data_dir", default="data/causal_fair", help="Directory containing normalized parquet data.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser


def load_counterfactuals(data_dir: Path, dataset: str, cf_method: str) -> pd.DataFrame | None:
    suffix = "cf_swap" if cf_method == "swap" else "cf_llm"
    cf_path = data_dir / f"{dataset}_test_{suffix}.parquet"
    if not cf_path.exists():
        return None
    cf_df = pd.read_parquet(cf_path)
    return cf_df if len(cf_df) > 0 else None


def main() -> None:
    args = parse_args_with_config(build_parser())
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless provided by --config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_stem = checkpoint_path.stem

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_model_name = checkpoint.get("args", {}).get("model_name", "")
    if saved_model_name and saved_model_name != args.model_name:
        raise ValueError(
            f"Checkpoint model_name ('{saved_model_name}') != args.model_name ('{args.model_name}'). "
            f"Use --model_name {saved_model_name} to load this checkpoint."
        )

    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_dir = Path(args.data_dir)
    test_path = data_dir / f"{args.dataset}_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")
    test_df = pd.read_parquet(test_path)
    cf_df = load_counterfactuals(data_dir, args.dataset, args.cf_method)

    results = evaluate_causal_fairness(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        cf_df=cf_df,
        device=device,
        max_len=args.max_len,
        threshold=args.threshold,
    )
    results["meta"] = {
        "method": "llm_clp",
        "dataset": args.dataset,
        "cf_method": args.cf_method,
        "checkpoint": str(checkpoint_path),
        "model_name": args.model_name,
        "n_test": int(len(test_df)),
        "n_cf_pairs": int(0 if cf_df is None else len(cf_df)),
    }

    default_name = f"{checkpoint_stem}_{args.cf_method}_metrics.json"
    output_path = Path(args.output) if args.output else EVAL_DIR / default_name
    ensure_dir(output_path.parent)
    write_json(results, output_path)
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
