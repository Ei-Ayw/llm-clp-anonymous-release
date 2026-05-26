#!/usr/bin/env python3
"""Aggregate LLM-CLP evaluation JSON files into CSV summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

EVAL_DIR = Path("outputs/eval")


def infer_checkpoint_info(stem: str) -> dict[str, str | int]:
    clean = stem
    for suffix in ("_swap_metrics", "_llm_metrics"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    cf_method = "swap" if stem.endswith("_swap_metrics") else "llm"

    dataset = "unknown"
    for candidate in ("hatexplain", "toxigen", "dynahate"):
        if candidate in clean:
            dataset = candidate
            break

    seed_match = re.search(r"_s(\d+)(?:_|$)", clean)
    seed = int(seed_match.group(1)) if seed_match else -1

    model = "unknown"
    for candidate in ("deberta-v3-base", "roberta-base", "bert-base-uncased"):
        if candidate in clean:
            model = candidate
            break

    return {"dataset": dataset, "seed": seed, "model": model, "cf_method": cf_method}


def read_metrics(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    info = infer_checkpoint_info(path.stem)
    task = data.get("task", {})
    causal = data.get("causal") or {}
    group = data.get("group") or {}
    meta = data.get("meta") or {}
    return {
        "file": path.name,
        "dataset": meta.get("dataset", info["dataset"]),
        "model": meta.get("model_name", info["model"]),
        "seed": info["seed"],
        "cf_method": meta.get("cf_method", info["cf_method"]),
        "accuracy": task.get("accuracy"),
        "macro_f1": task.get("macro_f1"),
        "binary_f1": task.get("binary_f1"),
        "auc_roc": task.get("auc_roc"),
        "precision": task.get("precision"),
        "recall": task.get("recall"),
        "cfr": causal.get("cfr"),
        "ctfg": causal.get("ctfg"),
        "fped": group.get("fped"),
        "fned": group.get("fned"),
        "per_group_f1_std": group.get("per_group_f1_std"),
        "n_cf_pairs": causal.get("n_pairs"),
    }


def main() -> None:
    files = sorted(EVAL_DIR.glob("*_metrics.json"))
    if not files:
        print(f"No metric files found under {EVAL_DIR}.")
        return

    rows = [read_metrics(path) for path in files]
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "model", "seed", "cf_method"]).reset_index(drop=True)
    raw_csv = EVAL_DIR / "llmclp_eval_rows.csv"
    df.to_csv(raw_csv, index=False)

    metric_cols = [
        "accuracy",
        "macro_f1",
        "binary_f1",
        "auc_roc",
        "precision",
        "recall",
        "cfr",
        "ctfg",
        "fped",
        "fned",
        "per_group_f1_std",
    ]
    summary = (
        df.groupby(["dataset", "model", "cf_method"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_csv = EVAL_DIR / "llmclp_eval_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved row-level metrics to {raw_csv}")
    print(f"Saved aggregate metrics to {summary_csv}")


if __name__ == "__main__":
    main()
