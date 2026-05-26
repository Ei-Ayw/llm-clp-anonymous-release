"""Training loop for Counterfactual Logit Pairing with LLM counterfactuals."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn.functional as F
from ..utils.paths import OUTPUT_DIR
from ..data.dataset import get_causal_fair_loader
from ..models import DebertaV3CausalFair
from ..models.utils.losses import CounterfactualLogitPairing
from ..utils.io import ensure_dir, write_json
from ..utils.seed import set_seed
from ..utils.parse import parse_args_with_config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal
import torch

DATASETS = ("hatexplain", "toxigen", "dynahate")

# Command-line interface.
def build_parser() -> argparse.ArgumentParser:
    """Create the method-specific training parser."""
    parser = argparse.ArgumentParser(
        description="Train LLM-CLP with counterfactual logit pairing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--dataset", default="hatexplain", choices=DATASETS, help="Dataset name.")
    parser.add_argument("--data_dir", type=str, default="data/causal_fair",
                        help="Directory containing split parquet files.")
    parser.add_argument("--cf_path", type=str, default=None, help="Counterfactual training parquet path.")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base",
                        help="HF model name or local path.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum token length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for cosine scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience.")
    parser.add_argument("--lambda_clp", type=float, default=1.0, help="Weight for CLP loss.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--cf_sample_ratio", type=float, default=1.0, help="Probability of using an available CF example for a sample.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--ablation_type", type=str, default=None, help="Ablation type (for ablation experiments).")
    parser.add_argument("--ablation_value", type=str, default=None, help="Ablation value (for ablation experiments).")
    parser.add_argument("--checkpoint_save_path", type=str, default=None, help="Optional explicit checkpoint save path.")
    return parser


# Early stopping on validation Macro-F1.
@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    patience: int = 3
    min_delta: float = 0.0
    mode: Literal["max", "min"] = "max"

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")
        self.counter = 0
        self.best: float | None = None

    def __call__(self, value: float) -> bool:
        """Return ``True`` when training should stop."""
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


# Helpers.
def amp_context(device: torch.device):
    """Return an autocast context only when CUDA is available."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def make_grad_scaler(device: torch.device) -> torch.cuda.amp.GradScaler:
    """Create a gradient scaler that is enabled only on CUDA devices."""
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def count_parameters(model: torch.nn.Module) -> int:
    """Count all trainable and non-trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


# Data and scheduler helpers.
def load_splits(data_dir: str | Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test split parquet files."""
    base_dir = Path(data_dir)
    return (
        pd.read_parquet(base_dir / f"{dataset}_train.parquet"),
        pd.read_parquet(base_dir / f"{dataset}_val.parquet"),
        pd.read_parquet(base_dir / f"{dataset}_test.parquet"),
    )


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        epochs: int,
        grad_accum: int,
        warmup_ratio: float,
) -> Any:
    """Create a cosine schedule with warmup."""
    # Actual optimizer updates after gradient accumulation.
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum)))
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


# Train one epoch.
def train_one_epoch(
        model: DebertaV3CausalFair,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        lambda_clp: float,
        grad_accum: int,
        class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Train one epoch and return averaged loss components."""
    model.train()
    # Initialize losses.
    if class_weights is not None:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        ce_loss = nn.CrossEntropyLoss()
    clp_loss = CounterfactualLogitPairing()

    totals = {"loss": 0.0, "ce": 0.0, "clp": 0.0}
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Train")
    for step, batch in enumerate(progress, start=1):
        # Original and counterfactual tensors are kept aligned in the batch.
        orig_ids = batch["orig_input_ids"].to(device)
        orig_mask = batch["orig_attention_mask"].to(device)
        cf_ids = batch["cf_input_ids"].to(device)
        cf_mask = batch["cf_attention_mask"].to(device)
        labels = batch["label"].to(device)
        cf_indices = batch["cf_indices"].to(device)

        with amp_context(device):
            # Forward pass on original texts.
            orig_output = model(orig_ids, orig_mask, return_features=False)
            loss_ce = ce_loss(orig_output["logits"], labels)
            loss_clp = torch.zeros((), device=device)

            # Compute CLP only for examples with retained counterfactuals.
            if cf_indices.numel() > 0:
                # Reuse original logits from the main batch.
                paired_cf_ids = cf_ids.index_select(0, cf_indices)
                paired_cf_mask = cf_mask.index_select(0, cf_indices)
                cf_output = model(paired_cf_ids, paired_cf_mask, return_features=False)

                # Slice original logits to match the paired counterfactual rows.
                orig_logits = orig_output["logits"].index_select(0, cf_indices)
                cf_logits = cf_output["logits"]
                loss_clp = clp_loss(orig_logits, cf_logits)

            loss = loss_ce + lambda_clp * loss_clp

        # Scale loss so accumulation is equivalent to a larger batch.
        actual_accum = grad_accum
        if step > len(loader) - (len(loader) % grad_accum):
            actual_accum = len(loader) % grad_accum or grad_accum

        scaler.scale(loss / actual_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.item())
        totals["ce"] += float(loss_ce.item())
        totals["clp"] += float(loss_clp.item())
        average_loss = totals["loss"] / step
        progress.set_postfix(loss=f"{average_loss:.4f}")

    num_batches = max(1, len(loader))
    return {name: value / num_batches for name, value in totals.items()}


# Validation and testing.
@torch.no_grad()
def evaluate(model: DebertaV3CausalFair, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate task performance on a DataLoader."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    for batch in tqdm(loader, desc="Eval"):
        ids = batch["orig_input_ids"].to(device)
        mask = batch["orig_attention_mask"].to(device)

        with amp_context(device):
            output = model(ids, mask, return_features=False)
            probs = F.softmax(output["logits"], dim=-1)[:, 1]

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())

    probs_arr = np.asarray(all_probs)
    labels_arr = np.asarray(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "macro_f1": float(f1_score(labels_arr, preds_arr, average="macro")),
        "binary_f1": float(f1_score(labels_arr, preds_arr, average="binary")),
        "auc_roc": float(roc_auc_score(labels_arr, probs_arr)) if len(set(labels_arr)) > 1 else 0.5,
    }


# Main flow.
def main() -> None:
    """Run LLM-CLP training."""
    args = parse_args_with_config(build_parser())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    result_dir = ensure_dir(output_dir / "eval")

    # Load ordinary splits and counterfactual training data.
    train_df, val_df, test_df = load_splits(args.data_dir, args.dataset)
    cf_df = pd.read_parquet(args.cf_path) if args.cf_path and Path(args.cf_path).exists() else None

    # Class weights: N / (2 * N_c).
    class_counts = train_df["binary_label"].value_counts().sort_index()
    if len(class_counts) == 2:
        total = len(train_df)
        class_weights = torch.tensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])], dtype=torch.float32)
    else:
        class_weights = None

    print(f"Device: {device}")
    print(f"Data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Counterfactual pairs: {0 if cf_df is None else len(cf_df)}")

    # Build DataLoaders.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = get_causal_fair_loader(
        train_df,
        cf_df,
        tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=True,
        num_workers=args.num_workers,
        cf_sample_ratio=args.cf_sample_ratio,
        seed=args.seed,
    )
    val_loader = get_causal_fair_loader(
        val_df,
        None,
        tokenizer,
        batch_size=args.batch_size * 2,
        max_len=args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = get_causal_fair_loader(
        test_df,
        None,
        tokenizer,
        batch_size=args.batch_size * 2,
        max_len=args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if hasattr(train_loader.dataset, "cf_coverage"):
        cov = train_loader.dataset.cf_coverage
        print(
            "Counterfactual coverage: "
            f"{cov['n_with_cf']}/{cov['n_samples']} = {cov['coverage']:.2%}"
        )

    # Initialize model, optimizer, and scheduler.
    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    print(f"Model: {args.model_name} ({count_parameters(model):,} parameters)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, train_loader, args.epochs, args.grad_accum, args.warmup_ratio)
    scaler = make_grad_scaler(device)
    early_stopping = EarlyStopping(patience=args.patience)

    # Honor an explicit checkpoint path when supplied by a driver script.
    if args.checkpoint_save_path:
        checkpoint_path = Path(args.checkpoint_save_path)
        ensure_dir(checkpoint_path.parent)
        # Store the training result JSON next to the checkpoint.
        results_path = checkpoint_path.parent / f"{checkpoint_path.stem}_results.json"
        experiment_name = checkpoint_path.stem
    else:
        # Default path for standalone runs.
        timestamp = datetime.now().strftime("%m%d_%H%M")
        backbone_slug = args.model_name.split('/')[-1]
        if args.ablation_type and args.ablation_value:
            experiment_name = f"{args.dataset}_{backbone_slug}_llmclp_{args.ablation_type}_{args.ablation_value}_s{args.seed}"
            model_dir = ensure_dir(output_dir / "ablation_models" / f"{args.ablation_type}_{args.ablation_value}")
            result_dir = ensure_dir(output_dir / "ablation_eval" / f"{args.ablation_type}_{args.ablation_value}")
        else:
            experiment_name = f"{args.dataset}_{backbone_slug}_llmclp_s{args.seed}"
            model_dir = ensure_dir(output_dir / "models")
            result_dir = ensure_dir(output_dir / "eval")
            
        checkpoint_path = model_dir / f"{experiment_name}.pth"
        results_path = result_dir / f"{experiment_name}_results.json"
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    best_f1 = float("-inf")

    # Training loop.
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            lambda_clp=args.lambda_clp,
            grad_accum=args.grad_accum,
            class_weights=class_weights,
        )
        val_metrics = evaluate(model, val_loader, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            "Train: "
            f"loss={train_metrics['loss']:.4f} ce={train_metrics['ce']:.4f} "
            f"clp={train_metrics['clp']:.4f}"
        )
        print(f"Val: macro_f1={val_metrics['macro_f1']:.4f} auc={val_metrics['auc_roc']:.4f}")

        # Save only when validation Macro-F1 improves.
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_dict = {
                "method": "llm_clp",
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "seed": args.seed,
                "dataset": args.dataset,
                "epoch": epoch,
                "best_val_f1": best_f1,
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

        if early_stopping(val_metrics["macro_f1"]):
            print(f"Early stopping triggered. Best validation F1: {best_f1:.4f}")
            break

    # Evaluate the best checkpoint on the test split.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: macro_f1={test_metrics['macro_f1']:.4f} auc={test_metrics['auc_roc']:.4f}")

    write_json(
        {
            "experiment": experiment_name,
            "args": vars(args),
            "best_val_f1": best_f1,
            "test_metrics": test_metrics,
            "cf_coverage": getattr(train_loader.dataset, "cf_coverage", None),
            "history": history,
            "checkpoint": str(checkpoint_path),
        },
        results_path,
    )
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
