"""Task, counterfactual-stability, and group-error metrics."""
from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Callable signature for model-specific logit prediction.
PredictLogitsFn = Callable


def compute_cfr(preds_orig: np.ndarray, preds_cf: np.ndarray) -> float:
    """Compute the Counterfactual Flip Rate."""
    if len(preds_orig) == 0:
        return 0.0
    return float(np.mean(preds_orig != preds_cf))


def compute_ctfg(probs_orig: np.ndarray, probs_cf: np.ndarray) -> float:
    """Compute the mean absolute positive-class probability difference."""
    if len(probs_orig) == 0:
        return 0.0
    return float(np.mean(np.abs(probs_orig - probs_cf)))


def _safe_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float((y_true == 0).sum())
    if denominator == 0:
        return float('nan')
    return float(((y_pred == 1) & (y_true == 0)).sum()) / denominator


def _safe_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float((y_true == 1).sum())
    if denominator == 0:
        return float('nan')
    return float(((y_pred == 0) & (y_true == 1)).sum()) / denominator


def compute_fped_fned(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        min_group_size: int = 1,
) -> tuple[float, float, dict[str, dict[str, float | int]]]:
    """Compute FPED/FNED as sums of group-vs-overall error-rate gaps."""
    # Overall FPR and FNR.
    overall_fpr = _safe_fpr(y_true, y_pred)
    overall_fnr = _safe_fnr(y_true, y_pred)

    fped = 0.0
    fned = 0.0
    details = {}

    # Per-group gaps.
    unique_groups = np.unique(groups)
    for group in unique_groups:
        group_mask = (groups == group)
        if group_mask.sum() < min_group_size:
            continue

        group_fpr = _safe_fpr(y_true[group_mask], y_pred[group_mask])
        group_fnr = _safe_fnr(y_true[group_mask], y_pred[group_mask])

        fpr_gap = abs(overall_fpr - group_fpr) if not np.isnan(group_fpr) else 0.0
        fnr_gap = abs(overall_fnr - group_fnr) if not np.isnan(group_fnr) else 0.0

        if not np.isnan(group_fpr):
            fped += fpr_gap
        if not np.isnan(group_fnr):
            fned += fnr_gap

        details[str(group)] = {
            "count": int(group_mask.sum()),
            "fpr": float(group_fpr) if not np.isnan(group_fpr) else None,
            "fnr": float(group_fnr) if not np.isnan(group_fnr) else None,
            "fpr_gap": float(fpr_gap) if not np.isnan(group_fpr) else None,
            "fnr_gap": float(fnr_gap) if not np.isnan(group_fnr) else None,
        }

    return float(fped), float(fned), details


def default_predict_logits(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Default logit prediction function for the LLM-CLP model signature."""
    return model(input_ids, attention_mask, return_features=False)["logits"]


@torch.no_grad()
def predict_probs(
        model: torch.nn.Module,
        texts: list[str],
        tokenizer: Any,
        device: torch.device,
        max_len: int = 128,
        batch_size: int = 32,
        predict_logits_fn: PredictLogitsFn = default_predict_logits,
) -> np.ndarray:
    """Predict positive-class probabilities for a list of texts."""
    model.eval()
    all_probs: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = predict_logits_fn(model, input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())

    return np.asarray(all_probs)


def task_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute standard binary-classification task metrics."""
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "binary_f1": float(f1_score(labels, preds, average="binary")),
        "auc_roc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def normalize_text_key(text: Any) -> str:
    """Normalize text keys for conservative dataset/CF alignment checks."""
    return " ".join(str(text).strip().split())


def _first_group(value: Any) -> str:
    """Extract a stable primary group from common scalar/list encodings."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "none"
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return stripped
            return _first_group(parsed)
        return stripped

    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]) if len(value) > 0 else "none"

    if pd.isna(value):
        return "none"
    return str(value)


def infer_group_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Find or create the primary group column used by FPED/FNED."""
    if "target_group" in df.columns:
        return df, "target_group"
    if "source_group" in df.columns:
        return df, "source_group"
    if "coarse_groups" in df.columns:
        output = df.copy()
        output["_primary_group"] = output["coarse_groups"].apply(_first_group)
        return output, "_primary_group"
    return df, None


def evaluate_causal_fairness(
        model: torch.nn.Module,
        tokenizer: Any,
        test_df: pd.DataFrame,
        cf_df: pd.DataFrame | None,
        device: torch.device,
        max_len: int = 128,
        threshold: float = 0.5,
        predict_logits_fn: PredictLogitsFn = default_predict_logits,
) -> dict[str, Any]:
    """Run task, counterfactual-stability, and group-error evaluation."""
    if "text" not in test_df.columns or "binary_label" not in test_df.columns:
        raise ValueError("test_df must contain 'text' and 'binary_label' columns.")

    # Task metrics on the ordinary test split.
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["binary_label"].to_numpy()
    probs = predict_probs(model, texts, tokenizer, device, max_len, predict_logits_fn=predict_logits_fn)
    preds = (probs >= threshold).astype(int)

    results: dict[str, Any] = {"task": task_metrics(labels, probs, threshold)}

    # Counterfactual stability metrics when a CF set is provided.
    if cf_df is not None and len(cf_df) > 0:
        if not {"original_text", "cf_text"}.issubset(cf_df.columns):
            raise ValueError("cf_df must contain 'original_text' and 'cf_text' columns.")
        original_texts = cf_df["original_text"].astype(str).tolist()
        cf_texts = cf_df["cf_text"].astype(str).tolist()
        probs_orig = predict_probs(model, original_texts, tokenizer, device, max_len,
                                   predict_logits_fn=predict_logits_fn)
        probs_cf = predict_probs(model, cf_texts, tokenizer, device, max_len, predict_logits_fn=predict_logits_fn)
        preds_orig = (probs_orig >= threshold).astype(int)
        preds_cf = (probs_cf >= threshold).astype(int)
        results["causal"] = {
            "cfr": compute_cfr(preds_orig, preds_cf),
            "ctfg": compute_ctfg(probs_orig, probs_cf),
            "n_pairs": int(len(cf_df)),
            "mean_prob_orig": float(np.mean(probs_orig)),
            "mean_prob_cf": float(np.mean(probs_cf)),
        }
    else:
        results["causal"] = None

    # Group-error metrics using one primary group label per example.
    grouped_df, group_col = infer_group_column(test_df)
    if group_col is not None and grouped_df[group_col].nunique() > 1:
        groups = grouped_df[group_col].to_numpy()
        fped, fned, group_details = compute_fped_fned(labels, preds, groups, min_group_size=1)
        per_group_f1 = {}
        for group in sorted(set(groups)):
            mask = groups == group
            if int(mask.sum()) > 0 and len(set(labels[mask])) > 1:
                per_group_f1[str(group)] = float(f1_score(labels[mask], preds[mask], average="macro"))
        results["group"] = {
            "fped": fped,
            "fned": fned,
            "group_details": group_details,
            "per_group_f1": per_group_f1,
            "per_group_f1_std": float(np.std(list(per_group_f1.values()))) if per_group_f1 else None,
        }
    else:
        results["group"] = None

    return results


__all__ = [
    "PredictLogitsFn",
    "compute_cfr",
    "compute_ctfg",
    "compute_fped_fned",
    "normalize_text_key",
    "default_predict_logits",
    "evaluate_causal_fairness",
    "infer_group_column",
    "predict_probs",
    "task_metrics",
]
