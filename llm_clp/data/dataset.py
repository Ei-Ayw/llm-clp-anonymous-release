"""Data loading utilities for LLM-CLP training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REQUIRED_DATA_COLUMNS = {"text", "binary_label"}
REQUIRED_CF_COLUMNS = {"original_text", "cf_text"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Validate that a DataFrame contains the required columns."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_text_key(text: Any) -> str:
    """
    Normalize text keys for original/counterfactual alignment.

    This function intentionally applies only conservative normalization:
    string conversion, trimming, and whitespace folding. It avoids lowercasing
    or punctuation changes so genuinely different examples are not merged.
    """
    return " ".join(str(text).strip().split())


def seed_worker(worker_id: int) -> None:
    """Seed Python randomness in DataLoader workers for reproducible CF sampling."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


class CausalFairDataset(Dataset):
    """
    Pair original examples with optional LLM-generated counterfactuals.

    Each item always contains original tensors, the label, ``has_cf``, and
    counterfactual tensors. Examples without a counterfactual receive zero
    placeholder tensors and ``has_cf == 0``.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            cf_df: pd.DataFrame | None,
            tokenizer: Any,
            max_len: int = 128,
            cf_sample_ratio: float = 1.0,
    ) -> None:
        _validate_columns(df, REQUIRED_DATA_COLUMNS, "df")
        if cf_df is not None and len(cf_df) > 0:
            _validate_columns(cf_df, REQUIRED_CF_COLUMNS, "cf_df")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cf_sample_ratio = cf_sample_ratio
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["binary_label"].astype(int).tolist()
        self.cf_map = self._build_counterfactual_map(cf_df)
        self.cf_coverage = self._compute_counterfactual_coverage()

    @staticmethod
    def _build_counterfactual_map(cf_df: pd.DataFrame | None) -> dict[str, list[str]]:
        """Build a mapping from original text to available counterfactual texts."""
        cf_map: dict[str, list[str]] = defaultdict(list)
        if cf_df is None or len(cf_df) == 0:
            return {}

        for _, row in cf_df.iterrows():
            key = normalize_text_key(row["original_text"])
            cf_text = normalize_text_key(row["cf_text"])
            if key and cf_text:
                cf_map[key].append(cf_text)
        return dict(cf_map)

    def _compute_counterfactual_coverage(self) -> dict[str, float | int]:
        """Compute counterfactual coverage for the current split."""
        if not self.texts:
            return {"n_samples": 0, "n_with_cf": 0, "coverage": 0.0}
        n_with_cf = sum(1 for text in self.texts if normalize_text_key(text) in self.cf_map)
        return {
            "n_samples": int(len(self.texts)),
            "n_with_cf": int(n_with_cf),
            "coverage": float(n_with_cf / len(self.texts)),
        }

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        """Encode text into tokenizer tensor outputs."""
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])
        original = self._encode(text)

        item = {
            "orig_input_ids": original["input_ids"],
            "orig_attention_mask": original["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
        }

        cf_candidates = self.cf_map.get(normalize_text_key(text), [])
        use_cf = bool(cf_candidates) and random.random() <= self.cf_sample_ratio
        if use_cf:
            counterfactual = self._encode(random.choice(cf_candidates))
            item.update(
                {
                    "cf_input_ids": counterfactual["input_ids"],
                    "cf_attention_mask": counterfactual["attention_mask"],
                    "has_cf": torch.tensor(1, dtype=torch.long),
                }
            )
        else:
            item.update(
                {
                    "cf_input_ids": torch.zeros(self.max_len, dtype=torch.long),
                    "cf_attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                    "has_cf": torch.tensor(0, dtype=torch.long),
                }
            )

        return item


def causal_fair_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack batch fields while keeping counterfactual tensors aligned."""
    result = {
        key: torch.stack([item[key] for item in batch])
        for key in (
            "orig_input_ids",
            "orig_attention_mask",
            "cf_input_ids",
            "cf_attention_mask",
            "label",
            "has_cf",
        )
    }
    result["cf_indices"] = result["has_cf"].nonzero(as_tuple=False).flatten()
    return result


def get_causal_fair_loader(
        df: pd.DataFrame,
        cf_df: pd.DataFrame | None,
        tokenizer: Any,
        batch_size: int = 16,
        max_len: int = 128,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        cf_sample_ratio: float = 1.0,
        seed: int | None = None,
) -> DataLoader:
    """Build a DataLoader for LLM-CLP training or evaluation."""
    dataset = CausalFairDataset(
        df=df,
        cf_df=cf_df,
        tokenizer=tokenizer,
        max_len=max_len,
        cf_sample_ratio=cf_sample_ratio,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=causal_fair_collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


__all__ = [
    "CausalFairDataset",
    "REQUIRED_CF_COLUMNS",
    "REQUIRED_DATA_COLUMNS",
    "causal_fair_collate_fn",
    "get_causal_fair_loader",
    "normalize_text_key",
    "seed_worker",
]
