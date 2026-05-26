"""Main classifier used by the LLM-CLP method."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .utils import extract_cls


class CausalFairClassifier(nn.Module):
    """Transformer classifier used by LLM-CLP.

    This is the main classifier for LLM-CLP, supporting any Transformer backbone.
    The projection head is retained for backward-compatible checkpoint loading.
    """

    def __init__(
            self,
            model_path: str,
            backbone_type: str = "bert",
            num_classes: int = 2,
            dropout: float = 0.1,
            projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            return_features: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return logits and optional projection features."""
        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.backbone_type == "bert":
            model_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**model_kwargs)
        cls_hidden = extract_cls(outputs)
        result = {"logits": self.classifier(self.dropout(cls_hidden))}
        if return_features:
            result["features"] = self.projector(cls_hidden)
            result["cls_hidden"] = cls_hidden
        return result

def _infer_backbone_type(model_path: str) -> str:
    """Infer backbone type from model name."""
    model_lower = model_path.lower()
    if "roberta" in model_lower:
        return "roberta"
    if "deberta" in model_lower:
        return "deberta"
    return "bert"  # default for bert-base-uncased, etc.


class DebertaV3CausalFair(CausalFairClassifier):
    """DeBERTa-v3 classifier alias for the LLM-CLP implementation.

    This is a backward-compatibility alias for CausalFairClassifier, which
    supports any Transformer backbone, not just DeBERTa-v3.
    Supports auto-detection of backbone type from model name.
    """

    def __init__(
            self,
            model_path: str = "microsoft/deberta-v3-base",
            num_classes: int = 2,
            dropout: float = 0.1,
            projection_dim: int = 128,
            backbone_type: str | None = None,
    ) -> None:
        inferred = backbone_type or _infer_backbone_type(model_path)
        super().__init__(
            model_path=model_path,
            backbone_type=inferred,
            num_classes=num_classes,
            dropout=dropout,
            projection_dim=projection_dim,
        )


__all__ = [
    "CausalFairClassifier",
    "DebertaV3CausalFair",
]
