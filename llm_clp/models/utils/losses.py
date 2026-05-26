"""Loss functions used by LLM-CLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CounterfactualLogitPairing(nn.Module):
    """Mean-squared error between original and counterfactual logits."""

    def forward(self, logits_orig: torch.Tensor, logits_cf: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(logits_orig, logits_cf)


__all__ = ["CounterfactualLogitPairing"]
