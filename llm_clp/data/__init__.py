"""Data loading and counterfactual generation utilities."""
from .dataset import (
    CausalFairDataset,
    causal_fair_collate_fn,
    get_causal_fair_loader,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_CF_COLUMNS,
    normalize_text_key,
    seed_worker,
)
from .generator_llm import (
    OllamaGenerator,
    generate_counterfactuals_for_dataset,
)
from .generator_swap import (
    IDENTITY_SWAP_GROUPS,
    FLAT_SWAP_MAP,
    generate_swap_counterfactuals,
    batch_generate_swap,
)

__all__ = [
    "CausalFairDataset",
    "causal_fair_collate_fn",
    "get_causal_fair_loader",
    "REQUIRED_DATA_COLUMNS",
    "REQUIRED_CF_COLUMNS",
    "normalize_text_key",
    "seed_worker",
    "OllamaGenerator",
    "generate_counterfactuals_for_dataset",
    "IDENTITY_SWAP_GROUPS",
    "FLAT_SWAP_MAP",
    "generate_swap_counterfactuals",
    "batch_generate_swap",
]
