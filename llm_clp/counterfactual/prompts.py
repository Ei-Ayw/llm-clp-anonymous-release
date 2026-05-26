"""Prompt template for counterfactual generation (FREE_FORM)."""

from __future__ import annotations

PROMPT_FREE_FORM = """Replace the identity group in the following text with a contextually plausible alternative identity, preserving syntax and sentiment.

STRICT RULES:
1. Identify the demographic/identity group being referenced
2. Replace it with a different but contextually appropriate group
3. Preserve the EXACT syntactic structure, sentiment, and toxicity level
4. Make culturally appropriate substitutions
5. Do NOT add, remove, or rephrase other content
6. Return ONLY the rewritten text, nothing else

Original text: "{text}"

Rewritten text:"""


__all__ = [
    "PROMPT_FREE_FORM",
]
