# LLM-CLP Main Model Card

## Model

LLM-CLP is a binary toxicity classifier trained with:

- A Transformer encoder backbone: `microsoft/deberta-v3-base`, `roberta-base`, or `bert-base-uncased`.
- A dropout layer and linear two-logit classifier.
- Class-weighted cross-entropy on original examples.
- Counterfactual Logit Pairing on available original/counterfactual pairs.

The main paper comparison uses `microsoft/deberta-v3-base`.

## Objective

For a batch `B` and paired subset `B_cf`, the training objective is:

```text
L = L_CE + lambda_clp * L_CLP
```

where `L_CLP` is mean-squared error between original and counterfactual logits. The paper-standard setting is `lambda_clp = 1.0`.

## Intended Use

This model is intended for research on counterfactual stability in toxicity and hate-speech classification. It is not intended for standalone deployment in moderation systems.

## Training Data

The release expects normalized HateXplain, ToxiGen, and DynaHate splits under `data/causal_fair/`. Users must follow the licenses and terms of the original datasets.

## Counterfactuals

LLM-generated counterfactuals are used as noisy training-time invariance signals. Deterministic lexical swaps are used as the primary shared counterfactual evaluation source in the main paper.

## Limitations

LLM-generated counterfactuals can contain semantic drift, toxicity drift, failed identity substitution, or culturally asymmetric substitutions. Any use in high-stakes moderation should include human review, appeal processes, and subgroup auditing.

## Checkpoints

This anonymous source package does not include trained checkpoint weights by default. If weights are added for review, place `.pth` files under `outputs/models/` or `checkpoints/` and ensure checkpoint metadata is anonymized.
