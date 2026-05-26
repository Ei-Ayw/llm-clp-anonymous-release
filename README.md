# LLM-CLP Anonymous Release

This repository contains the anonymous reproducibility package for the LLM-CLP main model pipeline.

LLM-CLP trains a Transformer toxicity classifier with class-weighted cross-entropy on original examples and Counterfactual Logit Pairing (CLP) on available LLM-generated identity counterfactual pairs.

## Scope

This anonymous release includes:

- LLM and lexical-swap counterfactual generation utilities.
- LLM-CLP training code.
- LLM-CF and swap-CF evaluation code.
- Three-backbone training entry points for `microsoft/deberta-v3-base`, `roberta-base`, and `bert-base-uncased`.
- Paper result artifacts used to audit the reported aggregate tables.

The baseline implementations used for the full paper comparison are not included in this reduced anonymous release; the release focuses on the proposed main pipeline.

## Repository Layout

```text
llm_clp/
  data/                    # dataset loading and counterfactual generation
  models/                  # LLM-CLP classifier and CLP loss
  training/                # LLM-CLP training entry point
  validation/              # task, group, and counterfactual evaluation
scripts/
  run_main_3seeds.py       # paper-standard LLM-CLP run for one dataset/backbone
  run_llmclp_3backbones_3seeds.py
  run_all_eval.py
  summarize_llmclp_results.py
paper_artifacts/           # CSV summaries shipped for result auditing
data/causal_fair/          # normalized data should be placed here
checkpoints/               # optional model checkpoints should be placed here
```

## Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The reported experiments used PyTorch, HuggingFace Transformers, CUDA 11.8, and NVIDIA RTX 3090 GPUs. Each independent training run uses a single GPU; independent runs can be scheduled in parallel.

## Data Preparation

The normalized parquet files are expected under `data/causal_fair/`:

```text
hatexplain_train.parquet
hatexplain_val.parquet
hatexplain_test.parquet
toxigen_train.parquet
toxigen_val.parquet
toxigen_test.parquet
dynahate_train.parquet
dynahate_val.parquet
dynahate_test.parquet
```

Counterfactual files follow the same naming pattern:

```text
{dataset}_train_cf_llm.parquet
{dataset}_test_cf_llm.parquet
{dataset}_test_cf_swap.parquet
```

The expected columns for ordinary splits are `text` and `binary_label`. Counterfactual files should contain `original_text` and `cf_text`.

## Counterfactual Generation

Generate LLM counterfactuals with a local Ollama server:

```bash
python -m llm_clp.data.generator_llm --dataset hatexplain --split train --model qwen2.5:3b
```

Generate deterministic lexical-swap counterfactuals:

```bash
python -m llm_clp.data.generator_swap --dataset hatexplain --split test
```

## Training

Run the paper-standard LLM-CLP configuration for one dataset:

```bash
python scripts/run_main_3seeds.py --dataset hatexplain --backbone microsoft/deberta-v3-base
```

Run the three-backbone diagnostic:

```bash
python scripts/run_llmclp_3backbones_3seeds.py
```

Important defaults match the paper: seeds `42, 123, 2024`, maximum length `128`, batch size `32`, gradient accumulation `1`, learning rate `2e-5`, weight decay `0.01`, cosine warmup ratio `0.1`, maximum `100` epochs, early stopping patience `3`, and `lambda_clp=1.0`.

## Evaluation

Evaluate all checkpoints under both swap-CF and LLM-CF sources:

```bash
python scripts/run_all_eval.py
python scripts/summarize_llmclp_results.py
```

Row-level and aggregate CSV summaries are written to `outputs/eval/`.

## Checkpoints

No trained model weights are included in this anonymous source package by default. To evaluate released checkpoints, place `.pth` files under `outputs/models/` or `checkpoints/` and pass the path to:

```bash
python -m llm_clp.validation.validate_llm_clp --checkpoint path/to/checkpoint.pth --dataset hatexplain --cf_method swap
```

## Anonymity

This repository is prepared for double-blind review. Do not add author names, institutional paths, personal URLs, or non-anonymous repository remotes before submission.

