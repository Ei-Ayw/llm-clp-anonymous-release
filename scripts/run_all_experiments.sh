#!/bin/bash

# LLM-CLP release driver for the paper-standard main-model pipeline.
# This script runs only the proposed LLM-CLP model. Baseline implementations are
# intentionally not included in the anonymous release repository.

set -e

if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
fi
if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
cd "$PROJECT_DIR"

BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-100}
MAX_LEN=${MAX_LEN:-128}
DATA_DIR=${DATA_DIR:-data/causal_fair}

HATEXPLAIN_CF="$DATA_DIR/hatexplain_train_cf_llm.parquet"
TOXIGEN_CF="$DATA_DIR/toxigen_train_cf_llm.parquet"
DYNAHATE_CF="$DATA_DIR/dynahate_train_cf_llm.parquet"

run_llmclp() {
    local dataset="$1"
    local cf_path="$2"

    echo "[LLM-CLP] dataset=${dataset}"
    python scripts/run_main_3seeds.py \
        --dataset "$dataset" \
        --data_dir "$DATA_DIR" \
        --cf_path "$cf_path" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --epochs "$EPOCHS" \
        --max_len "$MAX_LEN"
}

echo "=========================================="
echo "Part 1: main LLM-CLP runs"
echo "=========================================="
run_llmclp hatexplain "$HATEXPLAIN_CF"
run_llmclp toxigen "$TOXIGEN_CF"
run_llmclp dynahate "$DYNAHATE_CF"

echo "=========================================="
echo "Part 2: evaluation and aggregation"
echo "=========================================="
python scripts/run_all_eval.py
python scripts/summarize_llmclp_results.py

echo "=========================================="
echo "LLM-CLP training and evaluation jobs completed"
echo "=========================================="
