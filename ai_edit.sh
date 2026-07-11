#!/bin/bash
#SBATCH --job-name=ai_edit
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --constraint=h200|a100

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

SEED=42
DATA_PATH="/scratch/prj/inf_nlg_ai_detection/coe/data/raw/editlens/train.csv"
BERT_MODEL="bert-base-uncased"
BERT_EPOCHS=2
BERT_BATCH_SIZE=16
LLP_MODEL="llama_8b"
LLP_MODE="default"
COMPONENTS=50
TOKEN_MODE="last_token"

PYTHONPATH="${ROOT_DIR}" uv run -m src.ai_edit \
    --seed "$SEED" \
    --data_path "$DATA_PATH" \
    --bert_model "$BERT_MODEL" \
    --bert_epochs "$BERT_EPOCHS" \
    --bert_batch_size "$BERT_BATCH_SIZE" \
    --llp_model "$LLP_MODEL" \
    --llp_mode "$LLP_MODE" \
    --components "$COMPONENTS" \
    --token_mode "$TOKEN_MODE"
