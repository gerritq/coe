#!/bin/bash
#SBATCH --job-name=probe_logistic
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031
#SBATCH --constraint=h200|b200|a100

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"
DATASETS=("detectrl_arxiv")
TOKEN_MODE="last_token"
SMOKE_TEST=0
OOD=1
PCA=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE"
        echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, PCA=$PCA"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --token_mode "$TOKEN_MODE" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD" \
            --pca "$PCA"
    done
done
