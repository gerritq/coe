#!/bin/bash
#SBATCH --job-name=ld_pca
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=a100

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

DATASETS=("multisocial_full" "m4_wikipedia_chatgpt")
MODELS=("llama_8b") # qwen_06b
ANALYSIS="ld"  # "ld", "sv", "all"
SPLIT="val"
MODE="pooling"
N=500
PREFIX=0
SMOKE_TEST=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running LD PCA: Dataset=$DATASET, Model=$MODEL, Split=$SPLIT, Mode=$MODE, N=$N"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}"  uv run src/descriptives/descriptives.py \
            --model "$MODEL" \
            --data "$DATASET" \
            --split "$SPLIT" \
            --mode "$MODE" \
            --n "$N" \
            --prefix "$PREFIX" \
            --analysis "$ANALYSIS" \
            --smoke_test "$SMOKE_TEST"
    done
done
