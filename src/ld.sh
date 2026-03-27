#!/bin/bash
#SBATCH --job-name=ld_pca
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=a100

nvidia-smi

DATASETS=("wikihow_chatgpt" "reddit_chatgpt")
MODELS=("llama_8b") # qwen_06b

SPLIT="test"
MODE="last_token"
N=500
PREFIX=0
SMOKE_TEST=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running LD PCA: Dataset=$DATASET, Model=$MODEL, Split=$SPLIT, Mode=$MODE, N=$N"
        echo "------------------------------------------------"

        uv run ld.py \
            --model "$MODEL" \
            --data "$DATASET" \
            --split "$SPLIT" \
            --mode "$MODE" \
            --n "$N" \
            --prefix "$PREFIX" \
            --smoke_test "$SMOKE_TEST"
    done
done
