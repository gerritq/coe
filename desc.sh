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

DATASETS=("multisocial_en" "m4_wikipedia_chatgpt")
MODELS=("llama_8b") # qwen_06b
ANALYSIS="traj"  # "ld", "traj", "sv", "all"
SPLIT="val"
MODE="last_token"
DIM=3
PREFIX=0
SMOKE_TEST=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Descriptives: Dataset=$DATASET, Model=$MODEL, Split=$SPLIT, Mode=$MODE, Analysis=$ANALYSIS, Dim=$DIM"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}"  uv run src/descriptives/desc_run.py \
            --model "$MODEL" \
            --data "$DATASET" \
            --split "$SPLIT" \
            --mode "$MODE" \
            --dim "$DIM" \
            --prefix "$PREFIX" \
            --analysis "$ANALYSIS" \
            --smoke_test "$SMOKE_TEST"
    done
done
