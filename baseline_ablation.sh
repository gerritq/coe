#!/bin/bash
#SBATCH --job-name=baseline_enc_size
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=a100|h200
#SBATCH --exclude=erc-hpc-comp035

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

# folder and file_name adjusted ONLY for encoder, biscope, and repre

# DATASETS=("drlDomain_arxiv" "tsm_first" "multisocial_en" "raidModel_gpt4")
DATASETS=("drlDomain_arxiv")

TRAINING_SIZES=(-1)
SEEDS=(42)
MAX_CHARS_LIST=(25 50 75 100 125 150 175 200 250 300 400) # -1 | 25 50 75 100 125 150 175 200 250 300 400 500 600 800

FOLDER="ablation"
SMOKE_TEST=0
OOD=0

MODELS=("encoder")

# Nested loop to run every model on every dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for TRAINING_SIZE in "${TRAINING_SIZES[@]}"; do
            for MAX_CHARS in "${MAX_CHARS_LIST[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    echo "------------------------------------------------"
                    echo "Running Baseline: Dataset=$DATASET, Model=$MODEL, OOD=$OOD, Smoke=$SMOKE_TEST, TRAINING_SIZE=$TRAINING_SIZE, SEED=$SEED, MAX_CHARS=$MAX_CHARS"
                    echo "------------------------------------------------"

                PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST" \
                    --ood "$OOD" \
                    --training_size "$TRAINING_SIZE" \
                    --folder "$FOLDER" \
                    --seed "$SEED" \
                    --max_chars "$MAX_CHARS"
                done
            done
        done
    done
done
