#!/bin/bash
#SBATCH --job-name=baseline_ablation_size_all
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200|b200
#SBATCH --exclude=erc-hpc-vm053 

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

"""
- folder and file_name adjusted ONLY for encoder, biscope, and repre
"""

DATASETS=("drlDomain_arxiv" "tsm_first" "multisocial_en" "raidModel_gpt4")

TRAINING_SIZES=(10 50 100 250 500)
SEEDS=(42 43 44 45 46)

FOLDER="ablation"
SMOKE_TEST=0
OOD=0

MODELS=("repreguard" "biscope" "encoder") 

# Nested loop to run every model on every dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for TRAINING_SIZE in "${TRAINING_SIZES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                echo "------------------------------------------------"
                echo "Running Baseline: Dataset=$DATASET, Model=$MODEL, OOD=$OOD, Smoke=$SMOKE_TEST, TRAINING_SIZE=$TRAINING_SIZE, SEED=$SEED"
                echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST" \
                    --ood "$OOD" \
                    --training_size "$TRAINING_SIZE" \
                    --folder "$FOLDER" \
                    --seed "$SEED"
            done
        done
    done
done
