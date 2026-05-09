#!/bin/bash
#SBATCH --job-name=baseline_ablation_encoder
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200|b200|a100
#SBATCH --exclude=erc-hpc-vm053 

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

"""
- folder adjusted for encoder, biscope, and repre
"""

DATASETS=("tsm_first" "drlAttack_multi_llm_mixing" "multisocial_en" "m4_gpt4")

TRAINING_SIZES=(6 10 50 100 250 500)

FOLDER="ablation"
SMOKE_TEST=0
OOD=0

MODELS=("repreguard" "biscope")        

# Nested loop to run every model on every dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for TRAINING_SIZE in "${TRAINING_SIZES[@]}"; do
            echo "------------------------------------------------"
            echo "Running Baseline: Dataset=$DATASET, Model=$MODEL, OOD=$OOD, Smoke=$SMOKE_TEST, TRAINING_SIZE=$TRAINING_SIZE"
            echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST" \
                    --ood "$OOD" \
                    --training_size "$TRAINING_SIZE"
        done
    done
done
