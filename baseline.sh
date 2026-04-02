#!/bin/bash
#SBATCH --job-name=baselines_bin
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
# SBATCH --constraint=a100

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

DATASETS=("tsm_mulit")  # "multisocial_full" "m4_multilingual"
MODELS=(
        "encoder" 
        "llr" 
        "fastdetectgpt" 
        "rank" 
        "entropy"
        "likelihood"
        # "binoculars" 
        )        

SMOKE_TEST=0

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$DATASET, Model=$MODEL"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}"  uv run src/baseline/baseline.py \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --smoke_test "$SMOKE_TEST"
    done
done

