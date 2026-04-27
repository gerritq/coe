#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200|b200
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

DATASETS=(
    "detectrl_arxiv"
    "multisocial_en"
    "tsm_paras_en"
)
SMOKE_TEST=0
OOD=0
MODELS=(
        "text_fluoroscopy"
        # "raidar"
        "radar"
        "openai_roberta"
        "repreguard"
        "encoder" 
        "llr" 
        "fastdetectgpt" 
        "rank" 
        "entropy"
        "likelihood"
        "binoculars" 
        )        


# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$DATASET, Model=$MODEL, OOD=$OOD, Smoke=$SMOKE_TEST"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --smoke_test "$SMOKE_TEST" \
                --ood "$OOD"
    done
done
