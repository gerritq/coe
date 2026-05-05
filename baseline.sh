#!/bin/bash
#SBATCH --job-name=b_encoder
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200|b200|a100|l40s
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031,erc-hpc-comp038

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

# DATASETS=("detectrl_arxiv" "detectrl_writing_prompt" "detectrl_yelp_review" "detectrl_xsum")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh" "multisocial_pt")
# DATASETS=("tsm_paras_en" "tsm_paras_pt" "tsm_paras_vi" "tsm_sums_en" "tsm_sums_pt" "tsm_sums_vi")

DATASETS=("detectrl_arxiv" "detectrl_writing_prompt" "detectrl_yelp_review" "detectrl_xsum" "multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh" "multisocial_pt" "tsm_paras_en" "tsm_paras_pt" "tsm_paras_vi" "tsm_sums_en" "tsm_sums_pt" "tsm_sums_vi")
# DATASETS=("detectrl_arxiv")

SMOKE_TEST=0
OOD=0
MODELS=(
        # "revise"
        # "gescore"
        # "biscope"
        # "raidar"
        # "text_fluoroscopy"
        # "radar"
        # "openai_roberta"
        # "repreguard"
        "encoder" 
        # "llr" 
        # "fastdetectgpt" 
        # "rank" 
        # "entropy"
        # "likelihood"
        # "binoculars" 
        )        


# Nested loop to run every model on every dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
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
