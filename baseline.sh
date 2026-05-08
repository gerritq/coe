#!/bin/bash
#SBATCH --job-name=b_raidar_domain
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=06:00:00
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

# DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")
# DATASETS=("drlAttack_multi_llm_mixing" "drlAttack_paraphrase_attacks_llm" "drlAttack_perturbation_attacks_llm" "drlAttack_prompt_attacks_llm")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh")
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")
# DATASETS=("m4_gpt4" "m4_dolly" "m4_cohere" "m4_bloomz")


DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")

SMOKE_TEST=0
OOD=0
# "raidar"
MODELS=(
        "raidar"
        # "revise"
        # "gescore"
        # "biscope"
        # "text_fluoroscopy"
        # "radar"
        # "openai_roberta"
        # "repreguard"
        # "encoder" 
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
