#!/bin/bash
#SBATCH --job-name=baseline_ood_biscope
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200
#SBATCH --exclude=erc-hpc-vm053,erc-hpc-comp246

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

# DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")
# DATASETS=("drlAttack_multi_llm_mixing" "drlAttack_paraphrase_attacks_llm" "drlAttack_perturbation_attacks_llm" "drlAttack_prompt_attacks_llm")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh")
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")
# DATASETS=("drlDomain_xsum" "m4_gpt4" "m4_dolly" "m4_cohere" "m4_bloomz")

DATASETS=("raidModel_cohere_chat" "raidModel_gpt4" "raidModel_llama_chat" "raidModel_mistral_chat")
# DATASETS=("raidDomain_wiki" "raidDomain_reddit" "raidDomain_news" "raidDomain_abstracts")

SMOKE_TEST=0
OOD=1

# "raidar"
MODELS=(
        # "raidar"
        # "editlens"
        # "revise"
        # "gescore"
        "biscope"
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

# OOD
# MODELS=("biscope" "text_fluoroscopy" "repreguard" "encoder")        


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
