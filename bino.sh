#!/bin/bash
#SBATCH --job-name=binoculars
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200
#SBATCH --exclude=erc-hpc-vm053,erc-hpc-comp246,erc-hpc-comp035 

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"
export CUDA_LAUNCH_BLOCKING=1

source .venv_bino/bin/activate
which python
python --version

# DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh")
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")
# DATASETS=("raidModel_cohere_chat" "raidModel_gpt4" "raidModel_llama_chat" "raidModel_mistral_chat")


# DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh")
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")
# DATASETS=("raidModel_cohere_chat" "raidModel_gpt4" "raidModel_llama_chat" "raidModel_mistral_chat")


DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum" "multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh" "tsm_first" "tsm_extend" "tsm_sums" "tsm_tst" "raidModel_cohere_chat" "raidModel_gpt4" "raidModel_llama_chat" "raidModel_mistral_chat" "drlAttack_multi_llm_mixing" "drlAttack_paraphrase_attacks_llm" "drlAttack_perturbation_attacks_llm" "drlAttack_prompt_attacks_llm")
MODELS=("binoculars")
SMOKE_TEST=0
OOD=0

# Nested loop to run every model on every dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$DATASET, Model=$MODEL, OOD=$OOD, Smoke=$SMOKE_TEST"
        echo "------------------------------------------------"

        PYTHONPATH="${ROOT_DIR}" python src/baseline/baseline.py \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --smoke_test "$SMOKE_TEST" \
                --ood "$OOD"
    done
done
