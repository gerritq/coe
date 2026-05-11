#!/bin/bash
#SBATCH --job-name=probe_la
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --constraint=h200|b200|a100
#SBATCH --exclude=erc-hpc-comp054

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"

# DATASETS=("drlDomain_arxiv" "drlDomain_writing_prompt" "drlDomain_yelp_review" "drlDomain_xsum")
# DATASETS=("drlAttack_multi_llm_mixing" "drlAttack_paraphrase_attacks_llm" "drlAttack_perturbation_attacks_llm" "drlAttack_prompt_attacks_llm")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh")
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")
# DATASETS=("m4_gpt4" "m4_dolly" "m4_cohere" "m4_bloomz")
# DATASETS=("CB_drlDomain" "CB_multisocial" "CB_tsm" "CB_tsm")
# DATASETS=("apt" "apt_m4_train")

DATASETS=("raid_cohere_chat" "drlDomain_arxiv" "multisocial_en")

TOKEN_MODE="last_token"
SMOKE_TEST=0
FOLDER="layer_analysis"
TRAINING_SIZE=-1

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE"
            echo "SmokeTest=$SMOKE_TEST"
            echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.layer_analysis \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --token_mode "$TOKEN_MODE" \
                --smoke_test "$SMOKE_TEST" \
                --folder "$FOLDER"
    done
done


