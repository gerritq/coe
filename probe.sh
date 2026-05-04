#!/bin/bash
#SBATCH --job-name=probe_all_ood
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=05:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031,erc-hpc-comp053
#SBATCH --constraint=h200|b200|a100

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"
# DATASETS=("detectrl_arxiv" "detectrl_writing_prompt" "detectrl_yelp_review" "detectrl_xsum")
# DATASETS=("multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh" "multisocial_pt")
# DATASETS=("tsm_paras_en" "tsm_paras_pt" "tsm_paras_vi" "tsm_sums_en" "tsm_sums_pt" "tsm_sums_vi")

DATASETS=("detectrl_arxiv" "detectrl_writing_prompt" "detectrl_yelp_review" "detectrl_xsum" "multisocial_en" "multisocial_de" "multisocial_ru" "multisocial_zh" "multisocial_pt" "tsm_paras_en" "tsm_paras_pt" "tsm_paras_vi" "tsm_sums_en" "tsm_sums_pt" "tsm_sums_vi")


TOKEN_MODE="last_token"
MODES=("default" "pca" "meta" "meta_attn") # default | pca | meta | meta_attn
OOD=1
COMPONENTS=50
SMOKE_TEST=0

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "------------------------------------------------"
            echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE, Mode=$MODE"
            echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, COMPONENTS=$COMPONENTS"
            echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --token_mode "$TOKEN_MODE" \
                --smoke_test "$SMOKE_TEST" \
                --ood "$OOD" \
                --components "$COMPONENTS" \
                --mode "$MODE"
        done
    done
done
