#!/bin/bash
#SBATCH --job-name=probe_C
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --constraint=h200|b200|a100

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"

# DATASETS=("drlDomain_arxiv" "tsm_first" "multisocial_en" "m4_gpt4")
# Ablations
DATASETS=("drlDomain_arxiv" "tsm_first" "multisocial_en" "m4_gpt4")

TOKEN_MODE="last_token"
MODES=("mlp") # default | pca | meta | meta_attn
COMPONENTS_LIST=(50)
TRAINING_SIZES=(-1)

FOLDER="ablation"
SMOKE_TEST=0
OOD=0

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for COMPONENTS in "${COMPONENTS_LIST[@]}"; do
                for TRAINING_SIZE in "${TRAINING_SIZES[@]}"; do
                    for C in "${C_LIST[@]}"; do
                        echo "------------------------------------------------"
                        echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE, Mode=$MODE"
                        echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, COMPONENTS=$COMPONENTS, TRAINING_SIZE=$TRAINING_SIZE, C=$C"
                        echo "------------------------------------------------"
                
                        PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
                            --model "$MODEL" \
                            --dataset "$DATASET" \
                            --token_mode "$TOKEN_MODE" \
                            --smoke_test "$SMOKE_TEST" \
                            --ood "$OOD" \
                            --components "$COMPONENTS" \
                            --training_size "$TRAINING_SIZE" \
                            --mode "$MODE" \
                            --C "$C" \
                            --folder "$FOLDER"
                    done
                done
            done
        done
    done
done
