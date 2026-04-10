#!/bin/bash
#SBATCH --job-name=coe_zero_shot
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15GB
# SBATCH --constraint=a100

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

DATASETS=("tsm_multi") # "wikipedia_chatgpt" "arxiv_chatgpt" "reddit_chatgpt" 
MODELS=("llama_8b") # "qwen_8b" "llama_8b" "qwen_06b"
SAVE_VIZ=1
CLASSIFIER=0

# Fixed parameters
MODES=("default" "denoise")
TOKEN_MODES=("last_token") # last_token pooling horizontal
DIFF_VECTORS=(0)
NORMALIZE=(1)

PREFIX=(0)
SMOKE_TEST=0

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for TOKEN_MODE in "${TOKEN_MODES[@]}"; do
                for DIFF_VECTOR in "${DIFF_VECTORS[@]}"; do
                    for PREFIX_FLAG in "${PREFIX[@]}"; do
                        for NORMALIZE_FLAG in "${NORMALIZE[@]}"; do
                            echo "------------------------------------------------"
                            echo "Running Experiment: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, TokenMode=$TOKEN_MODE, DiffVec=$DIFF_VECTOR, Prefix=$PREFIX_FLAG, Normalize=$NORMALIZE_FLAG"
                            echo "------------------------------------------------"

                            PYTHONPATH="${ROOT_DIR}"  uv run src/coe/coe_run.py \
                                --dataset "$DATASET" \
                                --model "$MODEL" \
                                --smoke_test "$SMOKE_TEST" \
                                --mode "$MODE" \
                                --token_mode "$TOKEN_MODE" \
                                --diff_vectors "$DIFF_VECTOR" \
                                --prefix "$PREFIX_FLAG" \
                                --normalize "$NORMALIZE_FLAG" \
                                --save_viz "$SAVE_VIZ" \
                                --classifier "$CLASSIFIER"
                        done
                    done
                done
            done
        done
done
done
