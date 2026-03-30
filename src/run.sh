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

# 4 domains = 4 generators
# "wikipedia_chatgpt" "wikipedia_cohere" "wikipedia_bloomz"
# "reddit_chatgpt" "reddit_cohere" "reddit_bloomz"
# "wikihow_chatgpt" "wikihow_cohere" "wikihow_bloomz"
# "arxiv_chatgpt" "arxiv_cohere" "arxiv_bloomz"

# DATASETS=("wikipedia_chatgpt" "wikipedia_cohere" "wikipedia_bloomz" "arxiv_chatgpt" "arxiv_cohere" "arxiv_bloomz" "reddit_chatgpt" "reddit_cohere" "reddit_bloomz")
# MODELS=("qwen_06b" "qwen_8b" "llama_8b") # "qwen_32b"
# MODELS=("qwen_32b") # "qwen_32b"

DATASETS=("multisocial_full") # "wikipedia_chatgpt" "arxiv_chatgpt" "reddit_chatgpt" 
MODELS=("llama_8b") # "qwen_8b" "llama_8b" "qwen_06b"
SAVE_VIZ=0
CLASSIFIER=0
SCORE=1

# Fixed parameters
MODES=("last_token") #  "logits" "last_token" "pooling" "horizontal"
DIFF_VECTORS=(0)
NORMALIZE=(1)

PREFIX=(0)
SMOKE_TEST=0

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for DIFF_VECTOR in "${DIFF_VECTORS[@]}"; do
                for PREFIX_FLAG in "${PREFIX[@]}"; do
                    for NORMALIZE_FLAG in "${NORMALIZE[@]}"; do
                        echo "------------------------------------------------"
                        echo "Running Experiment: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, DiffVec=$DIFF_VECTOR, Prefix=$PREFIX_FLAG, Normalize=$NORMALIZE_FLAG"
                        echo "------------------------------------------------"

                        uv run run.py \
                            --dataset "$DATASET" \
                            --model "$MODEL" \
                            --smoke_test "$SMOKE_TEST" \
                            --mode "$MODE" \
                            --diff_vectors "$DIFF_VECTOR" \
                            --prefix "$PREFIX_FLAG" \
                            --normalize "$NORMALIZE_FLAG" \
                            --save_viz "$SAVE_VIZ" \
                            --classifier "$CLASSIFIER" \
                            --score "$SCORE"
                    done
                done
            done
        done
    done
done
