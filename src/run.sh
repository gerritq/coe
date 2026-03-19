#!/bin/bash
#SBATCH --job-name=coe_horizontal
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=03:30:00
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

DATASETS=("wikipedia_chatgpt" "arxiv_chatgpt" "wikihow_chatgpt" "reddit_chatgpt")
MODELS=("qwen_8b" "llama_8b") # "qwen_32b" "qwen_06b"

# Fixed parameters
MODES=("horizontal") #  "logits" "last_token" "pooling" "horizontal"
DIFF_VECTORS=(0 1)

TEST=1
SMOKE_TEST=0
N=1000

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for DIFF_VECTOR in "${DIFF_VECTORS[@]}"; do
                echo "------------------------------------------------"
                echo "Running Experiment: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, DiffVec=$DIFF_VECTOR"
                echo "------------------------------------------------"

                uv run run.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST" \
                    --n "$N" \
                    --mode "$MODE" \
                    --diff_vectors "$DIFF_VECTOR" \
                    --test "$TEST"
            done
        done
    done
done
