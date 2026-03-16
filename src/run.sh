#!/bin/bash
#SBATCH --job-name=coe
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15GB
# SBATCH --constraint=a100



# Define your lists
DATASETS=("wikipedia_chatgpt" "wikipedia_cohere" "wikipedia_bloomz" "arxiv_chatgpt" "arxiv_cohere" "arxiv_bloomz" "reddit_chatgpt" "reddit_cohere" "reddit_bloomz")
MODELS=("qwen_06b" "qwen_8b" "llama_8b") # "qwen_32b"

DATASETS=("wikipedia_chatgpt")
MODELS=("qwen_06b") # "qwen_32b"

# Fixed parameters
SMOKE_TEST=1
N=5000

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Experiment: Dataset=$DATASET, Model=$MODEL"
        echo "------------------------------------------------"

        uv run run.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --smoke_test "$SMOKE_TEST" \
            --n "$N"
            
    done
done