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
DATASETS=("wikipedia_chatgpt" "arxiv_chatGPT" "arxiv_cohere" "wikipedia_cohere")
MODELS=("qwen_06b" "qwen_8b" "llama_8b")

# Fixed parameters
SMOKE_TEST=0
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