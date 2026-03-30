#!/bin/bash
#SBATCH --job-name=baselines_bin
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=30GB
# SBATCH --constraint=a100

nvidia-smi

# 4 domains = 4 generators
# "wikipedia_chatgpt" "wikipedia_cohere" "wikipedia_bloomz"
# "reddit_chatgpt" "reddit_cohere" "reddit_bloomz"
# "wikihow_chatgpt" "wikihow_cohere" "wikihow_bloomz"
# "arxiv_chatgpt" "arxiv_cohere" "arxiv_bloomz"

# DATASETS=("wikipedia_chatgpt" "reddit_chatgpt" "wikihow_chatgpt" "arxiv_chatgpt")
DATASETS=("multisocial_full")
MODELS=(
        # "encoder" 
        # "llr" 
        # "fastdetectgpt" 
        # "rank" 
        # "entropy"
        # "likelihood"
        "binoculars" 
        )        

SMOKE_TEST=0

# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$DATASET, Model=$MODEL"
        echo "------------------------------------------------"

            uv run baseline.py \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --smoke_test "$SMOKE_TEST"
    done
done

