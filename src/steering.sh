#!/bin/bash
#SBATCH --job-name=coe_steering
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=a100

nvidia-smi

# Example:
# DATASETS=("wikipedia_chatgpt" "arxiv_chatgpt")
# MODELS=("llama_8b" "qwen_8b")

DATASETS=("wikipedia_chatgpt")
MODELS=("llama_8b")  # "llama_8b"

MODE="last_token"
VAL_SPLIT="val"
TEST_SPLIT="test"
N_VAL=-1
N_TEST=-1
PREFIX=0
SMOKE_TEST=0
OOD=1

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Steering: Dataset=$DATASET, Model=$MODEL, Mode=$MODE"
        echo "ValSplit=$VAL_SPLIT, TestSplit=$TEST_SPLIT, NVal=$N_VAL, NTest=$N_TEST, Prefix=$PREFIX, SmokeTest=$SMOKE_TEST, OOD=$OOD"
        echo "------------------------------------------------"

        uv run steering.py \
            --model "$MODEL" \
            --data "$DATASET" \
            --mode "$MODE" \
            --val_split "$VAL_SPLIT" \
            --test_split "$TEST_SPLIT" \
            --n_val "$N_VAL" \
            --n_test "$N_TEST" \
            --prefix "$PREFIX" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD"
    done
done
