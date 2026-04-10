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

ROOT_DIR="${BASE_COE}"

DATASETS=("multisocial_full")  # "multisocial_full" ""m4_multilingual""
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
MODE="last_token"

CENTERING=(0)
SMOKE_TEST=1
OOD=0
OOD_SETS=("")
MANIFOLD=0

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for CENTERING_FLAG in "${CENTERING[@]}"; do
            
        echo "------------------------------------------------"
        echo "Running Steering: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, Centering=$CENTERING_FLAG"
        echo "ValSplit=$VAL_SPLIT, TestSplit=$TEST_SPLIT, NVal=$N_VAL, NTest=$N_TEST, SmokeTest=$SMOKE_TEST, OOD=$OOD, MANIFOLD=$MANIFOLD"
        echo "------------------------------------------------"

        uv run "${ROOT_DIR}/src/sv/steering.py" \
            --model "$MODEL" \
            --data "$DATASET" \
            --mode "$MODE" \
            --centering "$CENTERING_FLAG" \
            --val_split "$VAL_SPLIT" \
            --test_split "$TEST_SPLIT" \
            --n_val "$N_VAL" \
            --n_test "$N_TEST" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD" \
            --manifold "$MANIFOLD"
        done
    done
done
