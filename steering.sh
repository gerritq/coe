#!/bin/bash
#SBATCH --job-name=coe_steering
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=a100

nvidia-smi

# ROOT_DIR="${BASE_COE:-$(pwd)}"
# cd "${ROOT_DIR}"

DATASETS=("m4_multilingual" "multisocial_full" "tsm_paras_en_first_deepseek" "tsm_sums_en_deepseek")  # "multisocial_full" "m4_multilingual"
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
MODE="last_token"

CENTERING=(0)
SMOKE_TEST=0
OOD=0
OOD_SETS=("")
MANIFOLD=0

if [ "$MANIFOLD" -eq 1 ]; then
    PCA_COMPONENTS=(1 3 5 7 10)
else
    PCA_COMPONENTS=(10)
fi

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for CENTERING_FLAG in "${CENTERING[@]}"; do
            for PCA_COMPONENTS_FLAG in "${PCA_COMPONENTS[@]}"; do
            
        echo "------------------------------------------------"
        echo "Running Steering: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, Centering=$CENTERING_FLAG, PCA=$PCA_COMPONENTS_FLAG"
        echo "ValSplit=$VAL_SPLIT, SmokeTest=$SMOKE_TEST, OOD=$OOD, MANIFOLD=$MANIFOLD"
        echo "------------------------------------------------"

        # PYTHONPATH="${ROOT_DIR}"  
        uv run -m src.sv.steering \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --mode "$MODE" \
            --centering "$CENTERING_FLAG" \
            --pca_components "$PCA_COMPONENTS_FLAG" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD" \
            --manifold "$MANIFOLD"
            done
        done
    done
done
