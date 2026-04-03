#!/bin/bash
#SBATCH --job-name=sv_m1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=h200|b200

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

# Full data run
# DATASETS=("tsm_multi" "m4_multi" "drl_t1_perturbation" "drl_t1_paraphrase" "multisocial_full")
DATASETS=("m4_wikipedia_chatgpt")
MODELS=("qwen_06b")  # "llama_8b" "qwen_06b"
SV_MODE="default"   # default | denoise | ldp
TOKEN_MODE="last_token"

SMOKE_TEST=1
OOD=""
NORMALIZE_SCORES=1

if [ "$SV_MODE" = "denoise" ] || [ "$SV_MODE" = "ldp" ]; then
    PCA_COMPONENTS=(5 10 15 20 25 30 40 50)
else
    PCA_COMPONENTS=(0)
fi

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PCA_COMPONENTS_FLAG in "${PCA_COMPONENTS[@]}"; do
            
        echo "------------------------------------------------"
        echo "Running Steering: Dataset=$DATASET, Model=$MODEL, SVMode=$SV_MODE, TokenMode=$TOKEN_MODE, PCA=$PCA_COMPONENTS_FLAG"
        echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, NormalizeScores=$NORMALIZE_SCORES"
        echo "------------------------------------------------"

        
        PYTHONPATH="${ROOT_DIR}"  uv run -m src.sv.run \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --mode "$SV_MODE" \
            --token_mode "$TOKEN_MODE" \
            --pca_components "$PCA_COMPONENTS_FLAG" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD" \
            --normalize_scores "$NORMALIZE_SCORES"
        done
    done
done
