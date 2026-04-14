#!/bin/bash
#SBATCH --job-name=sv_ood_ldp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
# SBATCH --constraint=h200|b200

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

# Full data run
# DATASETS=("tsm_multi" "m4_multi" "drl_t1_perturbation" "drl_t1_paraphrase" "multisocial_full")
DATASETS=("tsm_multi")
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
SV_MODES=("pca_align")   # default | denoise | denoise_layer | ldp | ldp_by_layer | pca_align
TOKEN_MODE="last_token"

SMOKE_TEST=0
# OOD="multisocial_de multisocial_nl multisocial_pt multisocial_ar"
OOD=""
NORMALIZE_SCORES=1

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SV_MODE in "${SV_MODES[@]}"; do
            if [ "$SV_MODE" != "default" ]; then
                # PCA_COMPONENTS=(5 10 15 20 25 30 40 50)
                PCA_COMPONENTS=(25 50 100)
                # PCA_COMPONENTS=(20)
            else
                PCA_COMPONENTS=(0)
            fi

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
done
