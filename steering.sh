#!/bin/bash
#SBATCH --job-name=ldp_sv_m0
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --constraint=h200|b200

nvidia-smi

# ROOT_DIR="${BASE_COE:-$(pwd)}"
# cd "${ROOT_DIR}"

DATASETS=("tsm_multi" "m4_multi" "drl_t1_perturbation" "drl_t1_paraphrase" "multisocial_full") # "multisocial_full" "m4_multilingual"
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
MODE="last_token"

SMOKE_TEST=0
OOD=0
OOD_SETS=("")
MANIFOLD=1

if [ "$MANIFOLD" -eq 1 ]; then
    PCA_COMPONENTS=(5 10 15 20 25 30 40 50)
else
    PCA_COMPONENTS=(0)
fi

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PCA_COMPONENTS_FLAG in "${PCA_COMPONENTS[@]}"; do
            
        echo "------------------------------------------------"
        echo "Running Steering: Dataset=$DATASET, Model=$MODEL, Mode=$MODE, PCA=$PCA_COMPONENTS_FLAG"
        echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, MANIFOLD=$MANIFOLD"
        echo "------------------------------------------------"

        # PYTHONPATH="${ROOT_DIR}"  
        uv run -m src.sv.steering \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --mode "$MODE" \
            --pca_components "$PCA_COMPONENTS_FLAG" \
            --smoke_test "$SMOKE_TEST" \
            --ood "$OOD" \
            --manifold "$MANIFOLD"
        done
    done
done
