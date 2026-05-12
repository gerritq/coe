#!/bin/bash
#SBATCH --job-name=pa_training_size_seeds
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --constraint=h200|b200|a100

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"

# DS for training size
DATASETS=("drlDomain_arxiv" "tsm_first" "multisocial_en" "raidModel_gpt4")

# DS for other ablations
# DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")


MODES=("default" "meta_no_pca") # default | pca | meta | meta_attn | poly
COMPONENTS_LIST=(50)
TRAINING_SIZES=(10 50 100 250 500) # -1 | 10 50 100 250 500
C_LIST=(1)
MLP_DEPTH_LIST=(1)
SEEDS=(42 43 44 45 46)

TOKEN_MODE="last_token"
FOLDER="ablation"
SMOKE_TEST=0
OOD=0

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for COMPONENTS in "${COMPONENTS_LIST[@]}"; do
                for TRAINING_SIZE in "${TRAINING_SIZES[@]}"; do
                    for C in "${C_LIST[@]}"; do
                        for MLP_DEPTH in "${MLP_DEPTH_LIST[@]}"; do
                            for SEED in "${SEEDS[@]}"; do
                                echo "------------------------------------------------"
                                echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE, Mode=$MODE"
                                echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, COMPONENTS=$COMPONENTS, TRAINING_SIZE=$TRAINING_SIZE, C=$C"
                                echo "MLP_DEPTH=$MLP_DEPTH, SEED=$SEED"
                                echo "------------------------------------------------"
                    
                            PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.probe_ablations \
                                --model "$MODEL" \
                                --dataset "$DATASET" \
                                --token_mode "$TOKEN_MODE" \
                                --smoke_test "$SMOKE_TEST" \
                                --ood "$OOD" \
                                --components "$COMPONENTS" \
                                --training_size "$TRAINING_SIZE" \
                                --mode "$MODE" \
                                --C "$C" \
                                --folder "$FOLDER" \
                                --mlp_depth "$MLP_DEPTH" \
                                --seed "$SEED"
                            done
                        done    
                    done
                done
            done
        done
    done
done
