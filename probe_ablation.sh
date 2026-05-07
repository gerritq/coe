#!/bin/bash
#SBATCH --job-name=probe_ablation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --constraint=h200|b200|a100|l40s

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("llama_8b") # "llama_8b" "qwen_06b"

DATASETS=("tsm_first" "tsm_extend" "tsm_sums" "tsm_tst")

TOKEN_MODE="last_token"
MODES=("default" "meta" "meta_attn" "pca") # default | pca | meta | meta_attn
COMPONENTS_LIST=(10 20 50 100)


SMOKE_TEST=0
OOD=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for COMPONENTS in "${COMPONENTS_LIST[@]}"; do
                echo "------------------------------------------------"
                echo "Running Probe: Dataset=$DATASET, Model=$MODEL, TokenMode=$TOKEN_MODE, Mode=$MODE"
                echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, COMPONENTS=$COMPONENTS"
                echo "------------------------------------------------"

                PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --token_mode "$TOKEN_MODE" \
                    --smoke_test "$SMOKE_TEST" \
                    --ood "$OOD" \
                    --components "$COMPONENTS" \
                    --mode "$MODE"
            done
        done
    done
done

