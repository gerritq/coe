#!/bin/bash
#SBATCH --job-name=probe_logistic
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031
#SBATCH --constraint=h200|b200|a100

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"
mkdir -p logs

MODELS=("llama_8b") # "llama_8b" "qwen_06b"
DATASETS=("detectrl_arxiv")
PROBE_MODES=("logistic") # "logistic" "logistic_m" "feature"
TOKEN_MODE="last_token"
SMOKE_TEST=1
OOD=""                  # keep empty for ID; for OOD pass space-separated datasets
NORMALIZE_SCORES=0
ABLATION_SET="all"


for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PROBE_MODE in "${PROBE_MODES[@]}"; do
            if [ "$PROBE_MODE" = "logistic_m" ]; then
                PCA_COMPONENTS=(5 10 25 50)
            else
                PCA_COMPONENTS=(0)
            fi

            for PCA_COMPONENTS_FLAG in "${PCA_COMPONENTS[@]}"; do
                echo "------------------------------------------------"
                echo "Running Probe: Dataset=$DATASET, Model=$MODEL, Mode=$PROBE_MODE, TokenMode=$TOKEN_MODE, PCA=$PCA_COMPONENTS_FLAG"
                echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, NormalizeScores=$NORMALIZE_SCORES, AblationSet=$ABLATION_SET"
                echo "------------------------------------------------"

                PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --mode "$PROBE_MODE" \
                    --token_mode "$TOKEN_MODE" \
                    --pca_components "$PCA_COMPONENTS_FLAG" \
                    --smoke_test "$SMOKE_TEST" \
                    --ood "$OOD" \
                    --normalize_scores "$NORMALIZE_SCORES" \
                    --ablation_set "$ABLATION_SET"
            done
        done
    done
done
