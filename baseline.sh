#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
# SBATCH --constraint=h200|b200

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

DATASETS=("detectrl_arxiv")
SMOKE_TEST=1
MODELS=(
        # "encoder" 
        "llr" 
        # "fastdetectgpt" 
        # "rank" 
        # "entropy"
        # "likelihood"
        # "binoculars" 
        )        


# Nested loop to run every model on every dataset
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$DATASET, Model=$MODEL"
        echo "------------------------------------------------"

        if [[ "$MODEL" == "binoculars" ]]; then
            # UV_PROJECT_ENVIRONMENT=".venv_bo" 
            PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST"
        else
            PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --smoke_test "$SMOKE_TEST"
        fi
    done
done
