#!/bin/bash
#SBATCH --job-name=desc_layer_pca
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODEL="llama_8b" # llama_8b
SMOKE_TEST="0"

echo "Running desc with MODEL=${MODEL}, SMOKE_TEST=${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/layer_pca.py \
#   --model "${MODEL}" \
#   --smoke_test "${SMOKE_TEST}"

PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/map.py \
--model "${MODEL}" \
--smoke_test "${SMOKE_TEST}"



# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/l1_probe.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/pca_ratio.py \
--model "${MODEL}" \
--smoke_test "${SMOKE_TEST}"

