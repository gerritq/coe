#!/bin/bash
#SBATCH --job-name=desc_layer_pca
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031,erc-hpc-comp053,erc-hpc-comp040,erc-hpc-comp039,erc-hpc-comp032,erc-hpc-comp033
#SBATCH --constraint=h200|b200|a100|l40s

set -euo pipefail

# nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODEL="qwen_06b"
SMOKE_TEST="1"

echo "Running layer_pca with MODEL=${MODEL}, SMOKE_TEST=${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/layer_pca.py \
#   --model "${MODEL}" \
#   --smoke_test "${SMOKE_TEST}"


# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/map.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"



# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/l1_probe.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/pca_ratio.py \
--model "${MODEL}" \
--smoke_test "${SMOKE_TEST}"

