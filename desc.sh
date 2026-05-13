#!/bin/bash
#SBATCH --job-name=desc_qual
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp054

nvidia-smi

set -euo pipefail

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

MODEL="llama_8b" # llama_8b qwen_06b
SMOKE_TEST="0"

echo "Running desc with MODEL=${MODEL}, SMOKE_TEST=${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/layer_pca.py \
#   --model "${MODEL}" \
#   --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/map.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/l1_probe.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/pca_ratio.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/lp.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/ranks.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# QUAL METRICS
METRICS=("von_neumann_entropy" "effective_rank" "anisotropy" "intrinsic_dimensionality")
SEEDS=(42 43 44 45 46)
for METRIC in "${METRICS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "Running qual_metrics with metric=${METRIC}, seed=${SEED}"
    PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/qual_metrics.py \
      --model "${MODEL}" \
      --smoke_test "${SMOKE_TEST}" \
      --metric "${METRIC}" \
      --seed "${SEED}"
  done
done
