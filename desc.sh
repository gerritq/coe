#!/bin/bash
#SBATCH --job-name=desc_qual_3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,nmes_gpu
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

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/lp.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/ranks.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# QUAL METRICS

# METRICS=("von_neumann_entropy" "anisotropy" "intrinsic_dimensionality")
# # METRICS=("effective_rank")
# SEEDS=(42 43 44 45 46)
# for METRIC in "${METRICS[@]}"; do
#   for SEED in "${SEEDS[@]}"; do
#     echo "Running qual_metrics with metric=${METRIC}, seed=${SEED}"
#     PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/qual_metrics.py \
#       --model "${MODEL}" \
#       --smoke_test "${SMOKE_TEST}" \
#       --metric "${METRIC}" \
#       --seed "${SEED}"
#   done
# done

# PROBE VECTORS

# PROBE_VECTOR_MODES=("pca_space") # "default" "pca" "pca_space"
# PCA_COMPONENTS=100
# for MODE in "${PROBE_VECTOR_MODES[@]}"; do
#     echo "Running probe_vectors with mode=${MODE}"
#     PYTHONPATH="${ROOT_DIR}" uv run python -m src.descriptives.probe_vectors \
#       --model "${MODEL}" \
#       --mode "${MODE}" \
#       --components "${PCA_COMPONENTS}"
# done

# ACTIVATIONS
# PYTHONPATH="${ROOT_DIR}" uv run python src/descriptives/activate.py \
# --model "${MODEL}" \
# --smoke_test "${SMOKE_TEST}"

# MLP
DOMAINS=("wikipedia") # "arxiv" "reddit" "peerread"
COMPLEXITIES=(1 2 3 4 5 6 7 8)
MODE="log" # mlp | log
OOD=0
for DOMAIN in "${DOMAINS[@]}"; do
    for COMPLEXITY in "${COMPLEXITIES[@]}"; do
        echo "Running mlp with domain=${DOMAIN} | complexity=${COMPLEXITY} | mode=${MODE} | ood=${OOD}"
        PYTHONPATH="${ROOT_DIR}" uv run python -m src.descriptives.mlp \
        --model "${MODEL}" \
        --mode "${MODE}" \
        --domain "${DOMAIN}" \
        --ood "${OOD}" \
        --smoke_test "${SMOKE_TEST}" \
        --complexity "${COMPLEXITY}"
    done
done
