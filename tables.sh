#!/bin/bash

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

# # t_ood
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ood

# # t_id
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_id

# # f_ood
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_ood

# f_samples
PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_samples

# t_ablations
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ablations

# f_complex
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_complex