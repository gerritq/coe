#!/bin/bash

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"


# ===========================================================================
# EDITS
# ===========================================================================

# t_edits
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_edits

# f_edits
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_edits

# ===========================================================================
# OOD
# ===========================================================================

# # t_ood
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ood

# # f_ood
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_ood

# # t_ood_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ood_pca

# f_ood_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_ood_pca

# ===========================================================================
# ID
# ===========================================================================
# # t_id
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_id

#  t_id_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_id_pca

#  t_id_attacks_pca
PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_id_attacks_pca

# ===========================================================================
# LAYER
# ===========================================================================

# f_layer
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_layer

# f_layer_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_layer_pca

# ===========================================================================
# ABLATIONS
# ===========================================================================

# t_ablations
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ablations

# t_ablations_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.t_ablations_pca

# ===========================================================================
# PROBES
# ===========================================================================

# t_probes
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_probes

# ===========================================================================
# QUAL
# ===========================================================================

# f_qual
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_qual

# f_qual_if
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_qual_if

# ===========================================================================
# SAMPLES
# ===========================================================================

# f_samples
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_samples

# f_samples_pca
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_samples_pca

# ===========================================================================
# COMPLEX
# ===========================================================================

# f_complex
# PYTHONPATH="${ROOT_DIR}"  uv run -m src.items.f_complex