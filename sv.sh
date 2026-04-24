#!/bin/bash
#SBATCH --job-name=sv_ood_ctv
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp050,erc-hpc-comp031
# SBATCH --constraint=h200|b200|a100

nvidia-smi

ROOT_DIR="${BASE_COE:-$(pwd)}"
cd "${ROOT_DIR}"

# Multitude - languages
# DATASETS=("multitude_en")
# OOD="multitude_de \
# multitude_nl \
# multitude_pt \
# multitude_ro \
# multitude_uk"

# Multisocial - languages
# DATASETS=("multisocial_en")
# OOD="multisocial_de \
# multisocial_nl \
# multisocial_pt \
# multisocial_ar"

# TSM - Generators
# DATASETS=("tsm_paras_en_first_gpt4o")
# OOD="tsm_paras_en_first_gemini \
# tsm_paras_en_first_deepseek \
# tsm_paras_pt_first_gpt4o \
# tsm_paras_pt_first_gemini \
# tsm_paras_pt_first_deepseek \
# tsm_paras_vi_first_gpt4o \
# tsm_paras_vi_first_gemini \
# tsm_paras_vi_first_deepseek"

# TSM - Tasks
# DATASETS=("tsm_paras_en_first_gpt4o")
# OOD="tsm_sums_en_gemini \
# tsm_sums_en_deepseek \
# tsm_sums_pt_gpt4o \
# tsm_sums_pt_gemini \
# tsm_sums_pt_deepseek \
# tsm_sums_vi_gpt4o \
# tsm_sums_vi_gemini \
# tsm_sums_vi_deepseek"

# Full ID data
# DATASETS=("tsm_multi" "m4_multi" "drl_t1_perturbation" "drl_t1_paraphrase" "multisocial_full")
# DATASETS=("tsm_multi")
DATASETS=("editlens")
OOD=""
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
SV_MODES=("default" "m_logistic" "lda" "m_lda")   # default | denoise | denoise_layer | denoise_layer_split | denoise_val | clean_topic | clean_topic_val | ldp | ldp_by_layer | lda | pca_align | pca_sv | pca_layer
TOKEN_MODE="last_token"
# this is for sv_topic | sv_topic_val
ABLATION_SET="human"  # human | machine | all

SMOKE_TEST=0

NORMALIZE_SCORES=1

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SV_MODE in "${SV_MODES[@]}"; do
            if [ "$SV_MODE" != "default" ] && [ "$SV_MODE" != "clean_topic" ] && [ "$SV_MODE" != "clean_topic_val" ] && [ "$SV_MODE" != "lda" ]; then
                # PCA_COMPONENTS=(5 10 15 20 25 30 40 50)
                PCA_COMPONENTS=(25 50 100)
                # PCA_COMPONENTS=(20)
            else
                PCA_COMPONENTS=(0)
            fi

            for PCA_COMPONENTS_FLAG in "${PCA_COMPONENTS[@]}"; do
                
            echo "------------------------------------------------"
            echo "Running Steering: Dataset=$DATASET, Model=$MODEL, SVMode=$SV_MODE, TokenMode=$TOKEN_MODE, PCA=$PCA_COMPONENTS_FLAG"
            echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, NormalizeScores=$NORMALIZE_SCORES, AblationSet=$ABLATION_SET"
            echo "------------------------------------------------"

            
            PYTHONPATH="${ROOT_DIR}"  uv run -m src.sv.run \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --mode "$SV_MODE" \
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
