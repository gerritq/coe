#!/bin/bash
#SBATCH --job-name=probe_logistic
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

# Full ID data examples
# DATASETS=("tsm_multi" "m4_multi" "drl_t1_perturbation" "drl_t1_paraphrase" "multisocial_full")
DATASETS=("editlens")
OOD=""
MODELS=("llama_8b")  # "llama_8b" "qwen_06b"
PROBE_MODES=("logistic")
TOKEN_MODE="last_token"

SMOKE_TEST=0
NORMALIZE_SCORES=1
ABLATION_SET="all"  # human | machine | all

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for PROBE_MODE in "${PROBE_MODES[@]}"; do
            echo "------------------------------------------------"
            echo "Running Probe: Dataset=$DATASET, Model=$MODEL, Mode=$PROBE_MODE, TokenMode=$TOKEN_MODE"
            echo "SmokeTest=$SMOKE_TEST, OOD=$OOD, NormalizeScores=$NORMALIZE_SCORES, AblationSet=$ABLATION_SET"
            echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run -m src.probes.run \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --mode "$PROBE_MODE" \
                --token_mode "$TOKEN_MODE" \
                --smoke_test "$SMOKE_TEST" \
                --ood "$OOD" \
                --normalize_scores "$NORMALIZE_SCORES" \
                --ablation_set "$ABLATION_SET"
        done
    done
done
