#!/bin/bash
#SBATCH --job-name=coe
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15GB
# SBATCH --constraint=a100



DATASET="wikipedia_chatgpt"
MODEL="qwen_8b" # qwen_8b qwen_06b
SMOKE_TEST=0
N=2000

uv run run.py --dataset $DATASET \
                --model $MODEL \
                --smoke_test $SMOKE_TEST \
                --n $N