


DATASET="wikipedia_chatgpt"
MODEL="qwen_06b"
SMOKE_TEST=0
N=30

uv run run.py --dataset $DATASET \
                --model $MODEL \
                --smoke_test $SMOKE_TEST \
                --n $N