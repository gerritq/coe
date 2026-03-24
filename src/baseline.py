import os
import json
from argparse import ArgumentParser, Namespace
import numpy as np
from typing import Any
import torch

from transformers import set_seed

from sklearn.model_selection import train_test_split

from utils import load_dataset, return_args
from baselines.utils import compute_auc

BASE_DIR = os.getenv("BASE_COE")
BASELINE_DIR = os.path.join(BASE_DIR, "baselines", "test")
os.makedirs(BASELINE_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

TEST_SIZE = 0.2
SEED = 42

def prepare_dataset(args: Namespace) -> tuple[list[str], list[int], list[str], list[int]]:
    data = load_dataset(args)
    
    # train and test split
    return train_test_split(
        [item["text"] for item in data],
        [item["label"] for item in data],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=[item["label"] for item in data],
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--prefix", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    args.smoke_test = bool(args.smoke_test)
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    args.prefix = bool(args.prefix)

    # select base models 
    if args.smoke_test:
        args.base_model_1 = "Qwen/Qwen3-0.6B-Base"
        args.base_model_2 = "Qwen/Qwen3-0.6B"
    else:
        args.base_model_1 = "meta-llama/Meta-Llama-3-8B"
        args.base_model_2 = "meta-llama/Meta-Llama-3-8B-instruct"

    # prep data
    splits = prepare_dataset(args) # x_train, x_test, y_train, y_test
    labels = splits[3]  # y_test
    
    if args.model == "encoder":
        from baselines.enc import EncoderBaseline
        baseline = EncoderBaseline(model_name="google-bert/bert-base-uncased", device=DEVICE)
        scores = baseline.run(args=args, splits=splits)
        
    if args.model == "binoculars":
        from baselines.binoculars import Binoculars
        baseline = Binoculars(observer_name_or_path=args.base_model_1, 
                              performer_name_or_path=args.base_model_2)
        scores = baseline.run(input_text=splits[2])
    
    if args.model == "llr":
        from baselines.llr import LLR
        baseline = LLR(model_name=args.base_model_1, device=DEVICE)
        scores = baseline.run(texts=splits[2])

    if args.model == "rank":
        from baselines.rank import Rank
        baseline = Rank(model_name=args.base_model_1, device=DEVICE)
        scores = baseline.run(texts=splits[2])

    if args.model == "entropy":
        from baselines.entropy import Entropy
        baseline = Entropy(model_name=args.base_model_1, device=DEVICE)
        scores = baseline.run(texts=splits[2])

    if args.model == "fastdetectgpt":
        from baselines.fastdetectgpt import FastDetectGPT
        baseline = FastDetectGPT(scoring_model=args.base_model_1, 
                                 reference_model=args.base_model_1,
                                 device=DEVICE)
        scores = baseline.run(texts=splits[2])

    metrics = compute_auc(
        labels=labels,
        values=scores,
    )

    #save results
    out_path = os.path.join(BASELINE_DIR, f"{args.model}_{args.dataset}.json")
    payload = {
        "metrics": metrics,
        "args": return_args(args),
        }
        
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
