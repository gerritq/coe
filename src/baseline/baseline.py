import os
import json
from argparse import ArgumentParser, Namespace
import numpy as np
from typing import Any
import torch
import sys

from transformers import set_seed

from src.utils import load_dataset, evaluation, return_args

BASE_DIR = os.getenv("BASE_COE")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
os.makedirs(BASELINE_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

TEST_SIZE = 0.2
SEED = 42
set_seed(SEED)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
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

    # get data
    data = load_dataset(args=args)
    val_x = data["val"]["text"]
    val_y = data["val"]["label"]
    test_x = data["test"]["text"]
    test_y = data["test"]["label"]

    args.n = len(test_x)
    
    val_scores = None
    test_scores = None

    if args.model == "encoder":
        from src.baseline.enc import EncoderBaseline
        baseline = EncoderBaseline(model_name="google-bert/bert-base-uncased", device=DEVICE)
        scores = baseline.run(args=args, data=data)
        val_scores = scores["val_scores"]
        test_scores = scores["test_scores"]
        
    if args.model == "binoculars":
        from src.baseline.binoculars import Binoculars
        baseline = Binoculars(
                            #   observer_name_or_path=args.base_model_1, 
                            #   performer_name_or_path=args.base_model_2
                              )
        val_scores = baseline.run(input_text=val_x)
        test_scores = baseline.run(input_text=test_x)
    
    if args.model == "llr":
        from src.baseline.llr import LLR
        baseline = LLR(model_name=args.base_model_1, device=DEVICE)
        val_scores = baseline.run(texts=val_x, args=args)
        test_scores = baseline.run(texts=test_x, args=args)

    if args.model == "likelihood":
        from src.baseline.llr import LLR
        baseline = LLR(model_name=args.base_model_1, device=DEVICE)
        val_scores = baseline.run(texts=val_x, args=args)
        test_scores = baseline.run(texts=test_x, args=args)

    if args.model == "rank":
        from src.baseline.rank import Rank
        baseline = Rank(model_name=args.base_model_1, device=DEVICE)
        val_scores = baseline.run(texts=val_x)
        test_scores = baseline.run(texts=test_x)

    if args.model == "entropy":
        from src.baseline.entropy import Entropy
        baseline = Entropy(model_name=args.base_model_1, device=DEVICE)
        val_scores = baseline.run(texts=val_x)
        test_scores = baseline.run(texts=test_x)

    if args.model == "fastdetectgpt":
        from src.baseline.fastdetectgpt import FastDetectGPT
        baseline = FastDetectGPT(scoring_model=args.base_model_1, 
                                 reference_model=args.base_model_1,
                                 device=DEVICE)
        val_scores = baseline.run(texts=val_x)
        test_scores = baseline.run(texts=test_x)


    metrics = evaluation(
        y_true=test_y,
        y_predict=test_scores,
        y_val_true=val_y,
        y_val_predict=val_scores,
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
