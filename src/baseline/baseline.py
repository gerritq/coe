import os
import json
import re
from argparse import ArgumentParser, Namespace
import torch
import sys
from transformers import set_seed
from src.utils import (load_dataset, 
                       optimal_thresholds,
                       metrics, 
                       return_args,
                       return_device,
                       OOD)

"""
- fix to onclude args in run
- add repreguard

"""


BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
os.makedirs(BASELINE_DIR, exist_ok=True)


DEVICE = return_device()
SEED = 42
set_seed(SEED)


def supervised_models(args):
    if args.model == "encoder":
        from src.baseline.enc import EncoderBaseline
        baseline = EncoderBaseline(model_name="google-bert/bert-base-uncased", device=DEVICE)

    if args.model == "repreguard":
        from src.baseline.repreguard.repre_main import RepreGuard
        baseline = RepreGuard(args=args)

    source_data = load_dataset(args=args)
    ood_data = []
    for target_dataset in OOD[args.dataset.split("_")[0]]:
        if not args.ood:
            if args.dataset != target_dataset:
                continue
        
        print("="*60)
        print(f"Loading target dataset: {target_dataset}")
        print("="*60)

        target_data = load_dataset(args=Namespace(dataset=target_dataset, smoke_test=args.smoke_test))
        ood_data.append({"data": target_data, "name": target_dataset})

    scores = baseline.run(args=args, training_data=source_data, ood_data=ood_data)

def return_model(args: Namespace):

    if args.model == "binoculars":
        from src.baseline.binoculars import Binoculars
        return Binoculars()

    if args.model in {"llr", "likelihood"}:
        from src.baseline.llr import LLR
        return LLR(model_name=args.base_model_1, device=DEVICE)

    if args.model == "rank":
        from src.baseline.rank import Rank
        return Rank(model_name=args.base_model_1, device=DEVICE)

    if args.model == "entropy":
        from src.baseline.entropy import Entropy
        return Entropy(model_name=args.base_model_1, device=DEVICE)

    if args.model == "fastdetectgpt":
        from src.baseline.fastdetectgpt import FastDetectGPT
        return FastDetectGPT(
            scoring_model=args.base_model_1,
            reference_model=args.base_model_1,
            device=DEVICE,
        )


def run(args):

    if args.model in ['encoder', "repreguard"]:
        supervised_models(args)
    
    else:
        # Source dataset for calibration.
        source_data = load_dataset(args=args)
        baseline = return_model(args)
        
        # VAL
        texts_val = source_data["val"]["text"]
        label_val_y = source_data["val"]["label"]
        scores_val = baseline.run(texts=texts_val, args=args)


        optimal_thresholds_dict = optimal_thresholds(y_true=label_val_y, 
                                                    y_predict=scores_val)

        # RUN EVAL
        for target_dataset in OOD[args.dataset.split("_")[0]]:
            if not args.ood:
                if args.dataset != target_dataset:
                    continue
            
            print("="*60)
            print(f"Evaluating on target dataset: {target_dataset}")
            print("="*60)

            # bit reduant for the ID case but whatever
            target_data = load_dataset(args=Namespace(dataset=target_dataset, smoke_test=args.smoke_test))
            target_y = target_data['test']["label"]
            target_text = target_data['test']["text"]
            
            # run on target
            target_scores = baseline.run(texts=target_text, args=args)

            # EVAL using the thesholds tuned on val
            test_metrics = metrics(y_true=target_y,
                                y_predict=target_scores,
                                f1_threshold=optimal_thresholds_dict["threshold_f1"],
                                acc_threshold=optimal_thresholds_dict["threshold_acc"])

        
            file_name = f"{args.model}_{args.dataset}_2_{target_dataset}.json"
            out_path = os.path.join(BASELINE_DIR, file_name)
            out = {"args": return_args(args),
                    "metrics": test_metrics}
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--ood", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)

    # select base models 
    if args.smoke_test:
        args.base_model_1 = "Qwen/Qwen3-0.6B-Base"
        args.base_model_2 = "Qwen/Qwen3-0.6B"
        # args.base_model_1 = "HuggingFaceTB/SmolLM-135M"
        # args.base_model_2 = "HuggingFaceTB/SmolLM-135M"

    else:
        args.base_model_1 = "meta-llama/Meta-Llama-3-8B"
        args.base_model_2 = "meta-llama/Meta-Llama-3-8B-instruct"

   
    run(args)

if __name__ == "__main__":
    main()
