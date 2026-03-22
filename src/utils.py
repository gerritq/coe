import os
import json
import random
import numpy as np

from typing import Any
from argparse import Namespace

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
TEXT_PREFIX = "Is this text human- or LLM-written?"

def load_dataset(args: Namespace):

    random.seed(42)

    with open(os.path.join(DATA_DIR, f"{args.dataset}.jsonl"), "r") as f:
        raw_data = []
        for line in f:
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        data = []
        for item in raw_data:
            human_text = item.get("human_text", item.get("text", item.get("correct")))
            machine_text = item.get("machine_text", item.get("incorrect"))
            if (human_text is None or human_text.strip() == "") or machine_text is None or machine_text.strip() == "":
                continue

            if args.prefix:
                human_text = f"{TEXT_PREFIX} {human_text}"
                machine_text = f"{TEXT_PREFIX} {machine_text}"

            data.append({'text': human_text,
                         'label': 0})
            data.append({'text': machine_text,
                         'label': 1})

    random.shuffle(data)

    if args.smoke_test:
        data = data[:30]

    print("=" * 50)
    print(f"Loaded {len(data)} samples for dataset: {args.dataset}")
    print("=" * 50)
    
    return data[:args.n]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "prec": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
    }

def return_args(args: Any | None) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    return {"value": str(args)}
