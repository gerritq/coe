import os
import json
import random
import numpy as np
import torch
from typing import Any
from argparse import Namespace
from datasets import Dataset, DatasetDict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

TEXT_PREFIX = "Is this text human- or LLM-written?"

"""
- hp sweep for 0-1 threshold only correct if the score are also in this range!
"""
def load_dataset(args: Namespace):
    
    data_path = os.path.join(DATA_DIR, f"{args.dataset}")
    data = DatasetDict({
        split: Dataset.from_json(os.path.join(data_path, f"{split}.jsonl"))
        for split in ["train", "val", "test"]
    })

    if args.smoke_test:
            # We wrap the dict comprehension in DatasetDict() to maintain the type
            data = DatasetDict({
                split: d.shuffle(seed=42).select(range(min(len(d), 30))) 
                for split, d in data.items()
            })

    return data


def optimal_thresholds(y_true: np.ndarray,
                       y_predict: np.ndarray,
                        ) -> dict[str, float]:
    
    n_thresholds = 1000
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))

    labels = np.asarray(y_true).astype(int).reshape(-1)
    preds = np.asarray(y_predict).astype(float).reshape(-1)

    best_threshold_f1 = 0.0
    best_f1 = -1.0
    best_threshold_acc = 0.0
    best_acc = -1.0

    for threshold in thresholds:
        pred_labels = (preds >= threshold).astype(int)

        # F1 score
        tp = np.sum((pred_labels == 1) & (labels == 1))
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        acc = float(np.mean(pred_labels == labels))

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold_f1 = float(threshold)

        if acc > best_acc:
            best_acc = float(acc)
            best_threshold_acc = float(threshold)

    return {
        "threshold_f1": best_threshold_f1,
        "best_f1": best_f1,
        "threshold_acc": best_threshold_acc,
        "best_acc": best_acc,
    }


def metrics(y_true: np.ndarray,
               y_predict: np.ndarray,
               f1_threshold: float,
               acc_threshold: float,
            ) -> dict[str, float]:
    
    # check for infinite or NaN values in y_predict
    if np.any(np.isinf(y_predict)) or np.any(np.isnan(y_predict)):
        raise ValueError("y_predict contains infinite or nan")

    # Acc and F1 with optimal thresholds    
    y_pred_acc = (y_predict >= acc_threshold).astype(int)
    y_pred_f1 = (y_predict >= f1_threshold).astype(int)

    # accuracy and F1 against ground truth labels
    acc = float(accuracy_score(y_true, y_pred_acc))
    f1 = float(f1_score(y_true, y_pred_f1, average="binary", zero_division=0))

    # AUROC
    auroc = float(roc_auc_score(y_true, y_predict))

    # AUPR
    aupr = float(average_precision_score(y_true, y_predict))

    # TPR@FPR=0.01
    fpr, tpr, _ = roc_curve(y_true, y_predict)

    mask_fpr = fpr <= 0.05
    tpr_at_fpr_0_05 = float(np.max(tpr[mask_fpr])) if np.any(mask_fpr) else 0.0

    return {
        "n_total": len(y_true),
        "auroc" : auroc,
        "accuracy": acc,
        "f1": f1,
        "tpr_at_fpr_0_05": tpr_at_fpr_0_05,
        "optimal_threshold_f1": f1_threshold,
        "optimal_threshold_acc": acc_threshold,
        "aupr": aupr,
    }



def return_args(args: Any | None) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    return {"value": str(args)}



def return_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE
