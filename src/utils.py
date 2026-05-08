import os
import json
import random
import numpy as np
import torch
from typing import Any
from argparse import Namespace
from datasets import Dataset, DatasetDict
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score
import textdistance
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

OOD = {
    "drlDomain": sorted([x for x in os.listdir(DATA_DIR) if re.match(r"^drlDomain_.+", x)]),
    "drlAttack": sorted([x for x in os.listdir(DATA_DIR) if re.match(r"^drlAttack_.+", x)]),
    "multisocial": [
        "multisocial_en",
        "multisocial_de",
        "multisocial_ru",
        "multisocial_zh",
    ],
    "tsm": sorted([x for x in os.listdir(DATA_DIR) if re.match(r"^tsm_.+", x)]),
    "CB": sorted([x for x in os.listdir(DATA_DIR) if re.match(r"^CB_.+", x)]),
    "m4": sorted([x for x in os.listdir(DATA_DIR) if re.match(r"^m4_.+", x)]),
}

"""
- hp sweep for 0-1 threshold only correct if the score are also in this range!
"""
def load_dataset(args: Namespace):
    random.seed(42)
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

    if args.training_size is not None:
        pos = [x for x in data["train"] if x["label"] == 1][:args.training_size//2]
        neg = [x for x in data["train"] if x["label"] == 0][:args.training_size//2]
        data["train"] = Dataset.from_list(pos + neg)

        assert len(data["train"]) == args.training_size, f"Too few training samples: {len(data['train'])} < {args.training_size}"

        random.shuffle(data["train"])

        print("="*60)
        print(f"New training size: {len(data['train'])}")
        print("="*60)

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

    y_true = np.asarray(y_true).astype(int)
    y_predict = np.asarray(y_predict).astype(float)

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

# Taken from https://github.com/ShoumikSaha/ai-polished-text/blob/53b1d2619a633e025998d6c7d0a12b0228b589da/data/cal_distance.py#L35
class SimCalculator:
    def __init__(self, model_name="allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.device = return_device()
        self.model.to(self.device)

        # GQ: added 
        self.model.eval()

    def bert_similarity(self, text1, text2):
        inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
        inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)
        inputs1 = inputs1.to(self.device)
        inputs2 = inputs2.to(self.device)        
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)

        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu()
        return cos_sim.item()

    def cal_similarity(self, text1, text2):
        split_1 = text1.split(" ")
        split_2 = text2.split(" ")
        try:
            levenshtein_distance = textdistance.levenshtein.normalized_distance(split_1, split_2)
            jaccard_distance = textdistance.jaccard.normalized_distance(split_1, split_2)
            sem_similarity = self.bert_similarity(text1, text2)
            return_dict = {
                "levenshtein_distance": round(levenshtein_distance, 4),
                "jaccard_distance": round(jaccard_distance, 4),
                "sem_similarity": round(sem_similarity, 4)
            }
        except:
            return_dict = {
                "levenshtein_distance": 1.0,
                "jaccard_distance": 1.0,
                "sem_similarity": 0.0
            }
        return return_dict
