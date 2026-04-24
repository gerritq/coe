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
DATA_DIR = os.path.join(BASE_DIR, "data")

TEXT_PREFIX = "Is this text human- or LLM-written?"

def load_dataset(args: Namespace):
    
    data_path = os.path.join(DATA_DIR, f"{args.dataset}")
    data = DatasetDict({
        split: Dataset.from_json(os.path.join(data_path, f"{split}.jsonl"))
        for split in ["train", "val", "test"]
    })

    if args.prefix:
        data = data.map(lambda x: {"text": f"{TEXT_PREFIX} {x['text']}", "label": x["label"]})

    if args.smoke_test:
            # We wrap the dict comprehension in DatasetDict() to maintain the type
            data = DatasetDict({
                split: d.shuffle(seed=42).select(range(min(len(d), 30))) 
                for split, d in data.items()
            })

    return data

def optimal_threshold(
    y_true: np.ndarray,
    y_predict: np.ndarray,
    num_thresholds: int = 1001,
) -> dict[str, float]:
    """Determine thresholds that maximize F1 and accuracy on a dev set."""
    labels = np.asarray(y_true).astype(int).reshape(-1)
    preds = np.asarray(y_predict).astype(float).reshape(-1)
    thresholds = np.linspace(0.0, 1.0, int(num_thresholds))

    best_threshold_f1 = 0.0
    best_f1 = -1.0
    best_threshold_acc = 0.0
    best_acc = -1.0

    for threshold in thresholds:
        pred_labels = (preds >= threshold).astype(int)

        tp = np.sum((pred_labels == 1) & (labels == 1))
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
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


def evaluation(
    y_true: np.ndarray,
    y_predict: np.ndarray,
    y_val_true: np.ndarray | None = None,
    y_val_predict: np.ndarray | None = None,
) -> dict[str, float]:
    
    def _valid_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        labels_raw = np.asarray(labels).reshape(-1)
        scores_raw = np.asarray(scores, dtype=object).reshape(-1)
        n_total_local = int(len(labels_raw))
        valid_labels_local: list[int] = []
        valid_scores_local: list[float] = []

        for idx in range(n_total_local):
            if idx >= len(scores_raw):
                continue
            try:
                score = float(scores_raw[idx])
                label = int(labels_raw[idx])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            valid_labels_local.append(label)
            valid_scores_local.append(score)

        return (
            np.asarray(valid_labels_local, dtype=int),
            np.asarray(valid_scores_local, dtype=float),
            n_total_local,
            int(len(valid_scores_local)),
        )

    y_true_arr, y_score_arr, n_total, n_valid = _valid_arrays(y_true, y_predict)
    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": n_valid,
            "acc": float("nan"),
            "f1": float("nan"),
            "pre": float("nan"),
            "recall": float("nan"),
            "auroc": float("nan"),
            "optimal_threshold": float("nan"),
            "tpr_at_fpr_0_01": float("nan"),
            "fpr95": float("nan"),
            "aupr": float("nan"),
        }

    if y_val_true is not None and y_val_predict is not None:
        y_val_true_arr, y_val_score_arr, n_val_total, n_val_valid = _valid_arrays(y_val_true, y_val_predict)
    else:
        y_val_true_arr, y_val_score_arr = y_true_arr, y_score_arr
        n_val_total, n_val_valid = n_total, n_valid

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    if n_val_valid > 0 and len(np.unique(y_val_true_arr)) > 1:
        threshold_stats = optimal_threshold(y_val_true_arr, y_val_score_arr)
    else:
        threshold_stats = optimal_threshold(y_true_arr, y_score_arr)

    optimal_thresh_f1 = float(threshold_stats["threshold_f1"])
    optimal_thresh_acc = float(threshold_stats["threshold_acc"])

    y_pred_f1 = (y_score_arr >= optimal_thresh_f1).astype(int)
    y_pred_acc = (y_score_arr >= optimal_thresh_acc).astype(int)

    fpr_target = 0.01
    mask_fpr = fpr <= fpr_target
    tpr_at_fpr_0_01 = float(np.max(tpr[mask_fpr])) if np.any(mask_fpr) else 0.0

    tpr_target = 0.95
    mask_tpr = tpr >= tpr_target
    fpr95 = float(np.min(fpr[mask_tpr])) if np.any(mask_tpr) else 1.0

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_val_total": n_val_total,
        "n_val_valid": n_val_valid,
        "acc": float(accuracy_score(y_true_arr, y_pred_acc)),
        "f1": float(f1_score(y_true_arr, y_pred_f1, average="binary", zero_division=0)),
        "pre": float(precision_score(y_true_arr, y_pred_f1, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_f1, average="binary", zero_division=0)),
        "auroc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "optimal_threshold": optimal_thresh_f1,
        "optimal_threshold_f1": optimal_thresh_f1,
        "optimal_threshold_acc": optimal_thresh_acc,
        "best_f1_on_threshold_sweep": float(threshold_stats["best_f1"]),
        "best_acc_on_threshold_sweep": float(threshold_stats["best_acc"]),
        "tpr_at_fpr_0_01": tpr_at_fpr_0_01,
        "fpr95": fpr95,
        "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
    }


# def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
#     return {
#         "acc": float(accuracy_score(y_true, y_pred)),
#         "f1": float(f1_score(y_true, y_pred, average="binary")),
#         "prec": float(precision_score(y_true, y_pred, average="binary")),
#         "recall": float(recall_score(y_true, y_pred, average="binary")),
#     }

def return_args(args: Any | None) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    return {"value": str(args)}


# def compute_auc_for_scores(
#     out: list[dict],
#     args: Any | None,
#     score_keys: list[str],
#     ) -> dict[str, Any]:

#     ZERO_SCORES_DIR = os.path.join(BASE_DIR, "scores", "test")
#     os.makedirs(ZERO_SCORES_DIR, exist_ok=True)


#     results: dict[str, Any] = {"args": return_args(args), "metrics": {}}

#     for key in score_keys:
#         values = []
#         labels = []
#         for item in out:
#             val = item.get(key)
#             if val is None:
#                 continue
#             if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
#                 continue
#             values.append(float(val))
#             labels.append(int(item["label"]))

    
#         y_true = np.array(labels)
#         y_score = np.array(values)
#         fpr, tpr, thresholds = roc_curve(y_true, y_score)
#         auc_val = float(roc_auc_score(y_true, y_score))
#         results["metrics"][key] = {
#             "auc": auc_val,
#             "fpr": fpr.tolist(),
#             "tpr": tpr.tolist(),
#             "thresholds": thresholds.tolist(),
#             "n": int(len(values)),
#         }

#     out_path = os.path.join(ZERO_SCORES_DIR, f"auroc_{args.suffix.replace(".pdf", ".json")}")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     return results

def return_args(args: Any | None) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    return {"value": str(args)}
