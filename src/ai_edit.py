import csv
import json
import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.inference import Inference
from src.utils import return_device


BASE_DIR = Path(os.getenv("BASE_COE", "."))
SCRATCH_EDITLENS_PATH = Path("/scratch/prj/inf_nlg_ai_detection/coe/data/raw/editlens/train.csv")
LOCAL_EDITLENS_PATH = Path(
    "/Users/k21157437/Library/CloudStorage/OneDrive-King'sCollegeLondon/phd/projects/coe/raw/editlens/train.csv"
)
EDITLENS_PATH = Path(os.getenv("EDITLENS_PATH", SCRATCH_EDITLENS_PATH if SCRATCH_EDITLENS_PATH.exists() else LOCAL_EDITLENS_PATH))
OUT_PATH = BASE_DIR / "notes" / "rebuttal" / "ai_edit.json"
EDITLENS_MODEL = "pangram/editlens_roberta-large"

LABEL_ORDER = ["human_written", "ai_edited", "ai_generated"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
DEFAULT_SPLIT_SIZES = {"train": 1000, "val": 200, "test": 200}


def load_editlens_csv(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("text")
            text_type = row.get("text_type")
            if not text or text_type not in LABEL_TO_ID:
                continue
            items.append({
                "text": text,
                "text_type": text_type,
                "label": LABEL_TO_ID[text_type],
            })
    return items


def balanced_split(
    items: list[dict[str, Any]],
    sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = {label: [] for label in LABEL_ORDER}
    for item in items:
        by_label[ID_TO_LABEL[int(item["label"])]].append(item)

    out: dict[str, list[dict[str, Any]]] = {}
    offsets = {label: 0 for label in LABEL_ORDER}
    for label_items in by_label.values():
        rng.shuffle(label_items)

    for split_name, requested_size in sizes.items():
        per_label = requested_size // len(LABEL_ORDER)
        split_items = []
        for label in LABEL_ORDER:
            start = offsets[label]
            end = start + per_label
            if end > len(by_label[label]):
                raise ValueError(f"Not enough examples for label={label} to create balanced splits.")
            split_items.extend(by_label[label][start:end])
            offsets[label] = end
        rng.shuffle(split_items)
        out[split_name] = split_items

    return out


def tune_two_thresholds(scores: np.ndarray, labels: np.ndarray) -> list[float]:
    unique_scores = np.unique(scores.astype(np.float64))
    if len(unique_scores) < 2:
        return [float(unique_scores[0]), float(unique_scores[0])]

    candidates = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    best_thresholds = [float(np.quantile(scores, 1 / 3)), float(np.quantile(scores, 2 / 3))]
    best_f1 = -1.0
    for low in candidates:
        for high in candidates:
            if high <= low:
                continue
            pred = labels_from_thresholds(scores, [float(low), float(high)])
            score = float(f1_score(labels, pred, average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thresholds = [float(low), float(high)]
    return best_thresholds


def labels_from_thresholds(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.digitize(scores, np.asarray(thresholds, dtype=np.float64), right=False).astype(int)


def labels_array(items: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(item["label"]) for item in items], dtype=np.int32)


def expected_label_score(probs: np.ndarray) -> np.ndarray:
    class_ids = np.arange(probs.shape[1], dtype=np.float64)
    return probs @ class_ids


def evaluate_predictions(pred_labels: np.ndarray, items: list[dict[str, Any]]) -> dict[str, Any]:
    gold_labels = labels_array(items)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(gold_labels, pred_labels)),
        "macro_f1": float(f1_score(gold_labels, pred_labels, average="macro", zero_division=0)),
        "by_group": {},
    }
    per_label_f1 = f1_score(gold_labels, pred_labels, labels=list(range(len(LABEL_ORDER))), average=None, zero_division=0)
    for label_id, label_name in ID_TO_LABEL.items():
        mask = gold_labels == label_id
        out["by_group"][label_name] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(gold_labels[mask], pred_labels[mask])) if mask.any() else 0.0,
            "f1": float(per_label_f1[label_id]),
        }
    return out


def train_bert(args: Namespace, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> dict[str, Any]:
    set_seed(args.seed)
    device = return_device()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=len(LABEL_ORDER))
    model.to(device)

    data = DatasetDict({
        "train": Dataset.from_list(train_items),
        "val": Dataset.from_list(val_items),
        "test": Dataset.from_list(test_items),
    })

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.bert_max_length)

    tok_data = data.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "output" / "tmp" / "ai_edit_bert"),
        per_device_train_batch_size=args.bert_batch_size,
        per_device_eval_batch_size=args.bert_batch_size,
        num_train_epochs=args.bert_epochs,
        learning_rate=args.bert_lr,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
        logging_steps=25,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_data["train"],
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    val_logits = torch.from_numpy(trainer.predict(tok_data["val"]).predictions)
    test_logits = torch.from_numpy(trainer.predict(tok_data["test"]).predictions)
    val_probs = torch.nn.functional.softmax(val_logits, dim=-1).numpy()
    test_probs = torch.nn.functional.softmax(test_logits, dim=-1).numpy()
    val_scores = expected_label_score(val_probs)
    test_scores = expected_label_score(test_probs)

    thresholds = tune_two_thresholds(val_scores, labels_array(val_items))
    return {
        "thresholds": thresholds,
        "test_labels": labels_from_thresholds(test_scores, thresholds),
    }


def eval_editlens(args: Namespace, val_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> dict[str, Any]:
    from src.baseline.mlmodel import MLModels

    model = MLModels(model_name=EDITLENS_MODEL)
    model_args = Namespace(model="editlens")
    val_scores = np.asarray(model.run([item["text"] for item in val_items], model_args), dtype=np.float64)
    test_scores = np.asarray(model.run([item["text"] for item in test_items], model_args), dtype=np.float64)

    thresholds = tune_two_thresholds(val_scores, labels_array(val_items))
    return {
        "thresholds": thresholds,
        "test_labels": labels_from_thresholds(test_scores, thresholds),
    }


def collect_hidden_states(items: list[dict[str, Any]], args: Namespace) -> dict[str, np.ndarray]:
    inference = Inference(args.llp_model)
    hidden_states = []
    labels = []

    probe_args = Namespace(model=args.llp_model, token_mode=args.token_mode)
    for item in items:
        out = inference.run(item=item, args=probe_args)
        layers = [layer.detach().to(torch.float32).cpu().numpy() for layer in out["hidden_states"]]
        hidden_states.append(np.stack(layers, axis=0))
        labels.append(int(item["label"]))

    x = np.stack(hidden_states, axis=0)  # (n_samples, n_layers, d_model)
    x = np.transpose(x, (1, 0, 2))  # (n_layers, n_samples, d_model)
    y = np.asarray(labels, dtype=np.int32)
    return {"hidden_x": x, "y": y}


def train_probes(args: Namespace, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> dict[str, Any]:
    train = collect_hidden_states(train_items, args)
    val = collect_hidden_states(val_items, args)
    test = collect_hidden_states(test_items, args)

    all_train_scores = []
    all_val_scores = []
    all_test_scores = []

    for layer in range(train["hidden_x"].shape[0]):
        x_layer_train = train["hidden_x"][layer]  # (n_samples, d_model)

        scaler = StandardScaler()
        x_layer_train_scaled = scaler.fit_transform(x_layer_train)

        if args.llp_mode == "pca":
            pca = PCA(n_components=args.components, random_state=args.seed)
            x_layer_train_scaled = pca.fit_transform(x_layer_train_scaled)
        else:
            pca = None

        model = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
        model.fit(x_layer_train_scaled, train["y"])

        x_layer_train_meta = scaler.transform(train["hidden_x"][layer])
        x_layer_val = scaler.transform(val["hidden_x"][layer])
        x_layer_test = scaler.transform(test["hidden_x"][layer])
        if pca is not None:
            x_layer_train_meta = pca.transform(x_layer_train_meta)
            x_layer_val = pca.transform(x_layer_val)
            x_layer_test = pca.transform(x_layer_test)

        train_score = expected_label_score(model.predict_proba(x_layer_train_meta))
        val_score = expected_label_score(model.predict_proba(x_layer_val))
        test_score = expected_label_score(model.predict_proba(x_layer_test))

        all_train_scores.append(np.nan_to_num(train_score, nan=0.0, posinf=0.0, neginf=0.0))
        all_val_scores.append(np.nan_to_num(val_score, nan=0.0, posinf=0.0, neginf=0.0))
        all_test_scores.append(np.nan_to_num(test_score, nan=0.0, posinf=0.0, neginf=0.0))

    train_layer_scores = np.stack(all_train_scores, axis=1)
    val_layer_scores = np.stack(all_val_scores, axis=1)
    test_layer_scores = np.stack(all_test_scores, axis=1)

    llp_val_scores = val_layer_scores.mean(axis=1)
    llp_test_scores = test_layer_scores.mean(axis=1)
    llp_thresholds = tune_two_thresholds(llp_val_scores, val["y"])

    meta = LogisticRegression(max_iter=2000, random_state=args.seed, C=1.0)
    meta.fit(train_layer_scores, train["y"])
    clp_val_scores = expected_label_score(meta.predict_proba(val_layer_scores))
    clp_test_scores = expected_label_score(meta.predict_proba(test_layer_scores))
    clp_thresholds = tune_two_thresholds(clp_val_scores, val["y"])

    return {
        "llp": {
            "thresholds": llp_thresholds,
            "test_labels": labels_from_thresholds(llp_test_scores, llp_thresholds),
        },
        "clp": {
            "thresholds": clp_thresholds,
            "test_labels": labels_from_thresholds(clp_test_scores, clp_thresholds),
        },
    }


def split_sizes() -> dict[str, int]:
    return {
        split: (size // len(LABEL_ORDER)) * len(LABEL_ORDER)
        for split, size in DEFAULT_SPLIT_SIZES.items()
    }


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=Path, default=EDITLENS_PATH)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_epochs", type=int, default=2)
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--bert_max_length", type=int, default=256)
    parser.add_argument("--llp_model", type=str, default="llama_8b")
    parser.add_argument("--llp_mode", type=str, choices=["default", "pca"], default="default")
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--token_mode", type=str, choices=["last_token", "pooling"], default="last_token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_items = load_editlens_csv(args.data_path)
    splits = balanced_split(all_items, split_sizes(), args.seed)
    train_items = splits["train"]
    val_items = splits["val"]
    test_items = splits["test"]

    bert_out = train_bert(args, train_items, val_items, test_items)
    editlens_out = eval_editlens(args, val_items, test_items)
    probe_out = train_probes(args, train_items, val_items, test_items)

    out = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "path": str(args.data_path),
            "labels": LABEL_ORDER,
            "requested_sizes": DEFAULT_SPLIT_SIZES,
            "train_size": len(train_items),
            "validation_size": len(val_items),
            "test_size": len(test_items),
        },
        "models": {
            "encoder": {
                "validation_score_thresholds": bert_out["thresholds"],
                "metrics": evaluate_predictions(bert_out["test_labels"], test_items),
            },
            "editlens": {
                "validation_score_thresholds": editlens_out["thresholds"],
                "metrics": evaluate_predictions(editlens_out["test_labels"], test_items),
            },
            "llp": {
                "validation_score_thresholds": probe_out["llp"]["thresholds"],
                "metrics": evaluate_predictions(probe_out["llp"]["test_labels"], test_items),
            },
            "clp": {
                "validation_score_thresholds": probe_out["clp"]["thresholds"],
                "metrics": evaluate_predictions(probe_out["clp"]["test_labels"], test_items),
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
