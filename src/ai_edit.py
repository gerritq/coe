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
APT_DIR = BASE_DIR / "data" / "sets" / "apt"
OUT_PATH = BASE_DIR / "notes" / "rebuttal" / "ai_edit.json"

METRICS = ["sem_similarity", "levenshtein_distance", "jaccard_distance"]
METRIC_DIRECTIONS = {
    "sem_similarity": -1.0,
    "levenshtein_distance": 1.0,
    "jaccard_distance": 1.0,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_train_val(items: list[dict[str, Any]], train_frac: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    train_size = int(round(len(items) * train_frac))
    return items[:train_size], items[train_size:]


def valid_metric_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if all(item.get(metric) is not None for metric in METRICS)]


def quantile_thresholds(scores: np.ndarray) -> list[float]:
    return np.quantile(scores, [0.25, 0.50, 0.75]).astype(float).tolist()


def labels_from_thresholds(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.digitize(scores, np.asarray(thresholds, dtype=np.float64), right=False).astype(int)


def metric_gold_labels(items: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, list[float]]:
    values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
    edit_strength = values * METRIC_DIRECTIONS[metric]
    thresholds = quantile_thresholds(edit_strength)
    return labels_from_thresholds(edit_strength, thresholds), thresholds


def evaluate_bins(pred_labels: np.ndarray, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for metric in METRICS:
        gold_labels, _ = metric_gold_labels(items, metric)
        out[metric] = {
            "accuracy": float(accuracy_score(gold_labels, pred_labels)),
            "macro_f1": float(f1_score(gold_labels, pred_labels, average="macro")),
        }
    return out


def train_bert(args: Namespace, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> dict[str, Any]:
    set_seed(args.seed)
    device = return_device()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
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
    val_scores = torch.nn.functional.softmax(val_logits, dim=-1).numpy()[:, 1]
    test_scores = torch.nn.functional.softmax(test_logits, dim=-1).numpy()[:, 1]

    thresholds = quantile_thresholds(val_scores)
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


def train_llp(args: Namespace, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> dict[str, Any]:
    train = collect_hidden_states(train_items, args)
    val = collect_hidden_states(val_items, args)
    test = collect_hidden_states(test_items, args)

    all_val_projections = []
    all_test_projections = []

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

        probe_vector = model.coef_[0]
        probe_vector = probe_vector / np.linalg.norm(probe_vector)

        x_layer_val = scaler.transform(val["hidden_x"][layer])
        x_layer_test = scaler.transform(test["hidden_x"][layer])
        if pca is not None:
            x_layer_val = pca.transform(x_layer_val)
            x_layer_test = pca.transform(x_layer_test)

        # Raw LLP projection scores, matching probe_main.py.
        val_projection = np.dot(x_layer_val, probe_vector)
        test_projection = np.dot(x_layer_test, probe_vector)

        all_val_projections.append(np.nan_to_num(val_projection, nan=0.0, posinf=0.0, neginf=0.0))
        all_test_projections.append(np.nan_to_num(test_projection, nan=0.0, posinf=0.0, neginf=0.0))

    val_scores = np.stack(all_val_projections, axis=0).mean(axis=0)
    test_scores = np.stack(all_test_projections, axis=0).mean(axis=0)

    thresholds = quantile_thresholds(val_scores)
    return {
        "thresholds": thresholds,
        "test_labels": labels_from_thresholds(test_scores, thresholds),
    }


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
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

    train_pool = load_jsonl(APT_DIR / "train.jsonl")
    test_pool = load_jsonl(APT_DIR / "test.jsonl")

    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train_frac must be between 0 and 1")

    train_items, val_items = split_train_val(train_pool, args.train_frac, args.seed)
    test_eval_items = valid_metric_items(test_pool)

    bert_out = train_bert(args, train_items, val_items, test_eval_items)
    llp_out = train_llp(args, train_items, val_items, test_eval_items)

    out = {
        "args": vars(args),
        "data": {
            "train_size": len(train_items),
            "validation_size": len(val_items),
            "test_size": len(test_eval_items),
        },
        "gold_metric_thresholds": {
            metric: metric_gold_labels(test_eval_items, metric)[1]
            for metric in METRICS
        },
        "models": {
            "bert": {
                "validation_score_thresholds": bert_out["thresholds"],
                "metrics": evaluate_bins(bert_out["test_labels"], test_eval_items),
            },
            "llp": {
                "validation_score_thresholds": llp_out["thresholds"],
                "metrics": evaluate_bins(llp_out["test_labels"], test_eval_items),
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
