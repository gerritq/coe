from argparse import ArgumentParser, Namespace
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference import Inference


BASE_DIR = os.getenv("BASE_COE", ".")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
SEED = 42


def load_train_items(dataset_name: str, smoke_test: bool) -> list[dict]:
    path = os.path.join(DATA_DIR, dataset_name, "train.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if not smoke_test:
        return items

    by_label = {0: [], 1: []}
    for item in items:
        label = int(item["label"])
        if label in by_label:
            by_label[label].append(item)

    random.seed(SEED)
    sampled = []
    for label in [0, 1]:
        random.shuffle(by_label[label])
        sampled.extend(by_label[label][:100])
    return sampled


def collect_middle_layer_embeddings(items: list[dict], model_name: str) -> tuple[np.ndarray, np.ndarray]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode="last_token")

    x = []
    y = []
    for item in items:
        out = inference.run(item=item, args=infer_args)
        # token representation from the middle layer
        hidden_states = out["hidden_states"]
        mid_idx = len(hidden_states) // 2
        vec = hidden_states[mid_idx].detach().to(torch.float32).cpu().numpy()
        x.append(vec)
        y.append(int(out["label"]))

    return np.stack(x, axis=0), np.asarray(y, dtype=np.int32)


def cumulative_variance_curve(x: np.ndarray) -> tuple[np.ndarray, int]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x_centered, compute_uv=False)
    var = np.cumsum(s ** 2) / np.sum(s ** 2)
    k90 = int(np.argmax(var >= 0.90) + 1)
    return var, k90


def plot_domain_curves(dataset_name: str, x: np.ndarray, y: np.ndarray, model_name: str, out_path: str) -> None:
    x_human = x[y == 0]
    x_ai = x[y == 1]
    if len(x_human) == 0 or len(x_ai) == 0:
        raise RuntimeError(f"{dataset_name}: expected both labels 0 and 1.")

    var_ai, k90_ai = cumulative_variance_curve(x_ai)
    var_human, k90_human = cumulative_variance_curve(x_human)

    n_plot = min(500, len(var_ai), len(var_human))
    xs = np.arange(1, n_plot + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, var_ai[:n_plot], label=f"AI (k90={k90_ai})", linewidth=2)
    plt.plot(xs, var_human[:n_plot], label=f"Human (k90={k90_human})", linewidth=2)
    plt.axhline(0.90, linestyle="--", color="gray", label="90% threshold")
    plt.xlabel("Number of components (k)")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"{dataset_name}: AI vs Human rank profile ({model_name})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

    print(f"{dataset_name} | AI k90={k90_ai} | Human k90={k90_human}")
    print(f"Saved figure: {out_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    smoke = bool(args.smoke_test)
    suffix = "_smoke" if smoke else ""

    for dataset_name in ["raidDomain_wiki", "raidDomain_reddit", "multisocial_de", "multisocial_en"]:
        items = load_train_items(dataset_name=dataset_name, smoke_test=smoke)
        x, y = collect_middle_layer_embeddings(items=items, model_name=args.model)
        out_path = os.path.join(OUT_DIR, f"{dataset_name}_cumvar_{args.model}{suffix}.pdf")
        plot_domain_curves(
            dataset_name=dataset_name,
            x=x,
            y=y,
            model_name=args.model,
            out_path=out_path,
        )


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
