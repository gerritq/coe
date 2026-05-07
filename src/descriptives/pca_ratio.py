import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.inference import Inference


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
SEED = 42


def load_jsonl(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if not smoke_test:
        return items

    print(f"Running smoke test: sampling 20 items per label from {len(items)} total items.")
    by_label: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        label = int(item["label"])
        by_label.setdefault(label, []).append(item)

    random.seed(SEED)
    sampled = []
    for label in sorted(by_label.keys()):
        label_items = by_label[label]
        random.shuffle(label_items)
        sampled.extend(label_items[:20])
    return sampled


def collect_hidden_states(items: list[dict[str, Any]], model_name: str) -> np.ndarray:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode="last_token")

    all_hidden = []
    for item in items:
        out = inference.run(item=item, args=infer_args)
        hidden_states = out["hidden_states"]
        sample_layers = [h.detach().to(torch.float32).cpu().numpy() for h in hidden_states]
        all_hidden.append(np.stack(sample_layers, axis=0))  # (n_layers, d_model)

    return np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)


def select_eight_layers(n_layers: int) -> list[int]:
    idx = np.linspace(0, n_layers - 1, num=8, dtype=int).tolist()
    return sorted(set(idx))


def layer_cumulative_variance(x_layer: np.ndarray) -> np.ndarray:
    pca = PCA(random_state=SEED)
    pca.fit(x_layer)
    return np.cumsum(pca.explained_variance_ratio_)


def plot_cumulative_variance(x: np.ndarray, layer_indices: list[int], out_path: str) -> None:
    plt.figure(figsize=(9, 6))

    for panel_idx, layer_idx in enumerate(layer_indices):
        cum_ratio = layer_cumulative_variance(x[:, layer_idx, :])
        n_comp = np.arange(1, len(cum_ratio) + 1)
        label = "Layer 0" if panel_idx == 0 else f"Layer {layer_idx}"
        plt.plot(n_comp, cum_ratio, linewidth=2.0, label=label)

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative PCA Variance Ratio by Layer")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(BASE_DIR, "data", "sets", "d_m4_domains", "data.jsonl")
    items = load_jsonl(data_path, smoke_test=bool(args.smoke_test))

    x = collect_hidden_states(items=items, model_name=args.model)
    layer_indices = select_eight_layers(x.shape[1])

    suffix = "_smoke" if bool(args.smoke_test) else ""
    out_path = os.path.join(OUT_DIR, f"d_m4_pca_cumvar_{args.model}{suffix}.pdf")
    plot_cumulative_variance(x=x, layer_indices=layer_indices, out_path=out_path)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
