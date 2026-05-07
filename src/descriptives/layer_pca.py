import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.inference import Inference
import torch

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


def group_items_by_domain(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        domain = str(item.get("source", "unknown"))
        groups.setdefault(domain, []).append(item)
    return groups


def select_five_layers(n_layers: int) -> list[int]:
    idx = np.linspace(0, n_layers - 1, num=5, dtype=int).tolist()
    return sorted(set(idx))


def collect_hidden_states(
    items: list[dict[str, Any]],
    model_name: str,
    token_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode=token_mode)

    all_hidden = []
    labels = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hidden_states = out["hidden_states"]
        sample_layers = [h.detach().to(torch.float32).cpu().numpy() for h in hidden_states]
        all_hidden.append(np.stack(sample_layers, axis=0))
        labels.append(int(out["label"]))

    x = np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def project_layer_pca(x_layer: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(x_layer)


def plot_layer_projections(
    x: np.ndarray,
    y: np.ndarray,
    layer_indices: list[int],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(4.2 * len(layer_indices), 4.0), squeeze=False)
    axes = axes[0]

    unique_labels = sorted(set(y.tolist()))
    cmap = plt.get_cmap("tab10")
    label_to_color = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}
    label_to_name = {0: "Human", 1: "Machine"}

    for panel_idx, (ax, layer_idx) in enumerate(zip(axes, layer_indices)):
        x_2d = project_layer_pca(x[:, layer_idx, :])
        for lab in unique_labels:
            mask = y == lab
            legend_label = label_to_name.get(lab, str(lab))
            ax.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                s=10,
                alpha=0.65,
                c=[label_to_color[lab]],
                label=legend_label,
                edgecolors="none",
            )
        if panel_idx == 0:
            ax.set_title("Layer 0")
        else:
            ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        if panel_idx == len(layer_indices) - 1:
            ax.legend(title="Label", loc="best")
    fig.suptitle("Per-layer PCA Projections (2D)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(BASE_DIR, "data", "sets", "d_m4_domains", "data.jsonl")
    items = load_jsonl(data_path, smoke_test=bool(args.smoke_test))
    by_domain = group_items_by_domain(items)

    suffix = "_smoke" if bool(args.smoke_test) else ""

    for domain, domain_items in sorted(by_domain.items()):
        x, y = collect_hidden_states(items=domain_items, model_name=args.model, token_mode="last_token")
        layer_indices = select_five_layers(x.shape[1])

        safe_domain = domain.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(
            OUT_DIR,
            f"d_m4_layer_pca_{safe_domain}_{args.model}{suffix}.pdf",
        )
        plot_layer_projections(x=x, y=y, layer_indices=layer_indices, out_path=out_path)
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
