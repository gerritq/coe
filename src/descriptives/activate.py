import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.inference import Inference


BASE_DIR = os.getenv("BASE_COE", ".")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
OUT_PATH = os.path.join(OUT_DIR, "f_dims.pdf")


def load_d_m4_domain_items(domain: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, "d_m4_domains", "data.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    target = domain.lower()
    return [x for x in all_items if str(x.get("source", "")).lower() == target]


def sample_balanced(items: list[dict[str, Any]], n_per_label: int, seed: int) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for item in items:
        label = int(item["label"])
        if label in by_label:
            by_label[label].append(item)

    if len(by_label[0]) < n_per_label or len(by_label[1]) < n_per_label:
        raise ValueError(
            f"Not enough samples for balanced subset: "
            f"human={len(by_label[0])}, machine={len(by_label[1])}, requested={n_per_label} each."
        )

    rng = random.Random(seed)
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    sampled = by_label[0][:n_per_label] + by_label[1][:n_per_label]
    rng.shuffle(sampled)
    return sampled


def collect_hidden_states(items: list[dict[str, Any]], model_name: str) -> tuple[np.ndarray, np.ndarray]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode="last_token")

    all_hidden = []
    labels = []
    for item in items:
        out = inference.run(item=item, args=infer_args)
        hs = out["hidden_states"]
        # keep transformer layers only (drop embedding layer)
        sample_layers = [h.detach().to(torch.float32).cpu().numpy() for h in hs[1:]]
        all_hidden.append(np.stack(sample_layers, axis=0))  # (n_layers, d_model)
        labels.append(int(out["label"]))

    x = np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)
    y = np.asarray(labels, dtype=np.int32)  # (n_samples,)
    return x, y


def compute_activation_maps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x: (n_samples, n_layers, d_model)
    # output maps: (d_model, n_layers) for plotting
    human = x[y == 0]  # (n_h, n_layers, d_model)
    machine = x[y == 1]  # (n_m, n_layers, d_model)
    if len(human) == 0 or len(machine) == 0:
        raise ValueError("Expected both human and machine samples.")

    # Per layer and class: subtract sample mean (across samples), then abs, then mean across samples.
    h_centered = human - human.mean(axis=0, keepdims=True)
    m_centered = machine - machine.mean(axis=0, keepdims=True)
    h_mag = np.abs(h_centered).mean(axis=0)  # (n_layers, d_model)
    m_mag = np.abs(m_centered).mean(axis=0)  # (n_layers, d_model)

    # Sort dimensions by overall size; highest at bottom -> ascending order.
    h_order = np.argsort(h_mag.mean(axis=0))
    m_order = np.argsort(m_mag.mean(axis=0))

    h_map = h_mag[:, h_order].T  # (d_model, n_layers)
    m_map = m_mag[:, m_order].T  # (d_model, n_layers)
    return h_map, m_map


def plot_maps(h_map: np.ndarray, m_map: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = "viridis"

    vmin = float(min(np.min(h_map), np.min(m_map)))
    vmax = float(max(np.max(h_map), np.max(m_map)))

    im0 = axes[0].imshow(h_map, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title("Human Activations")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Dimension (sorted, high at bottom)")

    im1 = axes[1].imshow(m_map, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title("Machine Activations")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Dimension (sorted, high at bottom)")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Mean |activation| after sample-centering")

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=240)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_label", type=int, default=250)
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    args.smoke_test = bool(args.smoke_test)

    n_per_label = 25 if args.smoke_test else int(args.n_per_label)
    items = load_d_m4_domain_items(domain="wikipedia")
    sampled = sample_balanced(items=items, n_per_label=n_per_label, seed=int(args.seed))
    print(f"Sampled {n_per_label} human + {n_per_label} machine from d_m4_domains:wikipedia.")

    x, y = collect_hidden_states(sampled, model_name=args.model)
    h_map, m_map = compute_activation_maps(x, y)
    plot_maps(h_map, m_map)


if __name__ == "__main__":
    main()
