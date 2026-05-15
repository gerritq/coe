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


def compute_activation_map_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x: (n_samples, n_layers, d_model)
    # output maps: (d_model, n_layers) for plotting
    human = x[y == 0]  # (n_h, n_layers, d_model)
    machine = x[y == 1]  # (n_m, n_layers, d_model)
    if len(human) == 0 or len(machine) == 0:
        raise ValueError("Expected both human and machine samples.")

    h_mag = np.abs(human).mean(axis=0)  # (n_layers, d_model)
    m_mag = np.abs(machine).mean(axis=0)  # (n_layers, d_model)
    diff = (m_mag - h_mag) / (h_mag.std(axis=0) + 1e-12)  # (n_layers, d_model)
    # no sorting on hidden dimension axis
    return diff.T  # (d_model, n_layers)


def plot_map_diff(diff_map: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    cmap = "coolwarm"
    vmax = float(np.max(np.abs(diff_map)))
    vmin = -vmax

    im = ax.imshow(diff_map, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title("Activation Difference (Machine - Human)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dimension")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean |activation| difference")

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
    diff_map = compute_activation_map_diff(x, y)
    plot_map_diff(diff_map)


if __name__ == "__main__":
    main()
