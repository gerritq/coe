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
SEED = 42


def load_d_m4_wikipedia_items() -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, "d_m4_domains", "data.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    # Keep only the wikipedia subset from d_m4_domains.
    items = [x for x in all_items if str(x.get("source", "")).lower() == "wikipedia"]
    if not items:
        raise RuntimeError("No wikipedia items found in d_m4_domains/data.jsonl")
    return items


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
        sample_layers = [h.detach().to(torch.float32).cpu().numpy() for h in hs]
        all_hidden.append(np.stack(sample_layers, axis=0))  # (n_layers, d_model)
        labels.append(int(out["label"]))

    x = np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)
    y = np.asarray(labels, dtype=np.int32)  # (n_samples,)
    return x, y


def von_neumann_entropy(h: torch.Tensor, eps: float = 1e-12) -> float:
    # h shape: (n_samples, hidden_dim)
    k = h @ h.T
    tr = torch.trace(k)
    if float(tr) <= 0.0:
        return float("nan")
    k = k / tr

    eigvals = torch.linalg.eigvalsh(k)
    eigvals = torch.clamp(eigvals, min=eps)
    entropy = -(eigvals * torch.log(eigvals)).sum()
    return float(entropy.item())


def compute_layer_entropies(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x: (n_samples, n_layers, d_model), y: (n_samples,)
    human_mask = y == 0
    machine_mask = y == 1
    if not np.any(human_mask) or not np.any(machine_mask):
        raise ValueError("Expected both human(label=0) and machine(label=1) samples.")

    n_layers = x.shape[1]
    h_ent = np.zeros(n_layers, dtype=np.float64)
    m_ent = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        h_layer = torch.tensor(x[human_mask, layer, :], dtype=torch.float32)
        m_layer = torch.tensor(x[machine_mask, layer, :], dtype=torch.float32)
        h_ent[layer] = von_neumann_entropy(h_layer)
        m_ent[layer] = von_neumann_entropy(m_layer)

    return h_ent, m_ent


def plot_entropies(h_ent: np.ndarray, m_ent: np.ndarray, out_path: str, title: str) -> None:
    layers = np.arange(len(h_ent))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, h_ent, marker="o", linewidth=2.0, label="Human")
    plt.plot(layers, m_ent, marker="o", linewidth=2.0, label="Machine")
    plt.xlabel("Layer")
    plt.ylabel("Von Neumann Entropy")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--n_per_label", type=int, default=500)
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    items = load_d_m4_wikipedia_items()

    n_per_label = 25 if bool(args.smoke_test) else int(args.n_per_label)
    sampled = sample_balanced(items=items, n_per_label=n_per_label, seed=SEED)
    print(f"Sampled {n_per_label} human + {n_per_label} machine from d_m4_domains:wikipedia.")

    x, y = collect_hidden_states(items=sampled, model_name=args.model)
    h_ent, m_ent = compute_layer_entropies(x=x, y=y)

    out_path = os.path.join(OUT_DIR, "qual.pdf")
    plot_entropies(
        h_ent=h_ent,
        m_ent=m_ent,
        out_path=out_path,
        title="Von Neumann Entropy by Layer | d_m4_domains:wikipedia",
    )
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
