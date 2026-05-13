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

def load_d_m4_domain_items(domain: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, "d_m4_domains", "data.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    target = domain.lower()
    items = [x for x in all_items if str(x.get("source", "")).lower() == target]
    return items


def load_main_data_items(dataset_name: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, dataset_name, "test.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
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


def von_neumann_entropy_1(h: torch.Tensor, eps: float = 1e-12) -> float:
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

def von_neumann_entropy_2(h: torch.Tensor):

    K = h @ h.T
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]

    mek = mek/mek.sum()
    H = -1*torch.sum(mek*torch.log(mek))
    return H

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
        h_ent[layer] = von_neumann_entropy_2(h_layer)
        m_ent[layer] = von_neumann_entropy_2(m_layer)

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
    n_per_label = 25 if bool(args.smoke_test) else int(args.n_per_label)

    # d_m4_domains: run one balanced plot per source domain.
    d_m4_domains = ["wikipedia", "arxiv", "reddit", "peerread"]
    for domain in d_m4_domains:
        items = load_d_m4_domain_items(domain=domain)
        sampled = sample_balanced(items=items, n_per_label=n_per_label, seed=SEED)
        print(f"Sampled {n_per_label} human + {n_per_label} machine from d_m4_domains:{domain}.")

        x, y = collect_hidden_states(items=sampled, model_name=args.model)
        h_ent, m_ent = compute_layer_entropies(x=x, y=y)

        dataset_name = f"entropy_m4_{domain}"
        out_path = os.path.join(OUT_DIR, f"qual_{dataset_name}.pdf")
        plot_entropies(
            h_ent=h_ent,
            m_ent=m_ent,
            out_path=out_path,
            title=f"Von Neumann Entropy by Layer | {dataset_name}",
        )
        print(f"Saved figure: {out_path}")

    # drlDomain_* datasets: use test split only, no rebalancing.
    drl_datasets = ["drlDomain_arxiv", "drlDomain_xsum"]
    for dataset_name in drl_datasets:
        items = load_main_data_items(dataset_name=dataset_name)
        print(f"Loaded {len(items)} test items from {dataset_name} (no rebalancing).")

        x, y = collect_hidden_states(items=items, model_name=args.model)
        h_ent, m_ent = compute_layer_entropies(x=x, y=y)

        out_path = os.path.join(OUT_DIR, f"entropy_{dataset_name}.pdf")
        plot_entropies(
            h_ent=h_ent,
            m_ent=m_ent,
            out_path=out_path,
            title=f"Von Neumann Entropy by Layer | {dataset_name}",
        )
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
