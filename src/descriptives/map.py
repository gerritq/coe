import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.inference import Inference

BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
SEED = 42


def load_jsonl(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # we only want mgt 
    items = [x for x in items if x["label"] == 1]

    if not smoke_test:
        return items

    print(f"Running smoke test: sampling 100 items per domain from {len(items)} total items.")
    by_domain = {}
    for item in items:
        domain = item["source"]
        by_domain.setdefault(domain, []).append(item)

    print("N domains found:", sorted(by_domain.keys()))

    random.seed(SEED)
    sampled = []
    for domain in sorted(by_domain.keys()):
        domain_items = by_domain[domain]
        random.shuffle(domain_items)
        sampled.extend(domain_items[:50])
    return sampled


def collect_middle_layer_representations(
    items: list[dict[str, Any]],
    model_name: str,
    token_mode: str,
) -> tuple[np.ndarray, list[str]]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode=token_mode)

    reps = []
    domains = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hidden_states = out["hidden_states"]
        middle_idx = len(hidden_states) // 2
        rep = hidden_states[middle_idx].detach().to(torch.float32).cpu().numpy()
        reps.append(rep)
        domains.append(str(item.get("source", "unknown")))

    x = np.stack(reps, axis=0)
    return x, domains


def project_tsne_2d(x: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    return tsne.fit_transform(x)


def plot_domains_2d(x_2d: np.ndarray, domains: list[str], out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    unique_domains = sorted(set(domains))
    cmap = plt.get_cmap("tab10")

    for i, domain in enumerate(unique_domains):
        mask = np.array([d == domain for d in domains], dtype=bool)
        plt.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            s=18,
            alpha=0.7,
            color=cmap(i % 10),
            label=domain,
            edgecolors="none",
        )

    plt.title("t-SNE (2D) of Middle-Layer Representations by Domain")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
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

    x, domains = collect_middle_layer_representations(
        items=items,
        model_name=args.model,
        token_mode="last_token",
    )
    x_2d = project_tsne_2d(x)

    suffix = "_smoke" if bool(args.smoke_test) else ""
    out_path = os.path.join(OUT_DIR, f"d_m4_domains_tsne2d_middle_{args.model}{suffix}.pdf")
    plot_domains_2d(x_2d=x_2d, domains=domains, out_path=out_path)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
