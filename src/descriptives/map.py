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


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_domain_data(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    """"
    Implementation also collects humam subsets - may wanna change this
    
    """
    items = _load_jsonl(path)
    
    # new label
    labels = set([str(item['source']) + "-" + str(item['model']) for item in items])
    print(labels)

    for item in items:
        item['label'] = str(item['source']) + "- " + str(item['model'])

    if not smoke_test:
        return items

    by_label = {}
    for item in items:
        label = item['label']
        by_label.setdefault(label, []).append(item)

    sampled_items = []
    for label, group_items in by_label.items():
        sampled_items.extend(group_items[:50])

    return sampled_items


def load_generator_data(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    items = _load_jsonl(path)

    # rm human subset
    items = [item for item in items if item['model'] != 'human']
    
    if not smoke_test:
        return items
    

    by_label = {}
    for item in items:
        label = item['model']
        by_label.setdefault(label, []).append(item)

    sampled_items = []
    for label, group_items in by_label.items():
        sampled_items.extend(group_items[:50])

    return sampled_items


def collect_middle_layer_representations(
    items: list[dict[str, Any]],
    model_name: str,
    token_mode: str,
    group_key: str,
) -> tuple[np.ndarray, list[str]]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode=token_mode)

    reps = []
    groups = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hidden_states = out["hidden_states"]
        middle_idx = len(hidden_states) // 2
        rep = hidden_states[middle_idx].detach().to(torch.float32).cpu().numpy()
        reps.append(rep)
        groups.append(str(item.get(group_key, "unknown")))

    x = np.stack(reps, axis=0)
    return x, groups


def project_tsne_2d(x: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    return tsne.fit_transform(x)


def plot_domains_2d(x_2d: np.ndarray, domains: list[str], out_path: str, title_suffix: str, legend_title: str) -> None:
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

    plt.title(f"t-SNE of Middle-Layer Representations by {title_suffix}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.legend(title=legend_title)
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

    suffix = "_smoke" if bool(args.smoke_test) else ""

    # A Domains plot
    domains_path = os.path.join(BASE_DIR, "data", "sets", "d_m4_domains", "data.jsonl")
    domain_items = load_domain_data(domains_path, smoke_test=bool(args.smoke_test))
    x, domains = collect_middle_layer_representations(
        items=domain_items,
        model_name=args.model,
        token_mode="last_token",
        group_key="label",
    )
    x_2d = project_tsne_2d(x)
    out_domains = os.path.join(OUT_DIR, f"d_m4_map_domains_mid_layer_{args.model}{suffix}.pdf")
    plot_domains_2d(
        x_2d=x_2d,
        domains=domains,
        out_path=out_domains,
        title_suffix="Domain",
        legend_title="Domain",
    )
    print(f"Saved figure: {out_domains}")

    # B Generators plot
    generators_path = os.path.join(BASE_DIR, "data", "sets", "d_m4_generators", "data.jsonl")
    generator_items = load_generator_data(generators_path, smoke_test=bool(args.smoke_test))
    x, generators = collect_middle_layer_representations(
        items=generator_items,
        model_name=args.model,
        token_mode="last_token",
        group_key="model",
    )
    x_2d = project_tsne_2d(x)
    out_generators = os.path.join(OUT_DIR, f"d_m4_map_generators_mid_layer_{args.model}{suffix}.pdf")
    plot_domains_2d(
        x_2d=x_2d,
        domains=generators,
        out_path=out_generators,
        title_suffix="Generator",
        legend_title="Generator",
    )
    print(f"Saved figure: {out_generators}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
