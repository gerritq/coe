import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.inference import Inference
from src.utils import load_dataset


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "detectrl": {
        "datasets": ["drlDomain_arxiv", "drlDomain_writing_prompt", "drlDomain_yelp_review", "drlDomain_xsum"],
        "reference": "drlDomain_arxiv",
        "labels": {
            "drlDomain_arxiv": "ArXiv",
            "drlDomain_writing_prompt": "Reddit",
            "drlDomain_yelp_review": "Yelp",
            "drlDomain_xsum": "News",
        },
    },
    "multisocial": {
        "datasets": ["multisocial_en", "multisocial_de", "multisocial_ru", "multisocial_zh"],
        "reference": "multisocial_en",
        "labels": {
            "multisocial_en": "en",
            "multisocial_de": "de",
            "multisocial_ru": "ru",
            "multisocial_zh": "zh",
        },
    },
    "tsm": {
        "datasets": ["tsm_first", "tsm_extend", "tsm_sums", "tsm_tst"],
        "reference": "tsm_first",
        "labels": {
            "tsm_first": "first",
            "tsm_extend": "extend",
            "tsm_sums": "sums",
            "tsm_tst": "tst",
        },
    },
    "raid": {
        "datasets": ["raidModel_cohere_chat", "raidModel_gpt4", "raidModel_llama_chat", "raidModel_mistral_chat"],
        "reference": "raidModel_gpt4",
        "labels": {
            "raidModel_cohere_chat": "cohere",
            "raidModel_gpt4": "gpt4",
            "raidModel_llama_chat": "llama",
            "raidModel_mistral_chat": "mistral",
        },
    },
}

M4_DOMAINS = ["wikipedia", "arxiv", "reddit", "peerread"]
M4_REFERENCE = "wikipedia"


# Fixed label colors across plots.
COLOR_MAP = {
    0: ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"],  # human
    1: ["#ff7f0e", "#d62728", "#e377c2", "#7f7f7f"],  # machine
}


def collect_mid_layer_representations(
    items: list[dict],
    inference: Inference,
) -> tuple[np.ndarray, np.ndarray]:
    infer_args = Namespace(mode="default", token_mode="last_token")
    x_all = []
    y_all = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hs = out["hidden_states"]
        mid_idx = len(hs) // 2
        vec = hs[mid_idx].detach().to(torch.float32).cpu().numpy()
        x_all.append(vec)
        y_all.append(int(out["label"]))

    x = np.stack(x_all, axis=0)  # (n_samples, d_model)
    y = np.asarray(y_all, dtype=np.int32)  # (n_samples,)
    return x, y


def load_benchmark_dataset_states(
    args: Namespace,
    benchmark_datasets: list[str],
    inference: Inference,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}

    for dataset_name in benchmark_datasets:
        ds = load_dataset(
            Namespace(
                dataset=dataset_name,
                smoke_test=args.smoke_test,
                training_size=None,
                seed=args.seed,
            )
        )

        test_items = ds.get("test", [])
        if len(test_items) == 0:
            test_items = ds.get("train", [])

        items = [dict(x) for x in test_items]
        x, y = collect_mid_layer_representations(items, inference=inference)
        out[dataset_name] = {"x": x, "y": y}
        print(f"Loaded {dataset_name}: n={len(y)}")

    return out


def load_m4_domain_states(
    args: Namespace,
    inference: Inference,
) -> dict[str, dict[str, np.ndarray]]:
    path = os.path.join(DATA_DIR, "d_m4_domains", "data.jsonl")
    items_by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in M4_DOMAINS}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            domain = str(item.get("source", "")).lower()
            if domain in items_by_domain:
                items_by_domain[domain].append(item)

    out: dict[str, dict[str, np.ndarray]] = {}
    for domain in M4_DOMAINS:
        items = items_by_domain[domain]
        if args.smoke_test:
            items = items[: min(200, len(items))]
        x, y = collect_mid_layer_representations(items, inference=inference)
        out[domain] = {"x": x, "y": y}
        print(f"Loaded d_m4_domains:{domain}: n={len(y)}")

    return out


def build_combined_pca_projection(
    dataset_states: dict[str, dict[str, np.ndarray]],
    dataset_order: list[str],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    x_concat = np.concatenate([dataset_states[name]["x"] for name in dataset_order], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(x_concat)

    projected: dict[str, dict[str, np.ndarray]] = {}
    for dataset_name in dataset_order:
        projected[dataset_name] = {
            "x2d": pca.transform(dataset_states[dataset_name]["x"]),
            "y": dataset_states[dataset_name]["y"],
        }
    return projected


def _plot_pair_subplot(
    ax: plt.Axes,
    projected: dict[str, dict[str, np.ndarray]],
    ref_name: str,
    other_name: str,
    labels_map: dict[str, str],
    pair_index: int,
) -> None:
    datasets = [ref_name, other_name]

    for i, dataset_name in enumerate(datasets):
        x2d = projected[dataset_name]["x2d"]
        y = projected[dataset_name]["y"]

        for lab in [0, 1]:
            mask = y == lab
            if not np.any(mask):
                continue
            ax.scatter(
                x2d[mask, 0],
                x2d[mask, 1],
                s=12,
                alpha=0.7,
                c=COLOR_MAP[lab][pair_index],
                label=f"{labels_map[dataset_name]} {'Human' if lab == 0 else 'Machine'}",
                edgecolors="none",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=8)


def plot_benchmark_parallel(
    benchmark_name: str,
    projected: dict[str, dict[str, np.ndarray]],
    dataset_order: list[str],
    ref_name: str,
    labels_map: dict[str, str],
) -> None:
    others = [d for d in dataset_order if d != ref_name]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), squeeze=False)

    for idx, other in enumerate(others):
        ax = axes[0, idx]
        _plot_pair_subplot(
            ax=ax,
            projected=projected,
            ref_name=ref_name,
            other_name=other,
            labels_map=labels_map,
            pair_index=idx,
        )

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"f_parallel_{benchmark_name}.pdf")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: Namespace) -> None:
    inference = Inference(model_name=args.model)

    for benchmark_name, spec in BENCHMARK_SPECS.items():
        dataset_order = spec["datasets"]
        ref_name = spec["reference"]
        labels_map = spec["labels"]

        dataset_states = load_benchmark_dataset_states(
            args=args,
            benchmark_datasets=dataset_order,
            inference=inference,
        )
        projected = build_combined_pca_projection(
            dataset_states=dataset_states,
            dataset_order=dataset_order,
            seed=args.seed,
        )
        plot_benchmark_parallel(
            benchmark_name=benchmark_name,
            projected=projected,
            dataset_order=dataset_order,
            ref_name=ref_name,
            labels_map=labels_map,
        )

    m4_states = load_m4_domain_states(args=args, inference=inference)
    m4_projected = build_combined_pca_projection(
        dataset_states=m4_states,
        dataset_order=M4_DOMAINS,
        seed=args.seed,
    )
    m4_labels = {d: d for d in M4_DOMAINS}
    plot_benchmark_parallel(
        benchmark_name="m4",
        projected=m4_projected,
        dataset_order=M4_DOMAINS,
        ref_name=M4_REFERENCE,
        labels_map=m4_labels,
    )


def main() -> None:
    args = parse_args()
    args.smoke_test = bool(args.smoke_test)
    run(args=args)


if __name__ == "__main__":
    main()
