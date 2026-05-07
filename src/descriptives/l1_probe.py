import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.inference import Inference


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
SEED = 42
N_RUNS = 5
RUN_SEEDS = [SEED + i for i in range(N_RUNS)]
C_VALUES = [0.01, 0.1, 1, 10]


def load_jsonl(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if not smoke_test:
        return items

    by_label: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        label = int(item["label"])
        by_label.setdefault(label, []).append(item)

    random.seed(SEED)
    sampled = []
    for label in sorted(by_label.keys()):
        label_items = by_label[label]
        random.shuffle(label_items)
        sampled.extend(label_items[:100])
    return sampled


def filter_wikipedia_human_machine(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if str(item.get("source")) != "wikipedia":
            continue
        model = str(item.get("model", ""))
        if model not in {"human", "gpt4"}:
            continue

        # enforce binary labels explicitly
        label = 0 if model == "human" else 1
        out.append({"text": item["text"], "label": label, "source": "wikipedia", "model": model})
    return out


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
        labels.append(int(item["label"]))

    x = np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def nonzero_share_per_layer(x: np.ndarray, y: np.ndarray, seed: int, c_value: float) -> np.ndarray:
    n_samples, n_layers, _ = x.shape
    shares = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        x_layer = x[:, layer, :]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_layer)

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=seed,
            max_iter=2000,
            C=c_value,
        )
        clf.fit(x_scaled, y)

        coef = clf.coef_[0]
        shares[layer] = float(np.mean(np.abs(coef) > 1e-12))

    return shares


def plot_nonzero_share_all(results: dict[float, tuple[np.ndarray, np.ndarray]], out_path: str) -> None:
    plt.figure(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, c_value in enumerate(C_VALUES):
        mean_vals, std_vals = results[c_value]
        x = np.arange(len(mean_vals))
        color = cmap(i % 10)
        plt.errorbar(
            x,
            mean_vals,
            yerr=std_vals,
            fmt="-o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            markersize=3.8,
            label=f"C={c_value}",
        )
    plt.xlabel("Layer")
    plt.ylabel("Share Non-Zero Weights")
    plt.title("L1 Probe Sparsity by Layer (Wikipedia: Human vs Machine)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(title="L1 Strength (C)")
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
    items = filter_wikipedia_human_machine(items)

    if not items:
        raise RuntimeError("No wikipedia human/machine items found in d_m4_domains.")

    x, y = collect_hidden_states(items=items, model_name=args.model)

    suffix = "_smoke" if bool(args.smoke_test) else ""
    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for c_value in C_VALUES:
        all_runs = []
        for seed in RUN_SEEDS:
            shares = nonzero_share_per_layer(x=x, y=y, seed=seed, c_value=c_value)
            all_runs.append(shares)

        all_runs = np.stack(all_runs, axis=0)  # (n_runs, n_layers)
        mean_vals = all_runs.mean(axis=0)
        std_vals = all_runs.std(axis=0, ddof=0)
        results[c_value] = (mean_vals, std_vals)

    out_path = os.path.join(
        OUT_DIR,
        f"d_m4_wikipedia_l1_nonzero_by_layer_{args.model}_allC{suffix}.pdf",
    )
    plot_nonzero_share_all(results=results, out_path=out_path)
    print(f"Saved figure: {out_path}")

    print(f"Runs per C: {N_RUNS} seeds = {RUN_SEEDS}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
