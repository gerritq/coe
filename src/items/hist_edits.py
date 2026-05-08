import json
import os

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASETS = ["apt", "beemo_human_edits", "beemo_machine_edits", "editlens"]
METRICS = [
    ("sem_similarity", "Semantic Similarity"),
    ("jaccard_distance", "Jaccard Distance"),
    ("levenshtein_distance", "Levenshtein Distance"),
]


def load_dataset_items(dataset: str) -> list[dict]:
    path = os.path.join(DATA_DIR, dataset, "test.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return [json.loads(line) for line in lines]


def collect_metric_values(items: list[dict], metric_key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        val = item.get(metric_key)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            values.append(v)
    return values


def run() -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(METRICS),
        figsize=(14, 10),
        constrained_layout=True,
    )

    for i, dataset in enumerate(DATASETS):
        dataset_items = load_dataset_items(dataset)
        for j, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            values = collect_metric_values(dataset_items, metric_key)

            if values:
                ax.hist(values, bins=30, alpha=0.85, color="#4c78a8", edgecolor="white")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

            if i == 0:
                ax.set_title(metric_label)
            if j == 0:
                ax.set_ylabel(dataset)
            ax.set_xlabel("Value")
            ax.grid(alpha=0.25, linewidth=0.5)

    out_path = os.path.join(OUT_DIR, "hist_edits.pdf")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
