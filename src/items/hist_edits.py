import json
import os

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASETS_ALL = ["apt", "beemo_human_edits", "beemo_machine_edits", "editlens"]
DATASETS_MAIN = ["apt", "editlens"]
DATASET_LABELS = {
    "apt": "APT-Eval (Test Set)",
    "editlens": "EditLens (Test Set)",
    "beemo_human_edits": "beemo_human_edits (Test Set)",
    "beemo_machine_edits": "beemo_machine_edits (Test Set)",
}
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


def make_plot(datasets: list[str], output_name: str) -> None:
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(METRICS),
        figsize=(14, 2.8 * len(datasets)),
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = np.array([axes])

    for i, dataset in enumerate(datasets):
        dataset_items = load_dataset_items(dataset)
        for j, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            values = collect_metric_values(dataset_items, metric_key)

            if values:
                ax.hist(values, bins=30, alpha=0.85, color="#4c78a8", edgecolor="white")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

            if j == 0:
                ax.set_ylabel("Frequency")
            else:
                ax.set_ylabel("")

            if i == len(datasets) - 1:
                ax.set_xlabel(metric_label)
            else:
                ax.set_xlabel("")

            if j == 1:
                ax.set_title(DATASET_LABELS.get(dataset, dataset))
            else:
                ax.set_title("")
            ax.grid(alpha=0.25, linewidth=0.5)

    out_path = os.path.join(OUT_DIR, output_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def run() -> None:
    make_plot(DATASETS_ALL, "hist_edits_all.pdf")
    make_plot(DATASETS_MAIN, "hist_edits.pdf")


if __name__ == "__main__":
    run()
