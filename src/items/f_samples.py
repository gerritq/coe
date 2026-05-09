import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

MODES = ["default", "meta_no_pca"]
MODE_MARKERS = {
    "default": "o",
    "meta_no_pca": "s",
}
MODE_COLORS = {
    "default": "#1f77b4",
    "meta_no_pca": "#d62728",
}
DATASET_ORDER = ["drlDomain_arxiv", "multisocial_en", "tsm_first", "m4_gpt4"]
DATASET_TITLES = {
    "drlDomain_arxiv": "DetectRL (ArXiv)",
    "multisocial_en": "MultiSocial (en)",
    "tsm_first": "TSM-Bench (First)",
    "m4_gpt4": "M4 (GPT4)",
}


def _extract_auroc(obj: dict, mode: str) -> float | None:
    tm = obj.get("test_metrics", {})
    if mode == "default":
        return tm.get("mean_projection_metrics", {}).get("auroc")
    if mode == "meta_no_pca":
        return tm.get("meta_metrics", {}).get("auroc")
    return None


def collect_points() -> tuple[dict[tuple[str, str], list[tuple[int, float]]], list[str]]:
    # (dataset, mode) -> [(N, auroc), ...]
    points: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    datasets = set()

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        mode = args.get("mode")
        dataset = args.get("dataset")
        target = args.get("target_dataset")
        n = args.get("training_size")

        if mode not in MODES:
            continue
        if not isinstance(dataset, str) or dataset != target:
            continue
        if n is None:
            continue
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue

        auroc = _extract_auroc(obj, mode)
        if auroc is None:
            continue

        datasets.add(dataset)
        points[(dataset, mode)].append((n, float(auroc)))

    for key in points:
        points[key] = sorted(points[key], key=lambda x: x[0])

    return points, sorted(datasets)


def plot_points(points: dict[tuple[str, str], list[tuple[int, float]]], datasets: list[str]) -> None:
    if not points:
        raise RuntimeError("No matching ablation points found for modes default/meta_no_pca.")

    datasets = [d for d in DATASET_ORDER if d in datasets and not d.startswith("drlAttack_")]
    if len(datasets) != 4:
        raise RuntimeError(f"Expected 4 datasets after removing drlAttack, got {len(datasets)}: {datasets}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=False)
    axes = axes.reshape(-1)
    legend_handles = {}

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        for mode in MODES:
            series = points.get((dataset, mode), [])
            if not series:
                continue
            x = np.array([p[0] for p in series], dtype=np.int64)
            y = np.array([p[1] for p in series], dtype=np.float64)
            line, = ax.plot(
                x,
                y,
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                linewidth=1.8,
                markersize=6,
                alpha=0.95,
                label=mode,
            )
            legend_handles[mode] = line

        ax.set_title(DATASET_TITLES.get(dataset, dataset))
        if idx in (2, 3):
            ax.set_xlabel("Number of samples")
        else:
            ax.set_xlabel("")
        if idx in (0, 2):
            ax.set_ylabel("AUROC")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.25)

    handles = [legend_handles[m] for m in MODES if m in legend_handles]
    labels = [m for m in MODES if m in legend_handles]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.subplots_adjust(bottom=0.10)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "f_samples.pdf")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    points, datasets = collect_points()
    plot_points(points, datasets)


if __name__ == "__main__":
    main()
