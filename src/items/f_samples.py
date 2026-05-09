import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
BASELINE_ABLATION_DIR = os.path.join(BASE_DIR, "output", "baseline", "ablation")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

MODES = ["default", "meta_no_pca", "encoder", "biscope", "repreguard"]
MODE_MARKERS = {
    "default": "o",
    "meta_no_pca": "s",
    "encoder": "^",
    "biscope": "D",
    "repreguard": "P",
}
MODE_COLORS = {
    "default": "#1f77b4",
    "meta_no_pca": "#d62728",
    "encoder": "#2ca02c",
    "biscope": "#9467bd",
    "repreguard": "#ff7f0e",
}
DATASET_ORDER = ["drlDomain_arxiv", "multisocial_en", "tsm_first", "m4_gpt4"]
DATASET_TITLES = {
    "drlDomain_arxiv": "DetectRL (ArXiv)",
    "multisocial_en": "MultiSocial (en)",
    "tsm_first": "TSM-Bench (First)",
    "m4_gpt4": "M4 (GPT4)",
}
FONT = {
    "title": 16,
    "axis": 14,
    "ticks": 12,
    "legend": 12,
}
FULL_TRAIN_SIZE = 1500


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
        if n < 10:
            continue
        if n == 750:
            continue

        auroc = _extract_auroc(obj, mode)
        if auroc is None:
            continue

        datasets.add(dataset)
        points[(dataset, mode)].append((n, float(auroc)))

    # baseline ablations: encoder, biscope, repreguard
    if os.path.isdir(BASELINE_ABLATION_DIR):
        for filename in sorted(os.listdir(BASELINE_ABLATION_DIR)):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(BASELINE_ABLATION_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            model = args.get("model")
            dataset = args.get("dataset")
            target = args.get("target_dataset")
            n = args.get("training_size")

            if model not in {"encoder", "biscope", "repreguard"}:
                continue
            if not isinstance(dataset, str) or dataset != target:
                continue
            if n is None:
                continue
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if n < 10:
                continue
            if n == 750:
                continue

            auroc = obj.get("metrics", {}).get("auroc")
            if auroc is None:
                continue
            auroc = float(auroc)
            if model == "repreguard" and auroc > 1.0:
                auroc = auroc / 100.0

            datasets.add(dataset)
            points[(dataset, model)].append((n, auroc))

    for key in points:
        points[key] = sorted(points[key], key=lambda x: x[0])

    return points, sorted(datasets)


def plot_points(points: dict[tuple[str, str], list[tuple[int, float]]], datasets: list[str]) -> None:
    if not points:
        raise RuntimeError("No matching ablation points found for selected modes.")

    datasets = [d for d in DATASET_ORDER if d in datasets and not d.startswith("drlAttack_")]
    if len(datasets) != 4:
        raise RuntimeError(f"Expected 4 datasets after removing drlAttack, got {len(datasets)}: {datasets}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=False)
    axes = axes.reshape(-1)
    legend_handles = {}

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        x_values_for_ticks = set()
        y_values_for_ylim: list[float] = []
        for mode in MODES:
            series = points.get((dataset, mode), [])
            if not series:
                continue
            x = np.array([p[0] for p in series], dtype=np.int64)
            y = np.array([p[1] for p in series], dtype=np.float64)
            x_values_for_ticks.update(int(v) for v in x.tolist())
            y_values_for_ylim.extend(float(v) for v in y.tolist())

            # vertical guide lines for each data point
            y_min_local = float(y.min())
            y_line_floor = max(0.0, y_min_local - 0.01)
            ax.vlines(
                x,
                ymin=y_line_floor,
                ymax=y,
                colors=MODE_COLORS[mode],
                alpha=0.25,
                linewidth=1.0,
                linestyles="--",
            )
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
            ax.set_xlabel("Share of data")
        else:
            ax.set_xlabel("")
        if idx in (0, 2):
            ax.set_ylabel("AUROC")
        else:
            ax.set_ylabel("")
        if x_values_for_ticks:
            xticks = sorted(x_values_for_ticks)
            xlabels = [f"{(n / FULL_TRAIN_SIZE) * 100:.1f}%" for n in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
        if y_values_for_ylim:
            y_min = min(y_values_for_ylim)
            y_max = max(y_values_for_ylim)
            pad = 0.01
            lo = max(0.0, y_min - pad)
            hi = min(1.0, y_max + pad)
            if hi - lo < 0.03:
                mid = 0.5 * (hi + lo)
                lo = max(0.0, mid - 0.015)
                hi = min(1.0, mid + 0.015)
            ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", labelsize=FONT["ticks"])
        ax.title.set_fontsize(FONT["title"])
        ax.xaxis.label.set_size(FONT["axis"])
        ax.yaxis.label.set_size(FONT["axis"])
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
            fontsize=FONT["legend"],
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.11, wspace=0.08, hspace=0.12)

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
