import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASETS = ["apt", "editlens"]
MODES = ["pca", "meta"]
METRICS = ["sem_similarity", "levenshtein_distance", "jaccard_distance"]

METRIC_LABELS = {
    "sem_similarity": "Semantic Similarity (↓)",
    "levenshtein_distance": "Levenshtein Distance (↑)",
    "jaccard_distance": "Jaccard Distance (↑)",
}


def _load_mode_dataset_payload(dataset: str, mode: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidates = []
    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if args.get("mode") != mode:
            continue
        if int(args.get("components", -1)) != 100:
            continue
        if args.get("dataset") != dataset or args.get("target_dataset") != dataset:
            continue
        candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"No file found for dataset={dataset}, mode={mode}, components=100.")

    obj = candidates[-1]
    payload = obj.get("plot_layer_payload", {})
    proj_scores = payload.get("projection_scores", {})
    sim_metrics = payload.get("similarity_metrics", {})

    if not proj_scores or not sim_metrics:
        raise RuntimeError(
            f"Missing plot_layer_payload content for dataset={dataset}, mode={mode}."
        )

    preferred_score_key = "mean_projection" if mode == "pca" else "meta_scores"
    if preferred_score_key in proj_scores:
        x = np.asarray(proj_scores[preferred_score_key], dtype=np.float64)
    else:
        first_key = next(iter(proj_scores.keys()))
        x = np.asarray(proj_scores[first_key], dtype=np.float64)

    y_by_metric = {
        metric: np.asarray(sim_metrics[metric], dtype=np.float64)
        for metric in METRICS
    }
    return x, y_by_metric


def _plot_one_dataset_mode(dataset: str, mode: str) -> None:
    axis_font = 16
    tick_font = 14
    corr_font = 13
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.2), squeeze=False)
    axes = axes[0]

    x, y_by_metric = _load_mode_dataset_payload(dataset=dataset, mode=mode)
    for c, metric in enumerate(METRICS):
        ax = axes[c]
        y = y_by_metric[metric]

        valid = np.isfinite(x) & np.isfinite(y)
        xv = x[valid]
        yv = y[valid]

        ax.scatter(xv, yv, s=12, alpha=0.7, edgecolors="none")

        if len(xv) >= 2:
            slope, intercept = np.polyfit(xv, yv, 1)
            xs = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.8)
            corr = float(np.corrcoef(xv, yv)[0, 1])
            if metric == "sem_similarity":
                x_text, ha = 0.97, "right"
            else:
                x_text, ha = 0.03, "left"
            ax.text(
                x_text,
                0.95,
                f"r={corr:.3f}",
                transform=ax.transAxes,
                va="top",
                ha=ha,
                fontsize=corr_font,
                bbox=dict(facecolor="lightgray", edgecolor="black", linewidth=0.8, alpha=0.8),
            )
        else:
            if metric == "sem_similarity":
                x_text, ha = 0.97, "right"
            else:
                x_text, ha = 0.03, "left"
            ax.text(
                x_text,
                0.95,
                "r=NA",
                transform=ax.transAxes,
                va="top",
                ha=ha,
                fontsize=corr_font,
                bbox=dict(facecolor="lightgray", edgecolor="black", linewidth=0.8, alpha=0.8),
            )

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=axis_font)
        if c == len(METRICS) // 2:
            ax.set_xlabel("Projection Score", fontsize=axis_font)
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=tick_font)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"f_edit_{dataset}_{mode}.pdf")
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    for dataset in DATASETS:
        for mode in MODES:
            _plot_one_dataset_mode(dataset, mode)


if __name__ == "__main__":
    main()
