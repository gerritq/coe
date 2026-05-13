import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
OUT_PATH = os.path.join(OUT_DIR, "f_complex.pdf")
OUT_PATH_OOD = os.path.join(OUT_DIR, "f_complex_ood.pdf")
OOD_SOURCE = "drlDomain_arxiv"
OOD_TARGETS = ["drlDomain_writing_prompt", "drlDomain_xsum", "drlDomain_yelp_review"]
OOD_LABELS = {
    "drlDomain_writing_prompt": "writing_prompts",
    "drlDomain_xsum": "xsum",
    "drlDomain_yelp_review": "yelp_review",
}
MAX_DEPTH = 5


def _extract_auroc(obj: dict) -> float | None:
    tm = obj.get("test_metrics", {})
    mean_metrics = tm.get("mean_projection_metrics", {})
    if "auroc" in mean_metrics:
        return float(mean_metrics["auroc"])

    meta_metrics = tm.get("meta_metrics", {})
    if "auroc" in meta_metrics:
        return float(meta_metrics["auroc"])

    weighted_metrics = tm.get("weighted_projection_metrics", {})
    if "auroc" in weighted_metrics:
        return float(weighted_metrics["auroc"])
    return None


def _collect_mlp_points() -> list[dict]:
    rows: list[dict] = []
    if not os.path.isdir(ABLATION_DIR):
        return rows

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        args = obj.get("args", {})
        if args.get("mode") != "mlp":
            continue

        if "mlp_depth" not in args:
            continue
        try:
            depth = int(args["mlp_depth"])
        except (TypeError, ValueError):
            continue
        if depth < 1 or depth > MAX_DEPTH:
            continue

        dataset = args.get("dataset")
        target_dataset = args.get("target_dataset")
        if not isinstance(dataset, str):
            continue

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "target_dataset": target_dataset,
                "mlp_depth": depth,
                "auroc": float(auroc),
            }
        )

    return rows


def collect_points_id_tsm() -> dict[str, dict[int, list[float]]]:
    # dataset -> depth -> list[auroc]
    points: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _collect_mlp_points():
        dataset = row["dataset"]
        target_dataset = row["target_dataset"]
        if not dataset.startswith("tsm_"):
            continue
        if target_dataset != dataset:
            continue
        points[dataset][row["mlp_depth"]].append(row["auroc"])
    return points


def collect_points_ood_drl() -> dict[str, dict[int, list[float]]]:
    # target_dataset -> depth -> list[auroc]
    points: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _collect_mlp_points():
        dataset = row["dataset"]
        target_dataset = row["target_dataset"]
        if dataset != OOD_SOURCE:
            continue
        if target_dataset not in OOD_TARGETS:
            continue
        points[target_dataset][row["mlp_depth"]].append(row["auroc"])
    return points


def plot(points: dict[str, dict[int, list[float]]], out_path: str, label_map: dict[str, str] | None = None) -> None:
    if not points:
        raise RuntimeError("No matching mlp ablation points found in output/probe/ablation.")

    plt.figure(figsize=(10, 6))

    datasets = sorted(points.keys())
    all_depths = sorted({d for ds in datasets for d in points[ds].keys()})
    all_y: list[float] = []

    n_ds = len(datasets)
    n_depths = len(all_depths)
    bar_group_width = 0.8
    bar_width = bar_group_width / max(1, n_depths)
    x_base = np.arange(n_ds, dtype=np.float64)
    depth_colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_depths)))

    for depth_idx, depth in enumerate(all_depths):
        y_vals = []
        for dataset in datasets:
            vals = points[dataset].get(depth, [])
            y_vals.append(float(np.mean(vals)) if vals else np.nan)
        y = np.asarray(y_vals, dtype=np.float64)
        finite_mask = np.isfinite(y)
        all_y.extend(float(v) for v in y[finite_mask].tolist())

        offsets = (depth_idx - (n_depths - 1) / 2.0) * bar_width
        x_bar = x_base + offsets
        plt.bar(
            x_bar[finite_mask],
            y[finite_mask],
            width=bar_width * 0.92,
            alpha=0.9,
            color=depth_colors[depth_idx],
            label=f"Depth {depth}",
            zorder=2,
        )
        plt.plot(
            x_base[finite_mask],
            y[finite_mask],
            color=depth_colors[depth_idx],
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )

    x_labels = [label_map.get(d, d) if label_map is not None else d for d in datasets]
    plt.xlabel("Dataset")
    plt.ylabel("AUROC")
    plt.xticks(x_base, x_labels, rotation=20, ha="right")
    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        pad = 0.01
        lo = max(0.0, y_min - pad)
        hi = min(1.0, y_max + pad)
        if hi - lo < 0.03:
            mid = 0.5 * (hi + lo)
            lo = max(0.0, mid - 0.015)
            hi = min(1.0, mid + 0.015)
        plt.ylim(lo, hi)
    else:
        plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def main() -> None:
    points = collect_points_id_tsm()
    plot(points, OUT_PATH)
    print(f"Saved: {OUT_PATH}")

    points_ood = collect_points_ood_drl()
    plot(points_ood, OUT_PATH_OOD, label_map=OOD_LABELS)
    print(f"Saved: {OUT_PATH_OOD}")


if __name__ == "__main__":
    main()
