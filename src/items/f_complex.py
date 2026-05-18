import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
MLP_DIR = os.path.join(BASE_DIR, "output", "mlp")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
OUT_PATH = os.path.join(OUT_DIR, "f_complex_id.pdf")
OUT_PATH_OOD = os.path.join(OUT_DIR, "f_complex_ood.pdf")
TRAIN_DOMAIN = "wikipedia"
ID_DOMAINS = ["wikipedia", "arxiv", "peerread", "reddit"]
OOD_TARGETS = ["arxiv", "peerread", "reddit"]
OOD_LABELS = {
    "wikipedia": "Wikipedia",
    "arxiv": "Arxiv",
    "peerread": "PeerRead",
    "reddit": "Reddit",
}
MAX_DEPTH = 5
FONT = {
    "title": 20,
    "axis": 18,
    "ticks": 15,
    "legend": 14,
}


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
    if not os.path.isdir(MLP_DIR):
        return rows

    for filename in sorted(os.listdir(MLP_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(MLP_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        args = obj.get("args", {})
        if args.get("mode") != "mlp":
            continue

        if "complexity" not in args:
            continue
        try:
            depth = int(args["complexity"])
        except (TypeError, ValueError):
            continue
        if depth < 1 or depth > MAX_DEPTH:
            continue

        dataset = args.get("domain")
        target_dataset = args.get("target_dataset")
        if not isinstance(dataset, str):
            continue
        if not isinstance(target_dataset, str):
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


def collect_points_id_domains() -> dict[str, dict[int, list[float]]]:
    # dataset -> depth -> list[auroc]
    points: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _collect_mlp_points():
        dataset = row["dataset"]
        target_dataset = row["target_dataset"]
        if dataset != TRAIN_DOMAIN:
            continue
        if target_dataset not in ID_DOMAINS:
            continue
        points[target_dataset][row["mlp_depth"]].append(row["auroc"])
    return points


def collect_points_ood_drl() -> dict[str, dict[int, list[float]]]:
    # target_dataset -> depth -> list[auroc]
    points: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _collect_mlp_points():
        dataset = row["dataset"]
        target_dataset = row["target_dataset"]
        if dataset != TRAIN_DOMAIN:
            continue
        if target_dataset not in OOD_TARGETS:
            continue
        points[target_dataset][row["mlp_depth"]].append(row["auroc"])
    return points


def plot(points: dict[str, dict[int, list[float]]], out_path: str, label_map: dict[str, str] | None = None) -> None:
    if not points:
        raise RuntimeError("No matching mlp points found in output/mlp.")

    plt.figure(figsize=(11, 7))

    datasets = sorted(points.keys())
    all_depths = sorted({d for ds in datasets for d in points[ds].keys()})
    all_y: list[float] = []

    n_ds = len(datasets)
    n_depths = len(all_depths)
    bar_group_width = 0.8
    bar_width = bar_group_width / max(1, n_depths)
    x_base = np.arange(n_ds, dtype=np.float64)
    # Blue-only palette: darker shade for larger depth.
    depth_colors = plt.cm.Blues(np.linspace(0.35, 0.9, max(1, n_depths)))

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
    # Draw lines per dataset group (not across groups).
    for ds_idx, dataset in enumerate(datasets):
        x_group = []
        y_group = []
        for depth_idx, depth in enumerate(all_depths):
            vals = points[dataset].get(depth, [])
            if not vals:
                continue
            y_val = float(np.mean(vals))
            x_val = x_base[ds_idx] + (depth_idx - (n_depths - 1) / 2.0) * bar_width
            x_group.append(x_val)
            y_group.append(y_val)
        if len(x_group) >= 2:
            plt.plot(
                x_group,
                y_group,
                color="#1f4e79",
                linewidth=1.2,
                alpha=0.8,
                marker="o",
                markersize=3.5,
                markerfacecolor="#1f4e79",
                markeredgecolor="#1f4e79",
                zorder=3,
            )

    default_label_map = OOD_LABELS
    x_labels = [
        label_map.get(d, default_label_map.get(d, d)) if label_map is not None else default_label_map.get(d, d)
        for d in datasets
    ]
    plt.xlabel("")
    plt.ylabel("AUC")
    plt.xticks(x_base, x_labels, rotation=0, ha="center")
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
    plt.tick_params(axis="both", labelsize=FONT["ticks"])
    plt.gca().xaxis.label.set_size(FONT["axis"])
    plt.gca().yaxis.label.set_size(FONT["axis"])
    plt.gca().title.set_fontsize(FONT["title"])
    plt.legend(frameon=True, fontsize=FONT["legend"])
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()


def main() -> None:
    points = collect_points_id_domains()
    plot(points, OUT_PATH, label_map=OOD_LABELS)
    print(f"Saved: {OUT_PATH}")

    points_ood = collect_points_ood_drl()
    plot(points_ood, OUT_PATH_OOD, label_map=OOD_LABELS)
    print(f"Saved: {OUT_PATH_OOD}")


if __name__ == "__main__":
    main()
