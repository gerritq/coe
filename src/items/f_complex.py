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
    bar_group_width = 0.8
    bar_width = bar_group_width / max(1, n_ds)

    for i, dataset in enumerate(datasets):
        depth_to_vals = points[dataset]
        depths = sorted(depth_to_vals.keys())
        x = np.asarray(depths, dtype=np.int64)
        y = np.asarray([np.mean(depth_to_vals[d]) for d in depths], dtype=np.float64)
        all_y.extend(float(v) for v in y.tolist())

        offsets = (i - (n_ds - 1) / 2.0) * bar_width
        x_bar = x + offsets
        label = label_map.get(dataset, dataset) if label_map is not None else dataset
        plt.bar(x_bar, y, width=bar_width * 0.92, alpha=1.0, label=label)

        # Annotate: AUROC at depth 1, deltas vs depth 1 for depth > 1.
        base = None
        if 1 in depth_to_vals:
            base = float(np.mean(depth_to_vals[1]))
        for xb, d, yy in zip(x_bar.tolist(), depths, y.tolist()):
            if base is None or d == 1:
                txt = f"{yy:.2f}"
            else:
                delta = yy - base
                sign = "+" if delta >= 0 else "-"
                txt = f"{sign}{abs(delta) * 100:.2f}%"
            plt.text(
                xb,
                yy + 0.003,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    plt.xlabel("MLP depth")
    plt.ylabel("AUROC")
    plt.xticks(np.asarray(all_depths, dtype=np.int64))
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
