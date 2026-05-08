import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.inference import Inference


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
SEED = 42


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_arxiv_human_machine(path: str, smoke_test: bool = False) -> list[dict[str, Any]]:
    items = load_jsonl(path)

    filtered = []
    for item in items:
        if str(item.get("source")) != "arxiv":
            continue
        model = str(item.get("model", ""))
        if model not in {"human", "gpt4"}:
            continue

        label = 0 if model == "human" else 1
        filtered.append({
            "text": item["text"],
            "label": label,
            "source": "arxiv",
            "model": model,
        })

    if not smoke_test:
        return filtered

    random.seed(SEED)
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for item in filtered:
        by_label[int(item["label"])].append(item)

    sampled = []
    for label in (0, 1):
        group = by_label[label]
        random.shuffle(group)
        sampled.extend(group[:120])
    return sampled


def collect_middle_layer_representations(
    items: list[dict[str, Any]], model_name: str, token_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode=token_mode)

    reps = []
    labels = []
    for item in items:
        out = inference.run(item=item, args=infer_args)
        hidden_states = out["hidden_states"]
        middle_idx = len(hidden_states) // 2
        rep = hidden_states[middle_idx].detach().to(torch.float32).cpu().numpy()
        reps.append(rep)
        labels.append(int(item["label"]))

    x = np.stack(reps, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def fit_pca_and_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, PCA, StandardScaler, LogisticRegression]:
    pca = PCA(n_components=2, random_state=SEED)
    x_2d = pca.fit_transform(x)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_2d)

    clf = LogisticRegression(random_state=SEED, max_iter=2000)
    clf.fit(x_scaled, y)
    return x_2d, pca, scaler, clf


def plot_pca_with_boundary(
    x_2d: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    out_path: str,
    model_name: str,
) -> None:
    plt.figure(figsize=(7.5, 6.0))
    cmap = plt.get_cmap("tab10")
    ax = plt.gca()

    human_color = cmap(0)
    machine_color = cmap(1)
    class_specs = [(0, "Human", human_color), (1, "Machine (gpt4)", machine_color)]
    legend_handles = []
    for label, name, color in class_specs:
        mask = y == label
        pts = plt.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            s=18,
            alpha=0.7,
            color=color,
            label=name,
            edgecolors="none",
        )
        legend_handles.append(pts)

    x_min, x_max = x_2d[:, 0].min() - 0.8, x_2d[:, 0].max() + 0.8
    y_min, y_max = x_2d[:, 1].min() - 0.8, x_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    zz = clf.decision_function(grid_scaled).reshape(xx.shape)

    if abs(clf.coef_[0, 1]) > 1e-12:
        boundary_y = -(clf.coef_[0, 0] * xx[0] + clf.intercept_[0]) / clf.coef_[0, 1]
        boundary_points = np.column_stack([xx[0], boundary_y])
    else:
        boundary_x = np.full_like(yy[:, 0], -clf.intercept_[0] / clf.coef_[0, 0])
        boundary_points = np.column_stack([boundary_x, yy[:, 0]])

    segment_pairs = np.stack([boundary_points[:-1], boundary_points[1:]], axis=1)
    segment_colors = np.linspace(0.0, 1.0, len(segment_pairs))
    boundary_cmap = mcolors.LinearSegmentedColormap.from_list(
        "boundary_colors",
        [human_color, machine_color],
    )
    boundary_segments = LineCollection(
        segment_pairs,
        colors=boundary_cmap(segment_colors),
        linewidths=2.4,
        linestyles="-",
        zorder=3,
    )
    ax.add_collection(boundary_segments)
    ax.plot(boundary_points[:, 0], boundary_points[:, 1], linestyle="--", color="black", linewidth=1.3, zorder=4)

    w = clf.coef_[0] / scaler.scale_
    midpoint = x_2d.mean(axis=0)
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        direction = w / w_norm
        arrow_length = 0.35 * max(float(np.ptp(x_2d[:, 0])), float(np.ptp(x_2d[:, 1])), 1e-6)
        ax.arrow(
            midpoint[0],
            midpoint[1],
            direction[0] * arrow_length,
            direction[1] * arrow_length,
            width=0.0,
            head_width=0.10 * arrow_length,
            head_length=0.14 * arrow_length,
            length_includes_head=True,
            color="black",
            linewidth=2.0,
            zorder=5,
        )
        ax.arrow(
            midpoint[0],
            midpoint[1],
            -direction[0] * arrow_length,
            -direction[1] * arrow_length,
            width=0.0,
            head_width=0.10 * arrow_length,
            head_length=0.14 * arrow_length,
            length_includes_head=True,
            color="black",
            linewidth=2.0,
            alpha=0.35,
            zorder=5,
        )

    legend_handles.append(Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Logistic decision boundary"))
    legend_handles.append(Line2D([0], [0], color="black", lw=2.0, label="Weight vector / normal"))

    plt.title(f"arXiv Middle-Layer PCA (2D) + Logistic Boundary ({model_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(BASE_DIR, "data", "sets", "d_m4_domains", "data.jsonl")
    items = load_arxiv_human_machine(data_path, smoke_test=bool(args.smoke_test))
    if not items:
        raise RuntimeError("No arxiv human/machine items found in d_m4_domains.")

    x, y = collect_middle_layer_representations(
        items=items,
        model_name=args.model,
        token_mode=args.token_mode,
    )
    x_2d, _, scaler, clf = fit_pca_and_logistic(x=x, y=y)

    suffix = "_smoke" if bool(args.smoke_test) else ""
    out_path = os.path.join(OUT_DIR, f"d_m4_arxiv_mid_layer_pca_lr_{args.model}{suffix}.pdf")
    plot_pca_with_boundary(
        x_2d=x_2d,
        y=y,
        scaler=scaler,
        clf=clf,
        out_path=out_path,
        model_name=args.model,
    )
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
