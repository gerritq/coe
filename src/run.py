import argparse
import os
import json
import random
import math
from argparse import Namespace

import matplotlib.pyplot as plt
from tqdm import tqdm

from inference import Inference
from coe import Metrics

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")

def load_dataset(args: Namespace):

    random.seed(42)

    with open(os.path.join(DATA_DIR, f"{args.dataset}.jsonl"), "r") as f:
        raw_data = [json.loads(line) for line in f]

        data = []
        for item in raw_data:
            data.append({'text': item["human_text"],
                         'label': 0})
            data.append({'text': item["machine_text"],
                         'label': 1})

    random.shuffle(data)

    if args.smoke_test:
        data = data[:10]

    return data[:args.n]

def plot_scores_by_label(args: Namespace,
                         out: list[dict], 
                         save_path: str | None = None) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_specs = [
        ("angle_change_mean", "Angle Mean"),
        ("angle_change_std", "Angle Std"),
        ("magnitude_change_mean", "Magnitude Mean"),
        ("magnitude_change_std", "Magnitude Std"),
    ]
    label_names = {
        0: "human",
        1: "machine",
    }
    label_colors = {
        0: "tab:blue",
        1: "tab:orange",
    }


    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    labels = sorted({item["label"] for item in out})

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for label in labels:
            values = [item[metric_key] for item in out if item["label"] == label]
            if not values:
                continue

            label_name = label_names.get(label, str(label))
            label_color = label_colors.get(label, "tab:gray")
            mean_value = sum(values) / len(values)

            axis.hist(
                values,
                bins=20,
                alpha=0.5,
                color=label_color,
                label=label_name,
            )
            axis.axvline(
                mean_value,
                color=label_color,
                linestyle="--",
                linewidth=1,
                label=f"{label_name} mean",
            )

        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Count")
        axis.legend()

    fig.suptitle(f"Angle and Magnitude Scores by Label | N{len(out)} | Model {args.model} | Data {args.dataset}")
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, f"scores_by_label_{args.model}_{args.dataset}{'_ST' if args.smoke_test else ''}.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)

def plot_layer_profiles(args: Namespace,
                        out: list[dict],
                        save_path: str | None = None) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    label_colors = {
        0: "tab:blue",
        1: "tab:orange",
    }

    label_names = {
        0: "human",
        1: "machine",
    }
    labels = sorted({item["label"] for item in out})

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label in labels:
        label_out = [item for item in out if item["label"] == label]
        angle_scores = [item["angle_change_scores"] for item in label_out]
        magnitude_scores = [item["magnitude_change_scores"] for item in label_out]

        min_layers = min(len(scores) for scores in angle_scores + magnitude_scores)
        angle_scores = [scores[:min_layers] for scores in angle_scores]
        magnitude_scores = [scores[:min_layers] for scores in magnitude_scores]

        angle_means = []
        angle_stds = []
        magnitude_means = []
        magnitude_stds = []

        for layer_index in range(min_layers):
            angle_vals = [scores[layer_index] for scores in angle_scores]
            magnitude_vals = [scores[layer_index] for scores in magnitude_scores]

            angle_mean, angle_std = _mean_std(angle_vals)
            magnitude_mean, magnitude_std = _mean_std(magnitude_vals)

            angle_means.append(angle_mean)
            angle_stds.append(angle_std)
            magnitude_means.append(magnitude_mean)
            magnitude_stds.append(magnitude_std)

        layers = list(range(1, min_layers + 1))
        label_name = label_names.get(label, str(label))
        label_color = label_colors.get(label, "tab:gray")

        avg_angle = sum(angle_means) / len(angle_means)
        axes[0].plot(
            layers,
            angle_means,
            color=label_color,
            label=f"{label_name} mean",
        )
        axes[0].axhline(
            avg_angle,
            linestyle="--",
            color=label_color,
            linewidth=1,
            label=f"{label_name} mean (avg)",
        )
        axes[0].fill_between(
            layers,
            [m - s for m, s in zip(angle_means, angle_stds)],
            [m + s for m, s in zip(angle_means, angle_stds)],
            color=label_color,
            alpha=0.2,
            label=f"{label_name} std",
        )

        avg_magnitude = sum(magnitude_means) / len(magnitude_means)
        axes[1].plot(
            layers,
            magnitude_means,
            color=label_color,
            label=f"{label_name} mean",
        )
        axes[1].axhline(
            avg_magnitude,
            linestyle="--",
            color=label_color,
            linewidth=1,
            label=f"{label_name} mean (avg)",
        )
        axes[1].fill_between(
            layers,
            [m - s for m, s in zip(magnitude_means, magnitude_stds)],
            [m + s for m, s in zip(magnitude_means, magnitude_stds)],
            color=label_color,
            alpha=0.2,
            label=f"{label_name} std",
        )

    axes[0].set_title("Angle by Layer")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].set_title("Magnitude by Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    fig.suptitle(
        f"Layer Profiles | N{len(out)} | Model {args.model} | Data {args.dataset}"
    )
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUT_DIR, f"layer_profiles_{args.model}_{args.dataset}{'_ST' if args.smoke_test else ''}.pdf")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

def run(args):
    data = load_dataset(args=args)

    
    inference = Inference(model_name=args.model)
    metrics = Metrics()

    out = []
    for item in tqdm(data, desc="Processing items ..."):
        hs_dict = inference.run(item=item)
        metrics_dict = metrics.run(hidden_states=hs_dict["hidden_states"])
        hs_dict.update(metrics_dict)
        del hs_dict["hidden_states"]
        out.append(hs_dict)

    figure_path = plot_scores_by_label(args=args, out=out)
    print(f"Saved figure to {figure_path}")

    layer_path = plot_layer_profiles(args=args, out=out)
    print(f"Saved layer profiles to {layer_path}")

    return out
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()

    assert args.smoke_test in (0, 1), "smoke_test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)

    run(args=args)
    

if __name__ == "__main__":
    main()
