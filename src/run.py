import argparse
import os
import json
import random
import math
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from inference import Inference
from coe import Metrics

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")

def load_dataset(args: Namespace):

    random.seed(42)

    with open(os.path.join(DATA_DIR, f"{args.dataset}.jsonl"), "r") as f:
        raw_data = []
        for line in f:
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        data = []
        for item in raw_data:
            human_text = item.get("human_text", item.get("text"))
            machine_text = item.get("machine_text")
            if (human_text is None or human_text.strip() == "") or machine_text is None or machine_text.strip() == "":
                continue

            data.append({'text': human_text,
                         'label': 0})
            data.append({'text': machine_text,
                         'label': 1})

    random.shuffle(data)

    if args.smoke_test:
        data = data[:30]

    print("=" * 50)
    print(f"Loaded {len(data)} samples for dataset: {args.dataset}")
    print("=" * 50)

    
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
        ("length_change_mean", "Length Mean"),
        ("length_change_std", "Length Std"),
    ]
    label_names = {
        0: "human",
        1: "machine",
    }
    label_colors = {
        0: "tab:blue",
        1: "tab:orange",
    }


    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
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

    fig.suptitle(
        f"Angle, Magnitude, and Length Scores by Label | N{len(out)} | Model {args.model} | Data {args.dataset} | LastToken {int(args.last_token)} | DiffVec {int(args.diff_vectors)}"
    )
    fig.tight_layout()

    if save_path is None:              
        save_path = os.path.join(
            OUT_DIR,
            f"coe_dist_{args.model}_{args.dataset}_LT{int(args.last_token)}_DV{int(args.diff_vectors)}{'_ST' if args.smoke_test else ''}.pdf",
        )

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

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for label in labels:
        label_out = [item for item in out if item["label"] == label]
        angle_scores = [item["angle_change_scores"] for item in label_out]
        magnitude_scores = [item["magnitude_change_scores"] for item in label_out]
        length_scores = [item["length_change_scores"] for item in label_out]

        min_layers = min(len(scores) for scores in angle_scores + magnitude_scores + length_scores)
        angle_scores = [scores[:min_layers] for scores in angle_scores]
        magnitude_scores = [scores[:min_layers] for scores in magnitude_scores]
        length_scores = [scores[:min_layers] for scores in length_scores]

        angle_means = []
        angle_stds = []
        magnitude_means = []
        magnitude_stds = []
        length_means = []
        length_stds = []

        for layer_index in range(min_layers):
            angle_vals = [scores[layer_index] for scores in angle_scores]
            magnitude_vals = [scores[layer_index] for scores in magnitude_scores]
            length_vals = [scores[layer_index] for scores in length_scores]

            angle_mean, angle_std = _mean_std(angle_vals)
            magnitude_mean, magnitude_std = _mean_std(magnitude_vals)
            length_mean, length_std = _mean_std(length_vals)

            angle_means.append(angle_mean)
            angle_stds.append(angle_std)
            magnitude_means.append(magnitude_mean)
            magnitude_stds.append(magnitude_std)
            length_means.append(length_mean)
            length_stds.append(length_std)

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

        avg_length = sum(length_means) / len(length_means)
        axes[2].plot(
            layers,
            length_means,
            color=label_color,
            label=f"{label_name} mean",
        )
        axes[2].axhline(
            avg_length,
            linestyle="--",
            color=label_color,
            linewidth=1,
            label=f"{label_name} mean (avg)",
        )
        axes[2].fill_between(
            layers,
            [m - s for m, s in zip(length_means, length_stds)],
            [m + s for m, s in zip(length_means, length_stds)],
            color=label_color,
            alpha=0.2,
            label=f"{label_name} std",
        )

    axes[0].set_title("Angle by Layer")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].set_title("Magnitude by Layer")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    axes[2].set_title("Length by Layer")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    fig.suptitle(
        f"Trajectory | N{len(out)} | Model {args.model} | Data {args.dataset} | LastToken {int(args.last_token)} | DiffVec {int(args.diff_vectors)}"
    )
    fig.tight_layout()

    if save_path is None:       
        save_path = os.path.join(
            OUT_DIR,
            f"trajectory_{args.model}_{args.dataset}_LT{int(args.last_token)}_DV{int(args.diff_vectors)}{'_ST' if args.smoke_test else ''}.pdf",
        )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def pair_plot(args: Namespace, 
              out: list[dict], 
              save_path: str | None = None
              ) -> str:

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "label": item["label"],
                "Angle": item["angle_change_mean"],
                "Magnitude": item["magnitude_change_mean"],
                "Length": item["length_change_mean"],
            }
            for item in out
        ]
    )
    df["label"] = df["label"].map({0: "human", 1: "machine"}).fillna(df["label"])

    grid = sns.pairplot(
        df,
        vars=["Angle", "Magnitude", "Length"],
        hue="label",
        kind="kde",
        diag_kind="kde",
        corner=True,
        plot_kws={"levels": 8, "fill": False},
    )
    grid.fig.suptitle(
        f"Pair Plot (Means) | N{len(out)} | Model {args.model} | Data {args.dataset} | LastToken {int(args.last_token)} | DiffVec {int(args.diff_vectors)}",
        y=1.02,
    )

    if save_path is None:
        save_path = os.path.join(
            OUT_DIR,
            f"pp_{args.model}_{args.dataset}_LT{int(args.last_token)}_DV{int(args.diff_vectors)}{'_ST' if args.smoke_test else ''}.pdf",
        )

    grid.fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    return save_path

def run(args):
    data = load_dataset(args=args)

    inference = Inference(model_name=args.model)
    metrics = Metrics()

    out = []
    for item in tqdm(data, desc="Processing items ..."):
        hs_dict = inference.run(item=item, args=args)
        metrics_dict = metrics.run(
            hidden_states=hs_dict["hidden_states"],
            use_diff_vectors=args.diff_vectors,
        )
        hs_dict.update(metrics_dict)
        del hs_dict["hidden_states"]
        out.append(hs_dict)

    figure_path = plot_scores_by_label(args=args, out=out)
    print(f"Saved figure to {figure_path}")

    layer_path = plot_layer_profiles(args=args, out=out)
    print(f"Saved layer profiles to {layer_path}")

    pair_path = pair_plot(args=args, out=out)
    print(f"Saved pair plot to {pair_path}")

    return out
    

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--last_token", type=int, default=1)
    parser.add_argument("--diff_vectors", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    assert args.smoke_test in (0, 1), "smoke_test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)
    assert args.last_token in (0, 1), "last_token must be 0 or 1"
    args.last_token = bool(args.last_token)
    assert args.diff_vectors in (0, 1), "diff_vectors must be 0 or 1"
    args.diff_vectors = bool(args.diff_vectors)
    assert args.test in (0, 1), "test must be 0 or 1"
    args.test = bool(args.test)

    if args.test:
        OUT_DIR = os.path.join(OUT_DIR, "test")
        os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 50)
    print(f"Running with args:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    run(args=args)
    

if __name__ == "__main__":
    main()
