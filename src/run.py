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
from coo import Metrics as EntropyMetrics
from classifier import ScoreGMM, ScoreLogistic, ScoreMLP

from utils import load_dataset, compute_auc_for_scores

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data")
VIZ_DIR = os.path.join(BASE_DIR, "out")
SCORE_DIR = os.path.join(BASE_DIR, "scores")
CLASSIFIER_DIR = os.path.join(BASE_DIR, "classifier")

TEXT_PREFIX = "Is this text human- or LLM-written?"

def plot_scores_by_label(args: Namespace,
                         out: list[dict], 
                         save_path: str | None = None) -> str:
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    metric_specs = [
        ("angle_change_mean", "Angle Mean"),
        ("angle_change_std", "Angle Std"),
        ("magnitude_change_mean", "Magnitude Mean"),
        ("magnitude_change_std", "Magnitude Std"),
        ("length_change_mean", "Length Mean"),
        ("length_change_std", "Length Std"),
    ]
    if args.dataset in ["counterfact"]:
        label_names = {
        0: "correct",
        1: "incorrect",
        }        
    else:
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
        f"Angle, Magnitude, and Length Scores by Label {args.title_info}"
    )
    fig.tight_layout()

    if save_path is None:              
        save_path = os.path.join(
            VIZ_DIR,
            f"coe_dist_{args.suffix}",
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
    
    os.makedirs(VIZ_DIR, exist_ok=True)

    if args.dataset in ["counterfact"]:
        label_names = {
        0: "correct",
        1: "incorrect",
        }        
    else:
        label_names = {
        0: "human",
        1: "machine",
        }
    label_colors = {
        0: "tab:blue",
        1: "tab:orange",
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
        f"Trajectory | {args.title_info}"
    )
    fig.tight_layout()

    if save_path is None:       
        save_path = os.path.join(
            OUT_DIR,
            f"trajectory_{args.suffix}"
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
    if args.dataset in ["counterfact"]:
        df["label"] = df["label"].map({0: "correct", 1: "incorrect"}).fillna(df["label"])
    else:
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
        f"Pair Plot (Means) | {args.title_info}",
        y=1.02,
    )

    if save_path is None:
        save_path = os.path.join(
            OUT_DIR,
            f"pp_{args.suffix}",
        )

    grid.fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    return save_path

def plot_entropy_by_label(
    args: Namespace,
    out: list[dict],
    save_path: str | None = None,
) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_specs = [
        ("vocab_entropy_mean", "Vocab Entropy Mean"),
        ("vocab_entropy_std", "Vocab Entropy Std"),
        ("topk_entropy_mean", "Top-K Entropy Mean"),
        ("topk_entropy_std", "Top-K Entropy Std"),
        ("vocab_entropy_change_mean", "Vocab Entropy Change Mean"),
        ("vocab_entropy_change_std", "Vocab Entropy Change Std"),
        ("topk_entropy_change_mean", "Top-K Entropy Change Mean"),
        ("topk_entropy_change_std", "Top-K Entropy Change Std"),
    ]

    label_names = {
        0: "human",
        1: "machine",
    }
    label_colors = {
        0: "tab:blue",
        1: "tab:orange",
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
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
        axis.set_xlabel("Entropy")
        axis.set_ylabel("Count")
        axis.legend()

    fig.suptitle(
        f"Entropy (Mean/Std/Change) by Label {args.title_info}"
    )
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            VIZ_DIR,
            f"entropy_dist_{args.suffix}",
        )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

def plot_tvd_by_label(
    args: Namespace,
    out: list[dict],
    save_path: str | None = None,
) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_specs = [
        ("vocab_tvd_mean", "Vocab TVD Mean"),
        ("vocab_tvd_std", "Vocab TVD Std"),
        ("topk_tvd_mean", "Top-K TVD Mean"),
        ("topk_tvd_std", "Top-K TVD Std"),
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
        axis.set_xlabel("TVD")
        axis.set_ylabel("Count")
        axis.legend()

    fig.suptitle(
        f"Total Variation Distance (Mean/Std) by Label {args.title_info}"
    )
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            VIZ_DIR,
            f"tvd_dist_{args.suffix}",
        )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

def run(args, data) -> None:

    inference = Inference(model_name=args.model)
    metrics = EntropyMetrics() if args.mode == "logits" else Metrics()

    out = []
    for item in tqdm(data, desc="Processing items ..."):
        hs_dict = inference.run(item=item, args=args)
        if args.mode == "logits":
            metrics_dict = metrics.run(logits=hs_dict["logits"])
            hs_dict.update(metrics_dict)
            del hs_dict["logits"]
        else:
            metrics_dict = metrics.run(
                hidden_states=hs_dict["hidden_states"],
                use_diff_vectors=args.diff_vectors,
                normalize=args.normalize,
            )
            hs_dict.update(metrics_dict)
            del hs_dict["hidden_states"]
        out.append(hs_dict)

    # PLOTTING AREA
    if args.save_viz:
        if args.mode not in ["logits"]:
            figure_path = plot_scores_by_label(args=args, out=out)
            print(f"Saved figure to {figure_path}")

            pair_path = pair_plot(args=args, out=out)
            print(f"Saved pair plot to {pair_path}")
        else:
            entropy_path = plot_entropy_by_label(args=args, out=out)
            print(f"Saved entropy plot to {entropy_path}")
            tvd_path = plot_tvd_by_label(args=args, out=out)
            print(f"Saved tvd plot to {tvd_path}")

        if args.mode not in ["logits", "horizontal"]:
            trajectory_path = plot_layer_profiles(args=args, out=out)
            print(f"Saved trajectory plot to {trajectory_path}")
    
    # ZERO-SHOT SCORES
    if args.score:
        if args.mode != "logits":
            auc_metrics = compute_auc_for_scores(
                out=out,
                args=args,
                score_keys=[
                    "magnitude_change_mean",
                    "angle_change_mean",
                    "length_change_mean",
                    "diff_change_mean",
                    "diff_diff_change_mean",
                    "diff_add_change_mean",
                    "fs_layer_wise_change_mean",
                    "fs_avg_change_mean"
                ],
            )
            print("=" * 50)
            print("AUC results (new metrics):")
            for key, stats in auc_metrics["metrics"].items():
                print(f"{key}: {stats['auc']}")
            print("=" * 50)
    
    # CLASSIFIER
    if args.classifier:
        if args.mode != "logits":
            gmm = ScoreGMM()
            logreg = ScoreLogistic()
            mlp = ScoreMLP()
            gmm_metrics = gmm.run(out=out, suffix=args.suffix, args=args)
            logreg_metrics = logreg.run(out=out, suffix=args.suffix, args=args)
            mlp_metrics = mlp.run(out=out, suffix=args.suffix, args=args)
            print("=" * 50)
            print("Classifier results:")
            print(f"GMM: {gmm_metrics}")
            print(f"Logistic: {logreg_metrics}")
            print(f"MLP: {mlp_metrics}")
            print("=" * 50)

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="last_token",
        choices=["last_token", "pooling", "horizontal", "logits"],
    )
    parser.add_argument("--diff_vectors", type=int, default=0)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--save_viz", type=int, default=0)
    parser.add_argument("--classifier", type=int, default=0)
    parser.add_argument("--score", type=int, default=0)

    args = parser.parse_args()

    # make int bool
    assert args.smoke_test in (0, 1), "smoke_test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)
    assert args.diff_vectors in (0, 1), "diff_vectors must be 0 or 1"
    args.diff_vectors = bool(args.diff_vectors)
    assert args.prefix in (0, 1), "prefix must be 0 or 1"
    args.prefix = bool(args.prefix)
    assert args.normalize in (0, 1), "normalize must be 0 or 1"
    args.normalize = bool(args.normalize)
    assert args.save_viz in (0, 1), "save_viz must be 0 or 1"
    args.save_viz = bool(args.save_viz)
    assert args.classifier in (0, 1), "classifier must be 0 or 1"
    args.classifier = bool(args.classifier)
    assert args.score in (0, 1), "score must be 0 or 1"
    args.score = bool(args.score)

    
    # dirs
    if args.save_viz:
        os.makedirs(OUT_DIR, exist_ok=True)

    if args.classifier:
        os.makedirs(CLASSIFIER_DIR, exist_ok=True)

    print("=" * 50)
    print(f"Running with args:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)

    # get data
    data = load_dataset(args=args)
    data = list(data['test'])

    args.n = len(data)

    # saving suffix and title info 
    suffix = f"{args.model}_{args.dataset}_MODE{args.mode}_DV{int(args.diff_vectors)}_PF{int(args.prefix)}_NO{int(args.normalize)}{'_ST' if args.smoke_test else ''}.pdf"
    title_info = f"{args.model} | {args.dataset} | N={len(data)} | Mode {args.mode} | DV {int(args.diff_vectors)} | Pre {int(args.prefix)} | Norm {int(args.normalize)}"
    args.suffix = suffix
    args.title_info = title_info

    # main run
    run(args=args, data=data)

if __name__ == "__main__":
    main()
