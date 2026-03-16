import argparse
import os
import json
import random
from argparse import Namespace

import matplotlib.pyplot as plt
from tqdm import tqdm

from inference import Inference
from coe import Metrics

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data")

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    labels = sorted({item["label"] for item in out})

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for label in labels:
            values = [item[metric_key] for item in out if item["label"] == label]
            if not values:
                continue

            axis.hist(
                values,
                bins=20,
                alpha=0.5,
                label=label_names.get(label, str(label)),
            )

        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Count")
        axis.legend()

    fig.suptitle(f"Angle and Magnitude Scores by Label | N{len(out)} | Model {args.model} | Data {args.dataset}")
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(BASE_DIR, f"scores_by_label_{args.model}_{args.dataset}.pdf")

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
