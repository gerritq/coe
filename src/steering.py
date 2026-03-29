import argparse
import os
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from inference import Inference
from utils import load_dataset

BASE_DIR = os.getenv("BASE_COE")
if BASE_DIR is None:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(BASE_DIR, "steering")
os.makedirs(OUT_DIR, exist_ok=True)

OOD_SETS = ["wikihow_chatgpt", "reddit_chatgpt", "wikipedia_chatgpt", "arxiv_chatgpt"]


def steering_vector(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    machine_label: int = 1,
    human_label: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    machine_mask = labels == machine_label
    human_mask = labels == human_label

    if not np.any(machine_mask):
        raise ValueError("No machine samples found to compute steering vector.")
    if not np.any(human_mask):
        raise ValueError("No human samples found to compute steering vector.")

    machine_mean = hidden_states[machine_mask].mean(axis=0)  # n_layers x d_model
    human_mean = hidden_states[human_mask].mean(axis=0)  # n_layers x d_model
    vectors = machine_mean - human_mean

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.clip(norms, eps, None)
    return normalized_vectors.astype(np.float32)


def steering_projection(
    hidden_states: np.ndarray,
    normalized_steering_vectors: np.ndarray,
) -> np.ndarray:
    if hidden_states.shape[1:] != normalized_steering_vectors.shape:
        raise ValueError(
            "Shape mismatch: hidden_states must be [n_samples, n_layers, d_model] "
            "and steering vectors must be [n_layers, d_model]."
        )

    # Dot product per sample and per layer: n_samples x n_layers
    return np.einsum("sld,ld->sl", hidden_states, normalized_steering_vectors)


class SteeringAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def _collect_hidden_states(
        self,
        split_data: Any,
        mode: str = "last_token",
        n_limit: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_limit > 0:
            split_data = split_data.select(range(min(len(split_data), n_limit)))

        inference_args = Namespace(mode=mode)

        all_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(split_data, desc="Inference", leave=False):
            out = self.inference.run(item, inference_args)
            hidden_states = out.get("hidden_states")
            if hidden_states is None:
                continue

            sample_layers = []
            for layer_tensor in hidden_states:
                layer_vec = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vec.astype(np.float32))

            all_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        if not all_hidden_states:
            raise ValueError("No hidden states collected. Check dataset and mode.")

        return (
            np.stack(all_hidden_states, axis=0),  # n_samples x n_layers x d_model
            np.array(labels, dtype=np.int32),
        )

    @staticmethod
    def _plot_test_projections(
        projections: np.ndarray,
        labels: np.ndarray,
        model: str,
        steering_domain: str,
        eval_domain: str,
    ) -> str:
        fig, axis = plt.subplots(figsize=(10, 6))
        layers = np.arange(1, projections.shape[1] + 1)
        label_names = {0: "human", 1: "machine"}
        colors = {0: "tab:blue", 1: "tab:orange"}

        for label in [0, 1]:
            mask = labels == label
            if not np.any(mask):
                continue

            class_proj = projections[mask]
            for sample_idx, sample_proj in enumerate(class_proj):
                axis.plot(
                    layers,
                    sample_proj,
                    color=colors[label],
                    alpha=0.2,
                    linewidth=1.0,
                    label=label_names[label] if sample_idx == 0 else None,
                )

            class_mean = class_proj.mean(axis=0)
            axis.plot(
                layers,
                class_mean,
                color=colors[label],
                linewidth=2.5,
            )

        axis.set_title(
            f"Steering Projection by Layer | {model} | steering={steering_domain} | eval={eval_domain}"
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Projection Score")
        axis.grid(alpha=0.25)
        axis.legend()

        out_path = os.path.join(
            OUT_DIR,
            f"steering_projection_{model}_{steering_domain}_on_{eval_domain}.png",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def run(self, args: Namespace) -> dict[str, Any]:
        data_args = Namespace(
            dataset=args.data,
            prefix=bool(args.prefix),
            smoke_test=bool(args.smoke_test),
        )
        dataset = load_dataset(data_args)

        if args.val_split not in dataset:
            raise ValueError(f"Validation split '{args.val_split}' not found in dataset.")
        if args.test_split not in dataset:
            raise ValueError(f"Test split '{args.test_split}' not found in dataset.")

        val_hidden, val_labels = self._collect_hidden_states(
            split_data=dataset[args.val_split],
            mode=args.mode,
            n_limit=args.n_val,
        )
        steering_vec = steering_vector(val_hidden, val_labels)

        test_hidden, test_labels = self._collect_hidden_states(
            split_data=dataset[args.test_split],
            mode=args.mode,
            n_limit=args.n_test,
        )
        test_projection = steering_projection(test_hidden, steering_vec)

        plot_path = self._plot_test_projections(
            projections=test_projection,
            labels=test_labels,
            model=args.model,
            steering_domain=args.data,
            eval_domain=args.data,
        )

        ood_plots: dict[str, str] = {}
        if args.ood:
            for ood_dataset_name in OOD_SETS:
                if ood_dataset_name == args.data:
                    continue

                ood_data_args = Namespace(
                    dataset=ood_dataset_name,
                    prefix=bool(args.prefix),
                    smoke_test=bool(args.smoke_test),
                )
                ood_dataset = load_dataset(ood_data_args)
                if args.test_split not in ood_dataset:
                    raise ValueError(
                        f"Test split '{args.test_split}' not found in OOD dataset '{ood_dataset_name}'."
                    )

                ood_hidden, ood_labels = self._collect_hidden_states(
                    split_data=ood_dataset[args.test_split],
                    mode=args.mode,
                    n_limit=args.n_test,
                )
                ood_projection = steering_projection(ood_hidden, steering_vec)
                ood_plot = self._plot_test_projections(
                    projections=ood_projection,
                    labels=ood_labels,
                    model=args.model,
                    steering_domain=args.data,
                    eval_domain=ood_dataset_name,
                )
                ood_plots[ood_dataset_name] = ood_plot

        return {
            "model": args.model,
            "data": args.data,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "n_val": int(val_hidden.shape[0]),
            "n_test": int(test_hidden.shape[0]),
            "n_layers": int(steering_vec.shape[0]),
            "d_model": int(steering_vec.shape[1]),
            "plot_path": plot_path,
            "ood": bool(args.ood),
            "ood_plots": ood_plots,
        }


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="last_token")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--n_val", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--ood", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    if args.mode != "last_token":
        raise ValueError("This script expects --mode last_token.")

    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)

    analyzer = SteeringAnalyzer(model_name=args.model)
    result = analyzer.run(args)
    print(result)


if __name__ == "__main__":
    main()
