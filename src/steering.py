import argparse
import os
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from inference import Inference
from utils import load_dataset

BASE_DIR = os.getenv("BASE_COE")

OUT_DIR = os.path.join(BASE_DIR, "steering", "test")
os.makedirs(OUT_DIR, exist_ok=True)

# OOD_SETS = ["wikihow_chatgpt", 
#             "reddit_chatgpt", 
#             "wikipedia_chatgpt", 
#             "arxiv_chatgpt"
#             ]

OOD_SETS = ["multisocial_full", 
            "multisocial_en",
            "wikipedia_chatgpt"
            ]


def steering_vector(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    machine_label: int = 1,
    human_label: int = 0,
    normalize: bool = True,
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

    if not normalize:
        return vectors.astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.clip(norms, eps, None)
    return normalized_vectors.astype(np.float32)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, eps, None)).astype(np.float32)


def manifold_components_by_layer(
    hidden_states: np.ndarray,
    n_components: int = 10,
) -> np.ndarray:
    n_samples, n_layers, d_model = hidden_states.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for PCA manifold estimation.")

    effective_k = min(n_components, n_samples - 1, d_model)
    components = np.zeros((n_layers, d_model, effective_k), dtype=np.float32)

    for layer_idx in range(n_layers):
        layer_states = hidden_states[:, layer_idx, :]  # n_samples x d_model
        centered = layer_states - layer_states.mean(axis=0, keepdims=True)
        # centered = U S V^T, rows of V^T are principal directions in feature space.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        layer_basis = vt[:effective_k].T  # d_model x k
        components[layer_idx] = layer_basis.astype(np.float32)

    return components


def denoise_steering_vector(
    steering_vectors: np.ndarray,
    manifold_components: np.ndarray,
) -> np.ndarray:
    if steering_vectors.shape[0] != manifold_components.shape[0]:
        raise ValueError("Layer count mismatch between steering vectors and manifold.")
    if steering_vectors.shape[1] != manifold_components.shape[1]:
        raise ValueError("Hidden dimension mismatch between steering vectors and manifold.")

    denoised = np.zeros_like(steering_vectors, dtype=np.float32)
    for layer_idx in range(steering_vectors.shape[0]):
        r = steering_vectors[layer_idx]  # d_model
        u_eff = manifold_components[layer_idx]  # d_model x k
        # s_flat = U_eff (U_eff^T r)
        denoised[layer_idx] = u_eff @ (u_eff.T @ r)

    return denoised.astype(np.float32)


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
        manifold: bool,
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

        # Layer-wise AUROC using projection score at each layer.
        layer_aurocs: list[float] = []
        if np.unique(labels).size == 2:
            for layer_idx in range(projections.shape[1]):
                auc = roc_auc_score(labels, projections[:, layer_idx])
                layer_aurocs.append(float(auc))
        else:
            layer_aurocs = [float("nan")] * projections.shape[1]

        auc_axis = axis.twinx()
        auc_axis.plot(
            layers,
            layer_aurocs,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="AUROC",
        )
        auc_axis.set_ylabel("AUROC")
        auc_axis.set_ylim(0.0, 1.0)

        last_layer_auc = layer_aurocs[-1] if layer_aurocs else float("nan")
        last_layer_auc_text = f"{last_layer_auc:.3f}" if not np.isnan(last_layer_auc) else "NA"

        axis.set_title(
            f"Steering Projection by Layer | {model} | steering={steering_domain} | eval={eval_domain} | manifold={int(manifold)} | last-layer AUROC={last_layer_auc_text}"
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Projection Score")
        axis.grid(alpha=0.25)
        mean_auc = np.nanmean(layer_aurocs)
        if not np.isnan(mean_auc):
            axis.text(
                0.01,
                0.98,
                f"Mean AUROC: {mean_auc:.3f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
            )

        lines_1, labels_1 = axis.get_legend_handles_labels()
        lines_2, labels_2 = auc_axis.get_legend_handles_labels()
        axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        out_path = os.path.join(
            OUT_DIR,
            f"steering_projection_{model}_{steering_domain}_on_{eval_domain}_manifold_{int(manifold)}.png",
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
        steering_vec = steering_vector(val_hidden, val_labels, normalize=False)
        if args.manifold:
            manifold_components = manifold_components_by_layer(
                hidden_states=val_hidden,
                n_components=10,
            )
            steering_vec = denoise_steering_vector(steering_vec, manifold_components)
        steering_vec = normalize_vectors(steering_vec)

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
            manifold=args.manifold,
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
                    manifold=args.manifold,
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
            "manifold": bool(args.manifold),
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
    parser.add_argument("--manifold", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    if args.manifold not in (0, 1):
        raise ValueError("manifold must be 0 or 1")
    if args.mode != "last_token":
        raise ValueError("This script expects --mode last_token.")

    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.manifold = bool(args.manifold)

    analyzer = SteeringAnalyzer(model_name=args.model)
    result = analyzer.run(args)
    print(result)


if __name__ == "__main__":
    main()
