import argparse
import os
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from inference import Inference
from utils import load_dataset

BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "ld")
os.makedirs(OUT_DIR, exist_ok=True)


class LayerPCAAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def pca_by_layer(self, hidden_states: np.ndarray) -> np.ndarray:
        n_samples, n_layers, dimension_size = hidden_states.shape # n_samples x n_layers x dimension_size
        components = np.zeros((n_layers, n_samples, 2), dtype=np.float32)

        for layer_idx in range(n_layers):
            # apply pca for a given layer, across samples on the hidden dim
            layer_vectors = hidden_states[:, layer_idx, :] # n_samples x dimension_size
            layer_pca = PCA(n_components=2) # n_samples x 2
            components[layer_idx] = layer_pca.fit_transform(layer_vectors)

        return components

    @staticmethod
    def _label_name(label: int) -> str:
        return "human" if label == 0 else "machine"

    def _plot_selected_layer_distributions(
        self,
        pca_components: np.ndarray,
        labels: np.ndarray,
        args: Namespace,
    ) -> str:
        n_layers = pca_components.shape[0]
        selected_layers = np.unique(np.linspace(0, n_layers - 1, num=10, dtype=int)).tolist()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        colors = {0: "tab:blue", 1: "tab:orange"}

        for panel_idx, layer_idx in enumerate(selected_layers):
            axis = axes[panel_idx]
            for label in [0, 1]:
                mask = labels == label
                if not np.any(mask):
                    continue

                points = pca_components[layer_idx, mask, :]
                pc1 = points[:, 0]
                pc2 = points[:, 1]
                finite_mask = np.isfinite(pc1) & np.isfinite(pc2)
                if not np.any(finite_mask):
                    continue

                axis.scatter(
                    pc1[finite_mask],
                    pc2[finite_mask],
                    s=12,
                    alpha=0.5,
                    color=colors[label],
                    label=self._label_name(label),
                )

            axis.set_title(f"Layer {layer_idx + 1}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.legend()
            axis.grid(alpha=0.2)

        for empty_idx in range(len(selected_layers), len(axes)):
            axes[empty_idx].axis("off")

        fig.suptitle(f"PCA Scatter by Layer | {args.model} | {args.data}", y=1.02)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"ld_scatter_{args.model}_{args.data}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _plot_trajectories(
        self,
        pca_components: np.ndarray,
        labels: np.ndarray,
        args: Namespace,
    ) -> str:
        n_layers, _n_samples, _ = pca_components.shape
        fig, axis = plt.subplots(figsize=(10, 8))
        colors = {0: "tab:blue", 1: "tab:orange"}

        for label in [0, 1]:
            mask = labels == label
            if not np.any(mask):
                continue
            traj = np.mean(pca_components[:, mask, :], axis=1)
            axis.plot(
                traj[:, 0],
                traj[:, 1],
                color=colors.get(label, "gray"),
                alpha=0.9,
                linewidth=2.0,
                label=self._label_name(label),
            )
            axis.scatter(
                traj[:, 0],
                traj[:, 1],
                color=colors.get(label, "gray"),
                alpha=0.9,
                s=16,
            )

        axis.set_title(f"Hidden-State Trajectories | {args.model} | {args.data}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend()
        axis.grid(alpha=0.2)

        out_path = os.path.join(OUT_DIR, f"ld_traj_{args.model}_{args.data}.png")
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
        split_data = dataset[args.split]

        if args.n > 0:
            split_data = split_data.select(range(min(len(split_data), args.n)))

        collected_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(split_data, desc=f"LD {args.model} {args.data}", leave=False):
            out = self.inference.run(item, args)
            hidden_states = out.get("hidden_states")
            if hidden_states is None:
                continue

            sample_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                layer_vector = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vector.astype(np.float32))

            # n_layers x n_dimension
            collected_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        if not collected_hidden_states:
            raise ValueError("No hidden states collected. Check mode and dataset.")

        # n_samples x n_layers x n_dimension
        hidden_state_array = np.stack(collected_hidden_states, axis=0)
        label_array = np.array(labels, dtype=np.int32)
        
        pca_components = self.pca_by_layer(hidden_state_array)

        dist_path = self._plot_selected_layer_distributions(pca_components, label_array, args)
        traj_path = self._plot_trajectories(pca_components, label_array, args)

        return {
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "n_samples": int(hidden_state_array.shape[0]),
            "n_layers": int(hidden_state_array.shape[1]),
            "distribution_plot": dist_path,
            "trajectory_plot": traj_path,
        }


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="last_token")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)

    analyzer = LayerPCAAnalyzer(model_name=args.model)
    result = analyzer.run(args)
    print(result)


if __name__ == "__main__":
    main()
