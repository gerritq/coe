from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.descriptives.desc_run import OUT_DIR
from src.inference import Inference
from src.utils import load_dataset


class PCAAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def pca_across_layers(self, hidden_states: np.ndarray) -> np.ndarray:
        n_samples, n_layers, _dimension_size = hidden_states.shape  # n_samples x n_layers x dimension_size
        flattened = hidden_states.reshape(n_samples * n_layers, -1)
        global_pca = PCA(n_components=3)
        transformed = global_pca.fit_transform(flattened)

        # reshape to n_layers x n_samples x 3 for downstream plotting compatibility
        return transformed.reshape(n_samples, n_layers, 3).transpose(1, 0, 2).astype(np.float32)

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

        fig, axes = plt.subplots(2, 5, figsize=(22, 9), subplot_kw={"projection": "3d"})
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
                pc3 = points[:, 2]
                finite_mask = np.isfinite(pc1) & np.isfinite(pc2) & np.isfinite(pc3)
                if not np.any(finite_mask):
                    continue

                axis.scatter(
                    pc1[finite_mask],
                    pc2[finite_mask],
                    pc3[finite_mask],
                    s=12,
                    alpha=0.5,
                    color=colors[label],
                    label=self._label_name(label),
                )

            axis.set_title(f"Layer {layer_idx + 1}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.set_zlabel("PC3")
            axis.legend()
            axis.grid(alpha=0.2)

        for empty_idx in range(len(selected_layers), len(axes)):
            axes[empty_idx].axis("off")

        fig.suptitle(
            f"PCA 3D Scatter by Layer | {args.model} | {args.data} | {args.mode} | {args.split}",
            y=1.02,
        )
        fig.tight_layout()

        out_path = f"{OUT_DIR}/ld_scatter3d_{args.model}_{args.data}_{args.mode}_{args.split}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _plot_trajectories(
        self,
        pca_components: np.ndarray,
        labels: np.ndarray,
        args: Namespace,
    ) -> str:
        _n_layers, _n_samples, _ = pca_components.shape
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

        axis.set_title(f"Hidden-State Trajectories | {args.model} | {args.data} | {args.mode} | {args.split}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend()
        axis.grid(alpha=0.2)

        out_path = f"{OUT_DIR}/ld_traj_{args.model}_{args.data}_{args.mode}_{args.split}.png"
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

        pca_components = self.pca_across_layers(hidden_state_array)

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
