from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.descriptives.desc_run import OUT_DIR
from src.inference import Inference
from src.utils import load_dataset


class PCATrajectoryAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    @staticmethod
    def _label_name(label: int) -> str:
        return "human" if label == 0 else "machine"

    @staticmethod
    def _validate_dim(dim: int) -> int:
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        return dim

    def pca_across_layers(self, hidden_states: np.ndarray) -> np.ndarray:
        n_samples, n_layers, _dimension_size = hidden_states.shape
        flattened = hidden_states.reshape(n_samples * n_layers, -1)

        # Global PCA so every layer/sample is represented in one shared space.
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(flattened)

        # Return n_layers x n_samples x 3 for convenient layer-wise plotting.
        return transformed.reshape(n_samples, n_layers, 3).transpose(1, 0, 2).astype(np.float32)

    def _plot_set_trajectories(
        self,
        pca_components: np.ndarray,
        labels: np.ndarray,
        args: Namespace,
        dim: int,
    ) -> str:
        dim = self._validate_dim(dim)

        if dim == 3:
            fig = plt.figure(figsize=(16, 7))
            human_axis = fig.add_subplot(1, 2, 1, projection="3d")
            machine_axis = fig.add_subplot(1, 2, 2, projection="3d")
        else:
            fig, (human_axis, machine_axis) = plt.subplots(1, 2, figsize=(16, 7))

        axis_by_label = {0: human_axis, 1: machine_axis}
        line_colors = {0: "tab:blue", 1: "tab:orange"}
        start_color = "tab:green"
        end_color = "tab:red"

        for label in [0, 1]:
            mask = labels == label
            axis = axis_by_label[label]
            set_name = self._label_name(label)

            axis.set_title(f"{set_name.capitalize()} Trajectory")
            if not np.any(mask):
                axis.text(0.5, 0.5, "No samples", transform=axis.transAxes, ha="center", va="center")
                continue

            # Mean set trajectory across layers: n_layers x 3
            trajectory = np.mean(pca_components[:, mask, :], axis=1)

            if dim == 3:
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color=line_colors[label],
                    linewidth=2.5,
                    alpha=0.95,
                    label=f"{set_name} trajectory",
                )
                axis.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    trajectory[0, 2],
                    color=start_color,
                    s=70,
                    marker="o",
                    label="start",
                    zorder=5,
                )
                axis.text(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    trajectory[0, 2],
                    "start",
                    color=start_color,
                )
                axis.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    trajectory[-1, 2],
                    color=end_color,
                    s=85,
                    marker="X",
                    label="end",
                    zorder=6,
                )
                axis.text(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    trajectory[-1, 2],
                    "end",
                    color=end_color,
                )
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")
                axis.set_zlabel("PC3")
            else:
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=line_colors[label],
                    linewidth=2.5,
                    alpha=0.95,
                    label=f"{set_name} trajectory",
                )
                axis.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    color=start_color,
                    s=70,
                    marker="o",
                    label="start",
                    zorder=5,
                )
                axis.annotate(
                    "start",
                    (trajectory[0, 0], trajectory[0, 1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    color=start_color,
                )
                axis.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    color=end_color,
                    s=85,
                    marker="X",
                    label="end",
                    zorder=6,
                )
                axis.annotate(
                    "end",
                    (trajectory[-1, 0], trajectory[-1, 1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    color=end_color,
                )
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")

            axis.grid(alpha=0.2)
            axis.legend(loc="best")

        fig.suptitle(
            f"Layer Trajectories ({dim}D) | {args.model} | {args.data} | {args.mode} | {args.split}",
            y=1.02,
        )
        fig.tight_layout()

        out_path = f"{OUT_DIR}/pca_traj_{dim}d_{args.model}_{args.data}_{args.mode}_{args.split}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def run(self, args: Namespace) -> dict[str, Any]:
        dim = self._validate_dim(int(getattr(args, "dim", 3)))

        data_args = Namespace(
            dataset=args.data,
            prefix=bool(args.prefix),
            smoke_test=bool(args.smoke_test),
        )
        dataset = load_dataset(data_args)
        split_data = dataset[args.split]

        collected_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(split_data, desc=f"PCA Traj {args.model} {args.data}", leave=False):
            out = self.inference.run(item, args)
            hidden_states = out.get("hidden_states")
            if hidden_states is None:
                continue

            sample_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                layer_vector = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vector.astype(np.float32))

            collected_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        if not collected_hidden_states:
            raise ValueError("No hidden states collected. Check mode and dataset.")

        hidden_state_array = np.stack(collected_hidden_states, axis=0)
        label_array = np.array(labels, dtype=np.int32)

        pca_components = self.pca_across_layers(hidden_state_array)
        plot_path = self._plot_set_trajectories(pca_components, label_array, args, dim=dim)

        return {
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "dim": dim,
            "n_samples": int(hidden_state_array.shape[0]),
            "n_layers": int(hidden_state_array.shape[1]),
            "trajectory_plot": plot_path,
        }
