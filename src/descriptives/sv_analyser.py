from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.descriptives.desc_run import COMPARE_DATASETS, OUT_DIR
from src.inference import Inference
from src.sv.sv_main import SVBase
from src.utils import load_dataset


class SVAnalyser:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def _collect_hidden_states(
        self,
        data: Any,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(data, desc="SV Inference", leave=False):
            out = self.inference.run(item, args=Namespace(mode=mode))
            hidden_states = out.get("hidden_states")
            if hidden_states is None:
                continue

            sample_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                layer_vector = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vector.astype(np.float32))

            all_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        if not all_hidden_states:
            raise ValueError("No hidden states collected for steering-vector analysis.")

        return (
            np.stack(all_hidden_states, axis=0),
            np.array(labels, dtype=np.int32),
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
        return float(np.dot(a, b) / denom)

    def _cosine_similarity_matrix(self, vectors: dict[str, np.ndarray]) -> np.ndarray:
        names = list(vectors.keys())
        size = len(names)
        matrix = np.zeros((size, size), dtype=np.float32)

        for i, left_name in enumerate(names):
            for j, right_name in enumerate(names):
                matrix[i, j] = self._cosine_similarity(vectors[left_name], vectors[right_name])

        return matrix

    def _plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: list[str],
        args: Namespace,
    ) -> str:
        fig, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)

        axis.set_xticks(np.arange(len(labels)))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_yticklabels(labels)
        axis.set_title(f"Last-Layer SV Cosine Similarity | {args.model} | {args.mode} | {args.split}")
        axis.set_xlabel("Dataset")
        axis.set_ylabel("Dataset")

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Cosine similarity")
        fig.tight_layout()

        out_path = f"{OUT_DIR}/sv_confusion_last_layer_{args.model}_{args.mode}_{args.split}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def run(self, args: Namespace) -> dict[str, Any]:
        last_layer_vectors: dict[str, np.ndarray] = {}
        n_layers: int | None = None

        for dataset_name in COMPARE_DATASETS:
            dataset = load_dataset(
                Namespace(
                    dataset=dataset_name,
                    prefix=bool(args.prefix),
                    smoke_test=bool(args.smoke_test),
                )
            )
            val_data = dataset["val"]
            if args.n > 0:
                val_data = val_data.select(range(min(len(val_data), args.n)))

            val_hidden, val_labels = self._collect_hidden_states(data=val_data, mode=args.mode)
            steering_vectors = SVBase.raw_steering_vector(hidden_states=val_hidden, labels=val_labels)

            if n_layers is None:
                n_layers = int(steering_vectors.shape[0])

            last_layer_vectors[dataset_name] = steering_vectors[-1].astype(np.float32)

        dataset_order = list(last_layer_vectors.keys())
        similarity_matrix = self._cosine_similarity_matrix(last_layer_vectors)
        matrix_path = self._plot_confusion_matrix(similarity_matrix, dataset_order, args)

        return {
            "model": args.model,
            "split": "val",
            "compare_datasets": dataset_order,
            "n_layers": n_layers,
            "matrix_shape": list(similarity_matrix.shape),
            "last_layer_cosine_similarity": similarity_matrix.tolist(),
            "confusion_matrix_plot": matrix_path,
        }
