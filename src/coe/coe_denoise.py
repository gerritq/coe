from typing import Any

import numpy as np
import torch

from src.coe.coe_main import COEAnalyzer


class COEDenoiseAnalyzer(COEAnalyzer):
    """COE analyzer with layer-wise PCA denoising before metric computation."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.n_components = 10

    @staticmethod
    def _fit_layer_pca(
        hidden_records: list[dict[str, Any]],
        n_components: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_layers = len(hidden_records[0]["hidden_states"])
        layer_means: list[np.ndarray] = []
        layer_components: list[np.ndarray] = []

        for layer_idx in range(n_layers):
            layer_vectors = []
            for record in hidden_records:
                vec = record["hidden_states"][layer_idx].detach().float().cpu().numpy().reshape(-1)
                layer_vectors.append(vec)
            x = np.stack(layer_vectors, axis=0)  # n_samples x d_model

            mean = x.mean(axis=0)
            centered = x - mean[None, :]
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            k = max(1, min(n_components, vt.shape[0]))
            components = vt[:k].T  # d_model x k

            layer_means.append(mean.astype(np.float32))
            layer_components.append(components.astype(np.float32))

        return layer_means, layer_components

    @staticmethod
    def _denoise_vector(
        vector: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
    ) -> np.ndarray:
        centered = vector - mean
        projected = components @ (components.T @ centered)
        return (mean + projected).astype(np.float32)

    def _postprocess_hidden_states(
        self,
        hidden_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not hidden_records:
            return hidden_records

        layer_means, layer_components = self._fit_layer_pca(
            hidden_records=hidden_records,
            n_components=self.n_components,
        )

        denoised_records: list[dict[str, Any]] = []
        for record in hidden_records:
            new_record = dict(record)
            denoised_layers: list[torch.Tensor] = []
            for layer_idx, layer_tensor in enumerate(record["hidden_states"]):
                vec = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                denoised_vec = self._denoise_vector(
                    vector=vec,
                    mean=layer_means[layer_idx],
                    components=layer_components[layer_idx],
                )
                denoised_layers.append(torch.tensor(denoised_vec, dtype=layer_tensor.dtype).reshape_as(layer_tensor))

            new_record["hidden_states"] = tuple(denoised_layers)
            denoised_records.append(new_record)

        return denoised_records
