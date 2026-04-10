from argparse import Namespace
from typing import Any

import numpy as np

from src.sv.sv_main import SVBase


class DenoiseSVBase(SVBase):
    """SV variant that denoises steering vectors with PCA fit across all layers."""

    @staticmethod
    def manifold_components_across_layers(
        hidden_states: np.ndarray,
        n_components: int = 10,
    ) -> np.ndarray:
        n_samples, n_layers, d_model = hidden_states.shape
        k = max(1, min(n_components, n_samples * n_layers, d_model))

        flattened = hidden_states.reshape(n_samples * n_layers, d_model)
        centered = flattened - flattened.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[:k].T.astype(np.float32)

    @staticmethod
    def denoise_steering_vector(
        steering_vectors: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        if steering_vectors.shape[1] != manifold_components.shape[0]:
            raise ValueError("Hidden dimension mismatch between steering vectors and manifold.")

        denoised = np.zeros_like(steering_vectors, dtype=np.float32)
        for layer_idx in range(steering_vectors.shape[0]):
            r = steering_vectors[layer_idx]
            denoised[layer_idx] = manifold_components @ (manifold_components.T @ r)

        return denoised.astype(np.float32)

    def fit_projection_space(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        if not args.manifold:
            return val_hidden, steering_vec, None

        manifold_components = self.manifold_components_across_layers(
            hidden_states=val_hidden,
            n_components=args.pca_components,
        )
        steering_vec = self.denoise_steering_vector(steering_vec, manifold_components)
        return val_hidden, steering_vec, None
