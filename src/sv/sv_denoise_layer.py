from argparse import Namespace
from typing import Any

import numpy as np

from src.sv.sv_main import SVBase


class DenoiseLayerSVBase(SVBase):
    """SV variant that denoises steering vectors with layer-wise PCA manifold."""

    @staticmethod
    def manifold_components_by_layer(
        hidden_states: np.ndarray,
        n_components: int = 10,
    ) -> np.ndarray:
        n_samples, n_layers, d_model = hidden_states.shape
        components = np.zeros((n_layers, d_model, n_components), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_states = hidden_states[:, layer_idx, :]
            centered = layer_states - layer_states.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            components[layer_idx] = vt[:n_components].T.astype(np.float32)

        return components

    @staticmethod
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
            r = steering_vectors[layer_idx]
            u_eff = manifold_components[layer_idx]
            denoised[layer_idx] = u_eff @ (u_eff.T @ r)

        return denoised.astype(np.float32)

    def _denoise_by_layer(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> np.ndarray:
        manifold_components = self.manifold_components_by_layer(
            hidden_states=val_hidden,
            n_components=args.pca_components,
        )
        return self.denoise_steering_vector(steering_vec, manifold_components)

    def fit_projection_space(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        if not args.manifold:
            return val_hidden, steering_vec, None

        steering_vec = self._denoise_by_layer(
            val_hidden=val_hidden,
            steering_vec=steering_vec,
            args=args,
        )
        return val_hidden, steering_vec, None
