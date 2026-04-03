from argparse import Namespace
from typing import Any

import numpy as np

from src.sv.sv_main import SVBase


class LDPSVBase(SVBase):
    """Low-dimensional projection variant built on top of SVBase."""
    @staticmethod
    def fit_pca_manifold_by_layer(
        hidden_states: np.ndarray,
        n_components: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_layers, d_model = hidden_states.shape
        k = max(1, min(n_components, n_samples, d_model))

        means = np.zeros((n_layers, d_model), dtype=np.float32)
        components = np.zeros((n_layers, d_model, k), dtype=np.float32)
        projected = np.zeros((n_samples, n_layers, k), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_states = hidden_states[:, layer_idx, :]
            layer_mean = layer_states.mean(axis=0)
            centered = layer_states - layer_mean[None, :]

            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            layer_basis = vt[:k].T

            means[layer_idx] = layer_mean.astype(np.float32)
            components[layer_idx] = layer_basis.astype(np.float32)
            projected[:, layer_idx, :] = (centered @ layer_basis).astype(np.float32)

        return projected.astype(np.float32), means.astype(np.float32), components.astype(np.float32)

    @staticmethod
    def project_hidden_states_with_manifold(
        hidden_states: np.ndarray,
        manifold_means: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        if hidden_states.shape[1] != manifold_components.shape[0]:
            raise ValueError("Layer count mismatch between hidden states and manifold.")
        if hidden_states.shape[2] != manifold_components.shape[1]:
            raise ValueError("Hidden dimension mismatch between hidden states and manifold.")
        if manifold_means.shape != manifold_components.shape[:2]:
            raise ValueError("Manifold means shape mismatch.")

        centered = hidden_states - manifold_means[None, :, :]
        return np.einsum("sld,ldk->slk", centered, manifold_components).astype(np.float32)

    @staticmethod
    def project_steering_vectors_with_manifold(
        steering_vectors: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        if steering_vectors.shape[0] != manifold_components.shape[0]:
            raise ValueError("Layer count mismatch between steering vectors and manifold.")
        if steering_vectors.shape[1] != manifold_components.shape[1]:
            raise ValueError("Hidden dimension mismatch between steering vectors and manifold.")

        return np.einsum("ld,ldk->lk", steering_vectors, manifold_components).astype(np.float32)

    def fit_projection_space(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        if not args.manifold:
            return val_hidden, steering_vec, None

        val_hidden_proj, manifold_means, manifold_components = self.fit_pca_manifold_by_layer(
            hidden_states=val_hidden,
            n_components=args.pca_components,
        )
        steering_vec_proj = self.project_steering_vectors_with_manifold(
            steering_vectors=steering_vec,
            manifold_components=manifold_components,
        )

        projection_state = {
            "means": manifold_means,
            "components": manifold_components,
        }
        return val_hidden_proj, steering_vec_proj, projection_state

    def transform_hidden_states(
        self,
        hidden_states: np.ndarray,
        projection_state: Any,
        args: Namespace,
    ) -> np.ndarray:
        if not args.manifold or projection_state is None:
            return hidden_states

        return self.project_hidden_states_with_manifold(
            hidden_states=hidden_states,
            manifold_means=projection_state["means"],
            manifold_components=projection_state["components"],
        )
