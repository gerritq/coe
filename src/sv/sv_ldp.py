from argparse import Namespace
from typing import Any

import numpy as np

from src.sv.sv_main import SVBase


class LDPSVBase(SVBase):
    """Low-dimensional projection variant with PCA fit across all layers."""

    @staticmethod
    def fit_pca_manifold_across_layers(
        hidden_states: np.ndarray,
        n_components: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_layers, d_model = hidden_states.shape
        k = max(1, min(n_components, n_samples * n_layers, d_model))

        flattened = hidden_states.reshape(n_samples * n_layers, d_model)
        manifold_mean = flattened.mean(axis=0).astype(np.float32)
        centered = flattened - manifold_mean[None, :]

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        manifold_components = vt[:k].T.astype(np.float32)
        projected_flat = (centered @ manifold_components).astype(np.float32)
        projected = projected_flat.reshape(n_samples, n_layers, k)

        return projected, manifold_mean, manifold_components

    @staticmethod
    def project_hidden_states_with_manifold(
        hidden_states: np.ndarray,
        manifold_mean: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        if hidden_states.shape[2] != manifold_components.shape[0]:
            raise ValueError("Hidden dimension mismatch between hidden states and manifold.")
        if manifold_mean.shape[0] != manifold_components.shape[0]:
            raise ValueError("Manifold mean shape mismatch.")

        centered = hidden_states - manifold_mean[None, None, :]
        return np.einsum("sld,dk->slk", centered, manifold_components).astype(np.float32)

    @staticmethod
    def project_steering_vectors_with_manifold(
        steering_vectors: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        if steering_vectors.shape[1] != manifold_components.shape[0]:
            raise ValueError("Hidden dimension mismatch between steering vectors and manifold.")

        return np.einsum("ld,dk->lk", steering_vectors, manifold_components).astype(np.float32)

    def fit_projection_space(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        if not args.manifold:
            return val_hidden, steering_vec, None

        val_hidden_proj, manifold_mean, manifold_components = self.fit_pca_manifold_across_layers(
            hidden_states=val_hidden,
            n_components=args.pca_components,
        )
        steering_vec_proj = self.project_steering_vectors_with_manifold(
            steering_vectors=steering_vec,
            manifold_components=manifold_components,
        )

        projection_state = {
            "mean": manifold_mean,
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
            manifold_mean=projection_state["mean"],
            manifold_components=projection_state["components"],
        )
