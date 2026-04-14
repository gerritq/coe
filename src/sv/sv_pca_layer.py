from argparse import Namespace
from typing import Any

import numpy as np

from src.sv.sv_main import SVBase
from src.utils import load_dataset


class PCALayerSVBase(SVBase):
    """Per-layer PCA manifold; steering and projection are computed in that manifold."""

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
            layer_mean = layer_states.mean(axis=0).astype(np.float32)
            centered = layer_states - layer_mean[None, :]

            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            layer_basis = vt[:k].T.astype(np.float32)  # d_model x k

            means[layer_idx] = layer_mean
            components[layer_idx] = layer_basis
            projected[:, layer_idx, :] = (centered @ layer_basis).astype(np.float32)

        return projected, means, components

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
        # Compatibility path for base flow; main run() computes SV directly in projected space.
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

    def run(self, args: Namespace) -> dict[str, Any]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)

        projection_state: dict[str, np.ndarray] | None = None
        if args.manifold:
            val_hidden_proj, manifold_means, manifold_components = self.fit_pca_manifold_by_layer(
                hidden_states=val_hidden,
                n_components=args.pca_components,
            )
            projection_state = {
                "means": manifold_means,
                "components": manifold_components,
            }
        else:
            val_hidden_proj = val_hidden

        # Requested order: compute SV after projecting activations to per-layer PCA manifold.
        steering_vec_proj = self.raw_steering_vector(hidden_states=val_hidden_proj, labels=val_labels)
        steering_vec_proj = self.normalize_vectors(steering_vec_proj)

        val_projection = self.compute_projection(
            val_hidden_states=val_hidden_proj,
            test_hidden_states=val_hidden_proj,
            steering_vectors=steering_vec_proj,
            args=args,
        )

        test_hidden, test_labels = self._collect_hidden_states(data=dataset["test"], token_mode=args.token_mode)
        test_hidden_proj = self.transform_hidden_states(
            hidden_states=test_hidden,
            projection_state=projection_state,
            args=args,
        )
        test_projection = self.compute_projection(
            val_hidden_states=val_hidden_proj,
            test_hidden_states=test_hidden_proj,
            steering_vectors=steering_vec_proj,
            args=args,
        )

        layer_min: np.ndarray | None = None
        layer_max: np.ndarray | None = None
        if args.normalize_scores:
            layer_min, layer_max = self.layer_minmax(val_projection)
            val_projection = self.apply_layer_minmax(val_projection, layer_min, layer_max)
            test_projection = self.apply_layer_minmax(test_projection, layer_min, layer_max)

        self._save_projection_metrics(
            projections=test_projection,
            labels=test_labels,
            val_projections=val_projection,
            val_labels=val_labels,
            args=args,
            steering_domain=args.dataset,
            eval_domain=args.dataset,
            out_dir=out_dir,
        )

        self._plot_test_projections(
            projections=test_projection,
            labels=test_labels,
            val_projections=val_projection,
            val_labels=val_labels,
            args=args,
            out_dir=out_dir,
        )

        for ood_dataset_name in args.ood:
            if ood_dataset_name == args.dataset:
                continue

            ood_data_args = Namespace(
                dataset=ood_dataset_name,
                prefix=bool(args.prefix),
                smoke_test=bool(args.smoke_test),
            )
            ood_dataset = load_dataset(ood_data_args)
            ood_hidden, ood_labels = self._collect_hidden_states(
                data=ood_dataset["test"],
                token_mode=args.token_mode,
            )
            ood_hidden_proj = self.transform_hidden_states(
                hidden_states=ood_hidden,
                projection_state=projection_state,
                args=args,
            )
            ood_projection = self.compute_projection(
                val_hidden_states=val_hidden_proj,
                test_hidden_states=ood_hidden_proj,
                steering_vectors=steering_vec_proj,
                args=args,
            )
            if args.normalize_scores:
                if layer_min is None or layer_max is None:
                    raise ValueError("Layer min/max stats not initialized for score normalization.")
                ood_projection = self.apply_layer_minmax(ood_projection, layer_min, layer_max)

            self._plot_test_projections(
                projections=ood_projection,
                labels=ood_labels,
                val_projections=val_projection,
                val_labels=val_labels,
                args=args,
                eval_domain=ood_dataset_name,
                out_dir=out_dir,
            )
            self._save_projection_metrics(
                projections=ood_projection,
                labels=ood_labels,
                val_projections=val_projection,
                val_labels=val_labels,
                args=args,
                steering_domain=args.dataset,
                eval_domain=ood_dataset_name,
                out_dir=out_dir,
            )

        return {
            "model": args.model,
            "dataset": args.dataset,
            "val_split": "val",
            "test_split": "test",
            "n_val": int(val_hidden.shape[0]),
            "n_test": int(test_hidden.shape[0]),
            "n_layers": int(steering_vec_proj.shape[0]),
            "d_model": int(steering_vec_proj.shape[1]),
            "ood": bool(args.ood),
            "manifold": bool(args.manifold),
        }
