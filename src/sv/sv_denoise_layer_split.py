from argparse import Namespace

import numpy as np

from src.sv.sv_main import SVBase
from src.utils import load_dataset


class DenoiseLayerSplitSVBase(SVBase):
    """Denoise val activations per layer with split-specific PCA (human/machine), then compute SV."""

    @staticmethod
    def manifold_components_by_layer(
        hidden_states: np.ndarray,
        n_components: int = 10,
    ) -> np.ndarray:
        n_samples, n_layers, d_model = hidden_states.shape
        k = max(1, min(n_components, n_samples, d_model))
        components = np.zeros((n_layers, d_model, k), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_states = hidden_states[:, layer_idx, :]
            centered = layer_states - layer_states.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            components[layer_idx] = vt[:k].T.astype(np.float32)

        return components

    @staticmethod
    def _split_components_by_layer(
        hidden_states: np.ndarray,
        labels: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        human_states = hidden_states[labels == 0]
        machine_states = hidden_states[labels == 1]
        if human_states.shape[0] == 0 or machine_states.shape[0] == 0:
            raise ValueError("Both human and machine samples are required for split-wise denoising.")

        human_components = DenoiseLayerSplitSVBase.manifold_components_by_layer(
            hidden_states=human_states,
            n_components=n_components,
        )
        machine_components = DenoiseLayerSplitSVBase.manifold_components_by_layer(
            hidden_states=machine_states,
            n_components=n_components,
        )
        return human_components, machine_components

    @staticmethod
    def _denoise_with_components(
        hidden_states: np.ndarray,
        manifold_components: np.ndarray,
    ) -> np.ndarray:
        # Down-project then up-project (layer-wise reconstruction in PCA subspace).
        low_dim = np.einsum("sld,ldk->slk", hidden_states, manifold_components)
        recon = np.einsum("slk,ldk->sld", low_dim, manifold_components)
        return recon.astype(np.float32)

    @staticmethod
    def denoise_hidden_states_by_split(
        hidden_states: np.ndarray,
        labels: np.ndarray,
        human_components: np.ndarray,
        machine_components: np.ndarray,
    ) -> np.ndarray:
        denoised = np.zeros_like(hidden_states, dtype=np.float32)

        human_mask = labels == 0
        machine_mask = labels == 1

        if np.any(human_mask):
            denoised[human_mask] = DenoiseLayerSplitSVBase._denoise_with_components(
                hidden_states=hidden_states[human_mask],
                manifold_components=human_components,
            )
        if np.any(machine_mask):
            denoised[machine_mask] = DenoiseLayerSplitSVBase._denoise_with_components(
                hidden_states=hidden_states[machine_mask],
                manifold_components=machine_components,
            )

        return denoised

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)

        if args.manifold:
            human_components, machine_components = self._split_components_by_layer(
                hidden_states=val_hidden,
                labels=val_labels,
                n_components=args.pca_components,
            )
            val_hidden_denoised = self.denoise_hidden_states_by_split(
                hidden_states=val_hidden,
                labels=val_labels,
                human_components=human_components,
                machine_components=machine_components,
            )
        else:
            val_hidden_denoised = val_hidden

        # Core idea: compute SV from split-denoised val activations.
        steering_vec = self.raw_steering_vector(hidden_states=val_hidden_denoised, labels=val_labels)
        steering_vec = self.normalize_vectors(steering_vec)

        val_projection = self.compute_projection(
            val_hidden_states=val_hidden_denoised,
            test_hidden_states=val_hidden_denoised,
            steering_vectors=steering_vec,
            args=args,
        )

        # Keep test/OOD activations raw in this variant.
        test_hidden, test_labels = self._collect_hidden_states(data=dataset["test"], token_mode=args.token_mode)
        test_projection = self.compute_projection(
            val_hidden_states=val_hidden_denoised,
            test_hidden_states=test_hidden,
            steering_vectors=steering_vec,
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
            ood_projection = self.compute_projection(
                val_hidden_states=val_hidden_denoised,
                test_hidden_states=ood_hidden,
                steering_vectors=steering_vec,
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
            "n_layers": int(steering_vec.shape[0]),
            "d_model": int(steering_vec.shape[1]),
            "ood": bool(args.ood),
            "manifold": bool(args.manifold),
        }
