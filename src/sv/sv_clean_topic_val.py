from argparse import Namespace

import numpy as np

from src.sv.sv_main import SVBase
from src.utils import load_dataset


class CleanTopicValSVBase(SVBase):
    """Ablate validation states only; learn SV on ablated val, score raw test/ood states."""

    @staticmethod
    def _subset_for_ablation_axes(
        hidden_states: np.ndarray,
        labels: np.ndarray,
        ablation_set: str,
    ) -> np.ndarray:
        if ablation_set == "all":
            subset = hidden_states
        elif ablation_set == "human":
            subset = hidden_states[labels == 0]
        elif ablation_set == "machine":
            subset = hidden_states[labels == 1]
        else:
            raise ValueError(f"Unsupported ablation_set: {ablation_set}")

        if subset.shape[0] == 0:
            raise ValueError(f"No samples available for ablation_set='{ablation_set}'.")
        return subset

    @staticmethod
    def fit_ablation_axes_by_layer(
        hidden_states: np.ndarray,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples, n_layers, d_model = hidden_states.shape
        means = np.zeros((n_layers, d_model), dtype=np.float32)
        axes = np.zeros((n_layers, d_model), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_states = hidden_states[:, layer_idx, :]
            layer_mean = layer_states.mean(axis=0).astype(np.float32)
            centered = layer_states - layer_mean[None, :]

            # PC1 of combined human+machine states for this layer.
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0].astype(np.float32)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < eps:
                axis = np.zeros_like(axis)
            else:
                axis = axis / axis_norm

            means[layer_idx] = layer_mean
            axes[layer_idx] = axis

        return means, axes

    @staticmethod
    def ablate_hidden_states_with_axes(
        hidden_states: np.ndarray,
        means: np.ndarray,
        axes: np.ndarray,
    ) -> np.ndarray:
        if hidden_states.shape[1] != axes.shape[0]:
            raise ValueError("Layer count mismatch between hidden states and ablation axes.")
        if hidden_states.shape[2] != axes.shape[1]:
            raise ValueError("Hidden dimension mismatch between hidden states and ablation axes.")
        if means.shape != axes.shape:
            raise ValueError("Ablation means shape mismatch.")

        # h_ablated = h - r(r^T h), with r normalized.
        projection_scalar = np.einsum("sld,ld->sl", hidden_states, axes)
        projection_vec = projection_scalar[:, :, None] * axes[None, :, :]
        ablated = hidden_states - projection_vec
        return ablated.astype(np.float32)

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)

        axis_fit_hidden = self._subset_for_ablation_axes(
            hidden_states=val_hidden,
            labels=val_labels,
            ablation_set=str(getattr(args, "ablation_set", "all")),
        )
        ablation_means, ablation_axes = self.fit_ablation_axes_by_layer(axis_fit_hidden)
        val_hidden_ablated = self.ablate_hidden_states_with_axes(
            hidden_states=val_hidden,
            means=ablation_means,
            axes=ablation_axes,
        )

        # Learn SV from ablated validation states only.
        steering_vec = self.raw_steering_vector(hidden_states=val_hidden_ablated, labels=val_labels)
        steering_vec = self.normalize_vectors(steering_vec)

        val_projection = self.compute_projection(
            val_hidden_states=val_hidden_ablated,
            test_hidden_states=val_hidden_ablated,
            steering_vectors=steering_vec,
            args=args,
        )

        test_hidden, test_labels = self._collect_hidden_states(data=dataset["test"], token_mode=args.token_mode)
        # Requested behavior: do NOT ablate test hidden states.
        test_projection = self.compute_projection(
            val_hidden_states=val_hidden_ablated,
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
            # Requested behavior: do NOT ablate OOD hidden states either.
            ood_projection = self.compute_projection(
                val_hidden_states=val_hidden_ablated,
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
