from argparse import Namespace

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.sv.sv_main import SVBase
from src.utils import load_dataset


class LdaSVBase(SVBase):
    """Compute steering vectors per layer using LDA coefficients."""

    @staticmethod
    def lda_steering_vector_by_layer(
        hidden_states: np.ndarray,
        labels: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        n_samples, n_layers, d_model = hidden_states.shape
        vectors = np.zeros((n_layers, d_model), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_x = hidden_states[:, layer_idx, :]
            layer_y = labels

            # Fallback if label support is degenerate for this layer.
            if np.unique(layer_y).size < 2:
                vectors[layer_idx] = np.zeros((d_model,), dtype=np.float32)
                continue

            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
            lda.fit(layer_x, layer_y)
            direction = np.asarray(lda.coef_[0], dtype=np.float32)

            norm = float(np.linalg.norm(direction))
            if norm < eps:
                vectors[layer_idx] = np.zeros((d_model,), dtype=np.float32)
            else:
                vectors[layer_idx] = direction / norm

        return vectors.astype(np.float32)

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)

        # Per-layer LDA steering vectors from validation activations.
        steering_vec = self.lda_steering_vector_by_layer(hidden_states=val_hidden, labels=val_labels)
        steering_vec = self.normalize_vectors(steering_vec)

        val_projection = self.compute_projection(
            val_hidden_states=val_hidden,
            test_hidden_states=val_hidden,
            steering_vectors=steering_vec,
            args=args,
        )

        test_hidden, test_labels = self._collect_hidden_states(data=dataset["test"], token_mode=args.token_mode)
        test_projection = self.compute_projection(
            val_hidden_states=val_hidden,
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
                val_hidden_states=val_hidden,
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
