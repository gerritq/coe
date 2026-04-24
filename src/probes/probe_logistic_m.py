from argparse import Namespace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.probes.probe_main import ProbeBase
from src.utils import load_dataset


class LogisticManifoldProbeBase(ProbeBase):
    """Per-layer PCA manifold from val + per-layer logistic probes in low-dim space."""

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
            basis = vt[:k].T.astype(np.float32)

            means[layer_idx] = layer_mean
            components[layer_idx] = basis
            projected[:, layer_idx, :] = (centered @ basis).astype(np.float32)

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
    def fit_logistic_models_by_layer(
        hidden_states: np.ndarray,
        labels: np.ndarray,
    ) -> list[dict[str, Any]]:
        _n_samples, n_layers, _k = hidden_states.shape
        models: list[dict[str, Any]] = []

        for layer_idx in range(n_layers):
            x = hidden_states[:, layer_idx, :]
            y = labels

            if np.unique(y).size < 2:
                models.append({"kind": "linear", "w": np.zeros((x.shape[1],), dtype=np.float32), "b": 0.0})
                continue

            logistic = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
            logistic.fit(x, y)
            models.append({"kind": "logistic", "model": logistic})

        return models

    @staticmethod
    def score_with_models(
        hidden_states: np.ndarray,
        layer_models: list[dict[str, Any]],
    ) -> np.ndarray:
        n_samples, n_layers, _k = hidden_states.shape
        if len(layer_models) != n_layers:
            raise ValueError("Layer-model count mismatch.")

        scores = np.zeros((n_samples, n_layers), dtype=np.float32)
        for layer_idx in range(n_layers):
            x = hidden_states[:, layer_idx, :]
            entry = layer_models[layer_idx]

            if entry["kind"] == "logistic":
                model: LogisticRegression = entry["model"]
                layer_scores = np.asarray(model.decision_function(x), dtype=np.float32).reshape(-1)
            else:
                w = np.asarray(entry["w"], dtype=np.float32)
                b = float(entry["b"])
                layer_scores = (x @ w + b).astype(np.float32)

            scores[:, layer_idx] = layer_scores

        return scores

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)
        val_hidden_proj, manifold_means, manifold_components = self.fit_pca_manifold_by_layer(
            hidden_states=val_hidden,
            n_components=args.pca_components,
        )
        projection_state = {"means": manifold_means, "components": manifold_components}

        logistic_models = self.fit_logistic_models_by_layer(hidden_states=val_hidden_proj, labels=val_labels)
        val_scores = self.score_with_models(hidden_states=val_hidden_proj, layer_models=logistic_models)

        test_hidden, test_labels = self._collect_hidden_states(data=dataset["test"], token_mode=args.token_mode)
        test_hidden_proj = self.project_hidden_states_with_manifold(
            hidden_states=test_hidden,
            manifold_means=projection_state["means"],
            manifold_components=projection_state["components"],
        )
        test_scores = self.score_with_models(hidden_states=test_hidden_proj, layer_models=logistic_models)

        layer_min: np.ndarray | None = None
        layer_max: np.ndarray | None = None
        if args.normalize_scores:
            layer_min, layer_max = self.layer_minmax(val_scores)
            val_scores = self.apply_layer_minmax(val_scores, layer_min, layer_max)
            test_scores = self.apply_layer_minmax(test_scores, layer_min, layer_max)

        self._save_projection_metrics(
            projections=test_scores,
            labels=test_labels,
            val_projections=val_scores,
            val_labels=val_labels,
            args=args,
            steering_domain=args.dataset,
            eval_domain=args.dataset,
            out_dir=out_dir,
        )
        self._plot_test_projections(
            projections=test_scores,
            labels=test_labels,
            val_projections=val_scores,
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
            ood_hidden_proj = self.project_hidden_states_with_manifold(
                hidden_states=ood_hidden,
                manifold_means=projection_state["means"],
                manifold_components=projection_state["components"],
            )
            ood_scores = self.score_with_models(hidden_states=ood_hidden_proj, layer_models=logistic_models)

            if args.normalize_scores:
                if layer_min is None or layer_max is None:
                    raise ValueError("Layer min/max stats not initialized for score normalization.")
                ood_scores = self.apply_layer_minmax(ood_scores, layer_min, layer_max)

            self._plot_test_projections(
                projections=ood_scores,
                labels=ood_labels,
                val_projections=val_scores,
                val_labels=val_labels,
                args=args,
                eval_domain=ood_dataset_name,
                out_dir=out_dir,
            )
            self._save_projection_metrics(
                projections=ood_scores,
                labels=ood_labels,
                val_projections=val_scores,
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
            "n_layers": int(val_scores.shape[1]),
            "d_model": int(val_hidden_proj.shape[2]),
            "ood": bool(args.ood),
            "manifold": True,
        }
