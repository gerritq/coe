import argparse
import json
import os
import sys
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BASE_DIR = os.getenv("BASE_COE")

from src.inference import Inference
from src.utils import load_dataset, evaluation

OUT_DIR = os.path.join(BASE_DIR, "output", "steering", "sandbox")
os.makedirs(OUT_DIR, exist_ok=True)

def raw_steering_vector(hidden_states: np.ndarray,
                        labels: np.ndarray,
                        ) -> np.ndarray:
    """
    hidden states n_samples x n_layers x d_model
    """

    machine_mask = labels == 1
    human_mask = labels == 0

    machine_mean = hidden_states[machine_mask].mean(axis=0)  # n_layers x d_model
    human_mean = hidden_states[human_mask].mean(axis=0)  # n_layers x d_model
    vectors = machine_mean - human_mean # n_layers x d_model

    return vectors.astype(np.float32)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, eps, None)).astype(np.float32)


def fit_pca_manifold_by_layer(hidden_states: np.ndarray, n_components: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_layers, d_model = hidden_states.shape
    k = max(1, min(n_components, n_samples, d_model))

    # means needed to center new data before projection
    means = np.zeros((n_layers, d_model), dtype=np.float32)
    # components = vt from svd, shape n_layers x d_model x k
    components = np.zeros((n_layers, d_model, k), dtype=np.float32)
    # projected 
    projected = np.zeros((n_samples, n_layers, k), dtype=np.float32)

    for layer_idx in range(n_layers):
        layer_states = hidden_states[:, layer_idx, :]  # n_samples x d_model
        # get mean across samples and subtract each sample 
        centered = layer_states - layer_states.mean(axis=0, keepdims=True)

        #  perform svd
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        # vt projects from d_model to k
        layer_basis = vt[:k].T  # d_model x k

        means[layer_idx] = layer_states.mean(axis=0).astype(np.float32)
        components[layer_idx] = layer_basis.astype(np.float32)
        projected[:, layer_idx, :] = (centered @ layer_basis).astype(np.float32)

    return projected.astype(np.float32), means.astype(np.float32), components.astype(np.float32)


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


def project_steering_vectors_with_manifold(
    steering_vectors: np.ndarray,
    manifold_components: np.ndarray,
) -> np.ndarray:
    if steering_vectors.shape[0] != manifold_components.shape[0]:
        raise ValueError("Layer count mismatch between steering vectors and manifold.")
    if steering_vectors.shape[1] != manifold_components.shape[1]:
        raise ValueError("Hidden dimension mismatch between steering vectors and manifold.")

    return np.einsum("ld,ldk->lk", steering_vectors, manifold_components).astype(np.float32)


def steering_projection(
    test_hidden_states: np.ndarray,
    steering_vectors: np.ndarray,
) -> np.ndarray:
    """
    compute projection score 
    
    """

    # Dot product per sample and per layer: n_samples x n_layers
    return np.einsum("sld,ld->sl", test_hidden_states, steering_vectors)


class SteeringAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def _collect_hidden_states(self, 
                               data: Any,
                               mode: str,
                                ) -> tuple[np.ndarray, np.ndarray]:

        all_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(data, desc="Inference", leave=False):
            out = self.inference.run(item, args=Namespace(mode=mode))
            hidden_states = out["hidden_states"]
            
            sample_layers = []
            for layer_tensor in hidden_states:
                layer_vec = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vec.astype(np.float32))

            all_hidden_states.append(np.stack(sample_layers, axis=0)) # n_layers x d_model
            labels.append(int(item["label"]))

        return (
            np.stack(all_hidden_states, axis=0),  # n_samples x n_layers x d_model
            np.array(labels, dtype=np.int32),
        )

    @staticmethod
    def _save_projection_metrics(
        projections: np.ndarray,
        labels: np.ndarray,
        val_projections: np.ndarray,
        val_labels: np.ndarray,
        args: Namespace,
        steering_domain: str,
        eval_domain: str,
    ) -> None:
        per_layer: dict[str, dict[str, float]] = {}
        for layer_idx in range(projections.shape[1]):
            per_layer[f"layer_{layer_idx + 1}"] = evaluation(
                y_true=labels,
                y_predict=projections[:, layer_idx],
                y_val_true=val_labels,
                y_val_predict=val_projections[:, layer_idx],
            )

        result = {
            "args": vars(args),
            "steering_domain": steering_domain,
            "eval_domain": eval_domain,
            "n_samples": int(projections.shape[0]),
            "n_layers": int(projections.shape[1]),
            "metrics_per_layer": per_layer,
        }

        out_path = os.path.join(
            OUT_DIR,
            f"svp_scores_{args.model}_{steering_domain}_on_{eval_domain}_M{int(args.manifold)}_P{int(args.pca_components)}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _plot_test_projections(
        projections: np.ndarray,
        labels: np.ndarray,
        val_projections: np.ndarray,
        val_labels: np.ndarray,
        model: str,
        steering_domain: str,
        eval_domain: str,
        manifold: bool,
        pca_components: int,
    ) -> None:
        fig = plt.figure(figsize=(20, 12))
        grid = fig.add_gridspec(3, 5, height_ratios=[1.5, 1, 1])
        axis = fig.add_subplot(grid[0, :])
        layers = np.arange(1, projections.shape[1] + 1)
        label_names = {0: "human", 1: "machine"}
        colors = {0: "tab:blue", 1: "tab:orange"}

        for label in [0, 1]:
            mask = labels == label
            if not np.any(mask):
                continue

            class_proj = projections[mask]
            for sample_idx, sample_proj in enumerate(class_proj):
                axis.plot(
                    layers,
                    sample_proj,
                    color=colors[label],
                    alpha=0.2,
                    linewidth=1.0,
                    label=label_names[label] if sample_idx == 0 else None,
                )

            class_mean = class_proj.mean(axis=0)
            axis.plot(
                layers,
                class_mean,
                color=colors[label],
                linewidth=2.5,
            )

        # Layer-wise AUROC using projection score at each layer.
        layer_aurocs: list[float] = []
        if np.unique(labels).size == 2:
            for layer_idx in range(projections.shape[1]):
                auc = roc_auc_score(labels, projections[:, layer_idx])
                layer_aurocs.append(float(auc))
        else:
            layer_aurocs = [float("nan")] * projections.shape[1]

        auc_axis = axis.twinx()
        auc_axis.plot(
            layers,
            layer_aurocs,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="AUROC",
        )
        auc_axis.set_ylabel("AUROC")
        auc_axis.set_ylim(0.0, 1.0)

        last_layer_auc = layer_aurocs[-1] if layer_aurocs else float("nan")
        last_layer_auc_text = f"{last_layer_auc:.3f}" if not np.isnan(last_layer_auc) else "NA"

        axis.set_title(
            f"Steering Projection by Layer | {model} | steering={steering_domain} | eval={eval_domain} | M{int(manifold)} | P{int(pca_components)} | last-layer AUROC={last_layer_auc_text}"
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Projection Score")
        axis.grid(alpha=0.25)
        mean_auc = np.nanmean(layer_aurocs)
        if not np.isnan(mean_auc):
            axis.text(
                0.01,
                0.98,
                f"Mean AUROC: {mean_auc:.3f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
            )

        lines_1, labels_1 = axis.get_legend_handles_labels()
        lines_2, labels_2 = auc_axis.get_legend_handles_labels()
        axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        selected_layers = np.unique(np.linspace(0, projections.shape[1] - 1, num=10, dtype=int)).tolist()
        dist_axes = [fig.add_subplot(grid[1 + (i // 5), i % 5]) for i in range(10)]

        for panel_idx, ax in enumerate(dist_axes):
            if panel_idx >= len(selected_layers):
                ax.axis("off")
                continue

            layer_idx = selected_layers[panel_idx]
            layer_scores = projections[:, layer_idx]
            human_scores = layer_scores[labels == 0]
            machine_scores = layer_scores[labels == 1]

            if len(human_scores) == 0 or len(machine_scores) == 0:
                ax.axis("off")
                continue

            if len(human_scores) > 1:
                sns.kdeplot(
                    x=human_scores,
                    fill=True,
                    alpha=0.35,
                    linewidth=1.5,
                    color=colors[0],
                    label="human",
                    ax=ax,
                )
            if len(machine_scores) > 1:
                sns.kdeplot(
                    x=machine_scores,
                    fill=True,
                    alpha=0.35,
                    linewidth=1.5,
                    color=colors[1],
                    label="machine",
                    ax=ax,
                )

            metrics = evaluation(
                y_true=labels,
                y_predict=layer_scores,
                y_val_true=val_labels,
                y_val_predict=val_projections[:, layer_idx],
            )
            auroc = metrics.get("auroc", float("nan"))
            f1 = metrics.get("f1", float("nan"))
            acc = metrics.get("acc", float("nan"))

            auroc_txt = f"{auroc:.2f}" if np.isfinite(auroc) else "NA"
            f1_txt = f"{f1:.2f}" if np.isfinite(f1) else "NA"
            acc_txt = f"{acc:.2f}" if np.isfinite(acc) else "NA"

            ax.set_title(f"L{layer_idx + 1} | AUC {auroc_txt} | F1 {f1_txt} | ACC {acc_txt}", fontsize=9)
            ax.set_xlabel("Projection Score")
            ax.set_ylabel("Density")
            if panel_idx == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2, axis="y")

        out_path = os.path.join(
            OUT_DIR,
            f"psp_{model}_{steering_domain}_on_{eval_domain}_M{int(manifold)}_P{int(pca_components)}.png",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self, args: Namespace) -> None:
        
        # load data
        dataset = load_dataset(args)

        # use the val set for finding SVs
        val_hidden, val_labels = self._collect_hidden_states(data=dataset['val'], mode=args.mode)

        # get raw SVs
        steering_vec = raw_steering_vector(hidden_states=val_hidden, labels=val_labels)

        if args.manifold:
            val_hidden_proj, manifold_means, manifold_components = fit_pca_manifold_by_layer(
                hidden_states=val_hidden,
                n_components=args.pca_components,
            )
            steering_vec = project_steering_vectors_with_manifold(
                steering_vectors=steering_vec,
                manifold_components=manifold_components,
            )
            val_hidden_for_projection = val_hidden_proj
        else:
            val_hidden_for_projection = val_hidden

        # normalize in the space where the projection is computed
        steering_vec = normalize_vectors(steering_vec)

        val_projection = steering_projection(
            test_hidden_states=val_hidden_for_projection,
            steering_vectors=steering_vec,
        )

        # get test set hidden states and project in the same space as val
        test_hidden, test_labels = self._collect_hidden_states(data=dataset['test'], mode=args.mode)
        if args.manifold:
            test_hidden_for_projection = project_hidden_states_with_manifold(
                hidden_states=test_hidden,
                manifold_means=manifold_means,
                manifold_components=manifold_components,
            )
        else:
            test_hidden_for_projection = test_hidden

        test_projection = steering_projection(
            test_hidden_states=test_hidden_for_projection,
            steering_vectors=steering_vec,
        )

        # plots
        self._save_projection_metrics(
            projections=test_projection,
            labels=test_labels,
            val_projections=val_projection,
            val_labels=val_labels,
            args=args,
            steering_domain=args.dataset,
            eval_domain=args.dataset,
        )

        # eval
        self._plot_test_projections(
            projections=test_projection,
            labels=test_labels,
            val_projections=val_projection,
            val_labels=val_labels,
            model=args.model,
            steering_domain=args.dataset,
            eval_domain=args.dataset,
            manifold=args.manifold,
            pca_components=args.pca_components,
        )

        for ood_dataset_name in args.ood_set:
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
                mode=args.mode,
            )

            if args.manifold:
                ood_hidden_for_projection = project_hidden_states_with_manifold(
                    hidden_states=ood_hidden,
                    manifold_means=manifold_means,
                    manifold_components=manifold_components,
                )
            else:
                ood_hidden_for_projection = ood_hidden

            ood_projection = steering_projection(
                test_hidden_states=ood_hidden_for_projection,
                steering_vectors=steering_vec,
            )
            
            
            self._plot_test_projections(
                projections=ood_projection,
                labels=ood_labels,
                val_projections=val_projection,
                val_labels=val_labels,
                model=args.model,
                steering_domain=args.dataset,
                eval_domain=ood_dataset_name,
                manifold=args.manifold,
                pca_components=args.pca_components,
            )
            
            self._save_projection_metrics(
                projections=ood_projection,
                labels=ood_labels,
                val_projections=val_projection,
                val_labels=val_labels,
                args=args,
                steering_domain=args.dataset,
                eval_domain=ood_dataset_name,
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


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="last_token")
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--ood", type=int, default=0)
    parser.add_argument("--ood_set", type=str, default="")
    parser.add_argument("--manifold", type=int, default=0)
    parser.add_argument("--pca_components", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    if args.manifold not in (0, 1):
        raise ValueError("manifold must be 0 or 1")
    if args.mode != "last_token":
        raise ValueError("This script expects --mode last_token.")

    if args.ood_set.strip():
        args.ood_set = args.ood_set.split(" ")
        

    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.manifold = bool(args.manifold)

    analyzer = SteeringAnalyzer(model_name=args.model)
    analyzer.run(args)


if __name__ == "__main__":
    main()
