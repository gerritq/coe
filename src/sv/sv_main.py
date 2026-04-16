import json
import os
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_dataset

BASE_DIR = os.getenv("BASE_COE")

class SVBase:
    """
    Base steering analyzer. Subclasses can override projection hooks to change
    the representation used for steering/projection.
    """

    def __init__(self, 
                 model_name: str
                 ) -> None:
        # inference to obtain hidden stagtes
        self.inference = Inference(model_name=model_name)

    @staticmethod
    def raw_steering_vector(hidden_states: np.ndarray, 
                            labels: np.ndarray
                            ) -> np.ndarray:
        
        """"
        Obtain steerting vector as the difference in means 
        @hidden_states: (n_samples, n_layers, d_model)
        @labels: (n_samples,)
        """
        machine_mask = labels == 1
        human_mask = labels == 0

        machine_mean = hidden_states[machine_mask].mean(axis=0) # n_layers x d_model
        human_mean = hidden_states[human_mask].mean(axis=0) # n_layers x d_model
        vectors = machine_mean - human_mean
        return vectors.astype(np.float32)

    @staticmethod
    def normalize_vectors(vectors: np.ndarray, 
                          eps: float = 1e-12
                          ) -> np.ndarray:
        """takes a set of vectors and normalizes them to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, eps, None)).astype(np.float32)

    @staticmethod
    def avg_projection_div_std(projections: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        avg_projection = projections.mean(axis=1)
        std_projection = projections.std(axis=1)
        return avg_projection / np.clip(std_projection, eps, None)

    @staticmethod
    def last_two_thirds_layers(projections: np.ndarray) -> np.ndarray:
        n_layers = projections.shape[1]
        start_idx = n_layers // 3
        return projections[:, start_idx:]

    @staticmethod
    def layer_minmax(
        val_projections: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            val_projections.min(axis=0, keepdims=True),
            val_projections.max(axis=0, keepdims=True),
        )

    @staticmethod
    def apply_layer_minmax(
        projections: np.ndarray,
        layer_min: np.ndarray,
        layer_max: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        denom = np.clip(layer_max - layer_min, eps, None)
        proj_norm = (projections - layer_min) / denom
        return np.clip(proj_norm, 0.0, 1.0)

    def _collect_hidden_states(self, data: Any, token_mode: str) -> tuple[np.ndarray, np.ndarray]:
        all_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(data, desc="Inference", leave=False):
            out = self.inference.run(item, args=Namespace(mode=token_mode))
            hidden_states = out["hidden_states"]

            sample_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                layer_vec = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vec.astype(np.float32))

            all_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        return (
            np.stack(all_hidden_states, axis=0),
            np.array(labels, dtype=np.int32),
        )

    def fit_projection_space(
        self,
        val_hidden: np.ndarray,
        steering_vec: np.ndarray,
        args: Namespace,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Default mode: keep hidden states and steering vectors unchanged."""
        return val_hidden, steering_vec, None

    def transform_hidden_states(
        self,
        hidden_states: np.ndarray,
        projection_state: Any,
        args: Namespace,
    ) -> np.ndarray:
        return hidden_states

    def compute_projection(
        self,
        val_hidden_states: np.ndarray,
        test_hidden_states: np.ndarray,
        steering_vectors: np.ndarray,
        args: Namespace,
    ) -> np.ndarray:
        return np.einsum("sld,ld->sl", test_hidden_states, steering_vectors)

    @staticmethod
    def _output_dir(args: Namespace) -> str:
        subdir = "sv_ood" if bool(args.ood) else "sv_id"
        out_dir = os.path.join(BASE_DIR, "output", subdir, f"sandbox_{args.mode}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _save_projection_metrics(
        projections: np.ndarray,
        labels: np.ndarray,
        val_projections: np.ndarray,
        val_labels: np.ndarray,
        args: Namespace,
        steering_domain: str,
        eval_domain: str,
        out_dir: str,
    ) -> None:
        per_layer: dict[str, dict[str, float]] = {}
        for layer_idx in range(projections.shape[1]):
            per_layer[f"layer_{layer_idx + 1}"] = evaluation(
                y_true=labels,
                y_predict=projections[:, layer_idx],
                y_val_true=val_labels,
                y_val_predict=val_projections[:, layer_idx],
            )

        avg_projection = projections.mean(axis=1)
        avg_val_projection = val_projections.mean(axis=1)
        avg_projection_metrics = evaluation(
            y_true=labels,
            y_predict=avg_projection,
            y_val_true=val_labels,
            y_val_predict=avg_val_projection,
        )

        avg_div_std_projection = SVBase.avg_projection_div_std(projections)
        avg_div_std_val_projection = SVBase.avg_projection_div_std(val_projections)
        avg_div_std_projection_metrics = evaluation(
            y_true=labels,
            y_predict=avg_div_std_projection,
            y_val_true=val_labels,
            y_val_predict=avg_div_std_val_projection,
        )

        projections_last_2_3 = SVBase.last_two_thirds_layers(projections)
        val_projections_last_2_3 = SVBase.last_two_thirds_layers(val_projections)
        avg_projection_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=projections_last_2_3.mean(axis=1),
            y_val_true=val_labels,
            y_val_predict=val_projections_last_2_3.mean(axis=1),
        )
        avg_div_std_projection_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=SVBase.avg_projection_div_std(projections_last_2_3),
            y_val_true=val_labels,
            y_val_predict=SVBase.avg_projection_div_std(val_projections_last_2_3),
        )

        result = {
            "args": vars(args),
            "steering_domain": steering_domain,
            "eval_domain": eval_domain,
            "n_samples": int(projections.shape[0]),
            "n_layers": int(projections.shape[1]),
            "metrics_per_layer": per_layer,
            "metrics_avg_projection": avg_projection_metrics,
            "metrics_avg_projection_div_std": avg_div_std_projection_metrics,
            "metrics_avg_projection_last_2_3_layers": avg_projection_last_2_3_metrics,
            "metrics_avg_projection_div_std_last_2_3_layers": avg_div_std_projection_last_2_3_metrics,
        }
        ablation_set = str(getattr(args, "ablation_set", "all"))

        out_path = os.path.join(
            out_dir,
            f"psm_{args.mode}_{args.model}_{steering_domain}_on_{eval_domain}_M{int(args.manifold)}_P{int(args.pca_components)}_NS{int(args.normalize_scores)}_AS{ablation_set}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _plot_test_projections(
        projections: np.ndarray,
        labels: np.ndarray,
        val_projections: np.ndarray,
        val_labels: np.ndarray,
        args: Namespace,
        out_dir: str,
        eval_domain: str | None = None,
    ) -> None:
        
        model = args.model
        steering_domain = args.dataset
        eval_domain = eval_domain or args.dataset
        manifold = args.manifold
        pca_components = args.pca_components
        ablation_set = str(getattr(args, "ablation_set", "all"))

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

        layer_aurocs: list[float] = []
        if np.unique(labels).size == 2:
            for layer_idx in range(projections.shape[1]):
                layer_aurocs.append(float(roc_auc_score(labels, projections[:, layer_idx])))
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

        avg_projection_metrics = evaluation(
            y_true=labels,
            y_predict=projections.mean(axis=1),
            y_val_true=val_labels,
            y_val_predict=val_projections.mean(axis=1),
        )
        avg_projection_auc = avg_projection_metrics.get("auroc", float("nan"))
        avg_projection_auc_text = f"{avg_projection_auc:.3f}" if np.isfinite(avg_projection_auc) else "NA"

        avg_div_std_projection_metrics = evaluation(
            y_true=labels,
            y_predict=SVBase.avg_projection_div_std(projections),
            y_val_true=val_labels,
            y_val_predict=SVBase.avg_projection_div_std(val_projections),
        )
        avg_div_std_projection_auc = avg_div_std_projection_metrics.get("auroc", float("nan"))
        avg_div_std_projection_auc_text = (
            f"{avg_div_std_projection_auc:.3f}" if np.isfinite(avg_div_std_projection_auc) else "NA"
        )

        projections_last_2_3 = SVBase.last_two_thirds_layers(projections)
        val_projections_last_2_3 = SVBase.last_two_thirds_layers(val_projections)
        avg_projection_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=projections_last_2_3.mean(axis=1),
            y_val_true=val_labels,
            y_val_predict=val_projections_last_2_3.mean(axis=1),
        )
        avg_projection_last_2_3_auc = avg_projection_last_2_3_metrics.get("auroc", float("nan"))
        avg_projection_last_2_3_auc_text = (
            f"{avg_projection_last_2_3_auc:.3f}" if np.isfinite(avg_projection_last_2_3_auc) else "NA"
        )

        avg_div_std_projection_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=SVBase.avg_projection_div_std(projections_last_2_3),
            y_val_true=val_labels,
            y_val_predict=SVBase.avg_projection_div_std(val_projections_last_2_3),
        )
        avg_div_std_projection_last_2_3_auc = avg_div_std_projection_last_2_3_metrics.get(
            "auroc", float("nan")
        )
        avg_div_std_projection_last_2_3_auc_text = (
            f"{avg_div_std_projection_last_2_3_auc:.3f}"
            if np.isfinite(avg_div_std_projection_last_2_3_auc)
            else "NA"
        )

        axis.set_title(
            f"Steering Projection by Layer | {model} | steering={steering_domain} | eval={eval_domain} | M{int(manifold)} | P{int(pca_components)} | NS{int(args.normalize_scores)} | AS{ablation_set} | last-layer AUROC={last_layer_auc_text} | avg-proj AUROC={avg_projection_auc_text} | avg/std AUROC={avg_div_std_projection_auc_text} | avg-proj(2/3) AUROC={avg_projection_last_2_3_auc_text} | avg/std(2/3) AUROC={avg_div_std_projection_last_2_3_auc_text}"
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
            out_dir,
            f"psp_{args.mode}_{model}_{steering_domain}_on_{eval_domain}_M{int(manifold)}_P{int(pca_components)}_NS{int(args.normalize_scores)}_AS{ablation_set}.png",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self, args: Namespace) -> dict[str, Any]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        val_hidden, val_labels = self._collect_hidden_states(data=dataset["val"], token_mode=args.token_mode)
        steering_vec = self.raw_steering_vector(hidden_states=val_hidden, labels=val_labels)

        val_hidden_proj, steering_vec_proj, projection_state = self.fit_projection_space(
            val_hidden=val_hidden,
            steering_vec=steering_vec,
            args=args,
        )
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
