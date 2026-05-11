from argparse import Namespace
import argparse
from typing import Any
import os
import copy
from itertools import combinations
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.inference import Inference
from src.utils import load_dataset, OOD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.getenv("BASE_COE")

class LinearProbing:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.out_dir = os.path.join(BASE_DIR, "output", "probe", args.folder)
        os.makedirs(self.out_dir, exist_ok=True)

        self.inference = Inference(self.args.model)

    def _collect_model_states(self, 
                              items: list[dict]
                              ) -> dict[str, Any]:
        
        all_hidden_states = []
        labels = []
        
        for item in items:
            out = self.inference.run(item=item, 
                                     args=self.args)
            labels.append(item["label"])
        
            # A HIDDEN STATES
            hidden_states = out["hidden_states"] # tuple of len layer
            hs_per_layer = []
            for l in hidden_states:
                # l: (d_model) 
                hs_per_layer.append(l.detach().to(torch.float32).cpu().numpy())
            
            # append: layer x d_model
            all_hidden_states.append(np.stack(hs_per_layer, axis=0))


        # STACK HS
        x = np.stack(all_hidden_states, axis=0)  # (n_samples, n_layers, d_model)
        x = np.transpose(x, (1, 0, 2))  # (n_layers, n_samples, d_model)
        y = np.asarray(labels, dtype=np.int32)   # (n_samples,)

        return {
            "hidden_x": x,
            "y": y,
        }

    def _train_linear_probe(self,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          ) -> dict[str, Any]:

        probing_vectors_by_layer = []
    
        for layer in range(x_train.shape[0]):

            # get data and scale
            x_layer_train = x_train[layer]  # (n_samples, d_model)

            scaler = StandardScaler()
            x_layer_train_scaled = scaler.fit_transform(x_layer_train)

            clf_binary = LogisticRegression(max_iter=2000, 
                                                random_state=42,
                                                C=self.args.C)
            clf_binary.fit(x_layer_train_scaled, y_train)

            # otbain the probing vector

            probe_vector = clf_binary.coef_[0]

            probing_vectors_by_layer.append(probe_vector)
        probing_vectors_by_layer = np.stack(probing_vectors_by_layer, axis=0)  # (n_layers, d_model)
        return {
            "probing_vectors_by_layer": probing_vectors_by_layer,
        }

    def single_domain_heatmap(self,
                              probing_vectors_by_layer: np.ndarray,
                              dataset_name: str):
        # compute cosine similarity across layer probe vectors
        vectors = probing_vectors_by_layer.astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        vectors = vectors / norms
        sim = vectors @ vectors.T  # (n_layers, n_layers)

        # produce heatmap
        n_layers = sim.shape[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ticks = np.arange(n_layers)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        # ax.set_title("Cross-layer cosine similarity of probing vectors")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
        fig.tight_layout()

        out_dir = os.path.join(BASE_DIR, "output", "probe", self.args.folder)
        os.makedirs(out_dir, exist_ok=True)

        if self.args.smoke_test:
            suffix = "_smoke_test"
        out_path = os.path.join(
            out_dir,
            f"heatmap_{dataset_name}_layers{suffix}.png",
        )
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {out_path}")


    def cross_domain_similarity_plot(self, probing_vectors: dict[str, np.ndarray]):
    
        datasets = sorted(probing_vectors.keys())
        if len(datasets) < 2:
            raise ValueError("Need at least two datasets for cross-domain similarity.")

        # Build pairwise layer-wise cosine similarities.
        pair_labels = []
        pair_sims = []
        for ds_a, ds_b in combinations(datasets, 2):
            vec_a = np.asarray(probing_vectors[ds_a], dtype=np.float64)  # (n_layers, d_model)
            vec_b = np.asarray(probing_vectors[ds_b], dtype=np.float64)  # (n_layers, d_model)

            if vec_a.shape != vec_b.shape:
                raise ValueError(
                    f"Shape mismatch for pair ({ds_a}, {ds_b}): "
                    f"{vec_a.shape} vs {vec_b.shape}"
                )

            # Normalize layer vectors for cosine similarity.
            norm_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(vec_b, axis=1, keepdims=True)
            vec_a = vec_a / np.clip(norm_a, 1e-12, None)
            vec_b = vec_b / np.clip(norm_b, 1e-12, None)

            # Layer-wise cosine: dot(v_a[layer], v_b[layer]).
            sim = np.sum(vec_a * vec_b, axis=1)  # (n_layers,)
            sim = np.clip(sim, -1.0, 1.0)

            pair_labels.append(f"{ds_a} vs {ds_b}")
            pair_sims.append(sim)

        pair_sims_arr = np.stack(pair_sims, axis=0)  # (n_pairs, n_layers)
        mean_sim = np.mean(pair_sims_arr, axis=0)  # (n_layers,)

        n_layers = pair_sims_arr.shape[1]
        x = np.arange(n_layers)

        fig, ax = plt.subplots(figsize=(11, 6))
        for label, sim in zip(pair_labels, pair_sims_arr):
            ax.plot(x, sim, linewidth=1.4, alpha=0.75, label=label)

        ax.plot(
            x,
            mean_sim,
            color="black",
            linewidth=2.8,
            linestyle="--",
            label="Mean across pairs",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])

        out_dir = os.path.join(BASE_DIR, "output", "probe", self.args.folder)
        os.makedirs(out_dir, exist_ok=True)

        if self.args.smoke_test:
            suffix = "_smoke_test"
        out_path = os.path.join(out_dir, f"cross_domain_similarity_{self.args.benchmark}{suffix}.png")
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def run(self, args: Namespace) -> None:

        datasets = OOD[args.benchmark]

        if args.smoke_test:
            datasets = datasets[:2]


        # collect hidden states for all ds
        data_probing_vectors = {}
        for dataset in datasets:
            ds_args = copy.copy(args)
            ds_args.dataset = dataset
            ds = load_dataset(args=ds_args)["train"]
            ds_hidden_states = self._collect_model_states(items=ds)

            train_out = self._train_linear_probe(x_train=ds_hidden_states["hidden_x"], 
                                                y_train=ds_hidden_states["y"])
            data_probing_vectors[dataset] = train_out["probing_vectors_by_layer"]

        
        # Single domain heatmap
        for dataset, probing_vectors in data_probing_vectors.items():
            self.single_domain_heatmap(
                probing_vectors_by_layer=probing_vectors,
                dataset_name=dataset,
            )

            if self.args.smoke_test:
                break

        # Cross-domain similarity plot
        self.cross_domain_similarity_plot(probing_vectors=data_probing_vectors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--folder", type=str, default="sandbox")
    parser.add_argument("--training_size", type=int, default=-1)
    args =  parser.parse_args()
    if args.training_size == -1:
        args.training_size = None

    if args.smoke_test:
        args.model = "qwen_06b"
    args.mode= "default"
    args.C= 1.0
    args.training_size= 1000

    return args


if __name__ == "__main__":
    args = main()
    analyzer = LinearProbing(args=args)
    analyzer.run(args)
