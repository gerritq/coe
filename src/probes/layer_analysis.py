from argparse import Namespace
import argparse
from typing import Any
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.inference import Inference
from src.utils import load_dataset, optimal_thresholds, metrics, OOD, return_args
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from datetime import datetime

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
                              probing_vectors_by_layer: np.ndarray,):
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
        out_path = os.path.join(
            out_dir,
            f"heatmap_{self.args.dataset}_layers.png",
        )
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {out_path}")

    def run(self, args: Namespace) -> None:
        source_data = load_dataset(args=args)
        train = self._collect_model_states(source_data["train"])

        train_out = self._train_linear_probe(x_train=train["hidden_x"], 
                                            y_train=train["y"])
        
        self.single_domain_heatmap(probing_vectors_by_layer=train_out["probing_vectors_by_layer"])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--folder", type=str, default="sandbox")
    parser.add_argument("--training_size", type=int, default=-1)
    args =  parser.parse_args()
    if args.training_size == -1:
        args.training_size = None

    args.mode= "default"
    args.C= 1.0
    args.training_size= 1000

    return args


if __name__ == "__main__":
    args = main()
    analyzer = LinearProbing(args=args)
    analyzer.run(args)
