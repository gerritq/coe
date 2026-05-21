import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.inference import Inference
from src.utils import load_dataset


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
OUT_PATH = os.path.join(OUT_DIR, "f_parallel.pdf")

DATASETS = ["multisocial_en", "multisocial_de"]


def collect_mid_layer_representations(
    items: list[dict],
    inference: Inference,
) -> tuple[np.ndarray, np.ndarray]:
    infer_args = Namespace(mode="default", token_mode="last_token")
    x_all = []
    y_all = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hs = out["hidden_states"]
        mid_idx = len(hs) // 2
        vec = hs[mid_idx].detach().to(torch.float32).cpu().numpy()
        x_all.append(vec)
        y_all.append(int(out["label"]))

    x = np.stack(x_all, axis=0)  # (n_samples, d_model)
    y = np.asarray(y_all, dtype=np.int32)  # (n_samples,)
    return x, y


def load_dataset_states(args: Namespace) -> dict[str, dict[str, np.ndarray]]:
    inference = Inference(model_name=args.model)
    out: dict[str, dict[str, np.ndarray]] = {}

    for dataset_name in DATASETS:
        ds = load_dataset(
            Namespace(
                dataset=dataset_name,
                smoke_test=args.smoke_test,
                training_size=None,
                seed=args.seed,
            )
        )
        test_items = [dict(x) for x in ds["test"]]
        x, y = collect_mid_layer_representations(test_items, inference=inference)
        out[dataset_name] = {"x": x, "y": y}
        print(f"Loaded {dataset_name}: n={len(y)}")
    return out


def build_combined_pca_projection(
    dataset_states: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    x_concat = np.concatenate([dataset_states[name]["x"] for name in DATASETS], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(x_concat)

    projected: dict[str, dict[str, np.ndarray]] = {}
    for dataset_name in DATASETS:
        projected[dataset_name] = {
            "x2d": pca.transform(dataset_states[dataset_name]["x"]),
            "y": dataset_states[dataset_name]["y"],
        }
    return projected


def plot_parallel(projected: dict[str, dict[str, np.ndarray]], out_path: str) -> None:
    colors = {
        ("multisocial_en", 0): "#1f77b4",  # en human
        ("multisocial_en", 1): "#ff7f0e",  # en machine
        ("multisocial_de", 0): "#2ca02c",  # de human
        ("multisocial_de", 1): "#d62728",  # de machine
    }
    labels = {
        ("multisocial_en", 0): "en Human",
        ("multisocial_en", 1): "en Machine",
        ("multisocial_de", 0): "de Human",
        ("multisocial_de", 1): "de Machine",
    }

    plt.figure(figsize=(8.0, 6.0))
    for dataset_name in DATASETS:
        x2d = projected[dataset_name]["x2d"]
        y = projected[dataset_name]["y"]
        for lab in [0, 1]:
            mask = y == lab
            if not np.any(mask):
                continue
            plt.scatter(
                x2d[mask, 0],
                x2d[mask, 1],
                s=12,
                alpha=0.7,
                c=colors[(dataset_name, lab)],
                label=labels[(dataset_name, lab)],
                edgecolors="none",
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: Namespace) -> None:
    dataset_states = load_dataset_states(args=args)
    projected = build_combined_pca_projection(dataset_states=dataset_states, seed=args.seed)
    plot_parallel(projected=projected, out_path=OUT_PATH)


def main() -> None:
    args = parse_args()
    args.smoke_test = bool(args.smoke_test)
    run(args=args)


if __name__ == "__main__":
    main()
