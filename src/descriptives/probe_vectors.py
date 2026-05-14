import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.inference import Inference
from src.utils import load_dataset, return_args


BASE_DIR = os.getenv("BASE_COE", ".")
OUT_DIR = os.path.join(BASE_DIR, "output", "desc")

DATASETS = [
    "drlDomain_arxiv",
    "drlDomain_writing_prompt",
    "drlDomain_yelp_review",
    "drlDomain_xsum",
    "multisocial_en",
    "multisocial_de",
    "multisocial_ru",
    "multisocial_zh",
    "tsm_first",
    "tsm_extend",
    "tsm_sums",
    "tsm_tst",
    "raidModel_cohere_chat",
    "raidModel_gpt4",
    "raidModel_llama_chat",
    "raidModel_mistral_chat",
]


def collect_middle_layer_states(items: list[dict[str, Any]], inference: Inference) -> tuple[np.ndarray, np.ndarray]:
    infer_args = Namespace(mode="default", token_mode="last_token")
    x_all = []
    y_all = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hs = out["hidden_states"]
        mid_idx = len(hs) // 2  # exact middle layer representation
        vec = hs[mid_idx].detach().to(torch.float32).cpu().numpy()
        x_all.append(vec)
        y_all.append(int(out["label"]))

    x = np.stack(x_all, axis=0)  # (n_samples, d_model)
    y = np.asarray(y_all, dtype=np.int32)  # (n_samples,)
    return x, y


def train_probe_vector(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x, y)
    coef = clf.coef_[0].astype(np.float64)
    norm = float(np.linalg.norm(coef))
    if norm > 0.0:
        coef = coef / norm
    return coef


def run(args: Namespace) -> None:
    inference = Inference(model_name=args.model)

    dataset_states: dict[str, dict[str, Any]] = {}
    all_x = []

    for dataset_name in DATASETS:
        ds = load_dataset(
            Namespace(
                dataset=dataset_name,
                smoke_test=False,
                training_size=None,
                seed=args.seed,
            )
        )
        test_items = [dict(x) for x in ds["test"]]
        x, y = collect_middle_layer_states(test_items, inference=inference)
        dataset_states[dataset_name] = {"x": x, "y": y}
        all_x.append(x)

    n_total = sum(v["x"].shape[0] for v in dataset_states.values())
    print(f"Loaded {len(DATASETS)} test sets with total {n_total} items.")

    if args.mode == "pca":
        x_concat = np.concatenate(all_x, axis=0)
        pca = PCA(n_components=args.components, random_state=args.seed)
        pca.fit(x_concat)
        for dataset_name in DATASETS:
            x = dataset_states[dataset_name]["x"]
            x_low = pca.transform(x)
            x_recon = pca.inverse_transform(x_low)
            dataset_states[dataset_name]["x"] = x_recon.astype(np.float32)

    vectors: dict[str, dict[str, Any]] = {}
    for dataset_name in DATASETS:
        x = dataset_states[dataset_name]["x"]
        y = dataset_states[dataset_name]["y"]
        coef = train_probe_vector(x=x, y=y, seed=args.seed)
        vectors[dataset_name] = {
            "n_test": int(len(y)),
            "n_human": int(np.sum(y == 0)),
            "n_machine": int(np.sum(y == 1)),
            "probe": coef.tolist(),
            "probe_l2_norm": float(np.linalg.norm(coef)),
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = f"probe_vectors_{args.mode}_{args.model}_seed{args.seed}.json"
    out_path = os.path.join(OUT_DIR, out_name)

    out = {
        "args": return_args(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_datasets": len(DATASETS),
        "n_total_test_items": int(n_total),
        "datasets": DATASETS,
        "vectors": vectors,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--mode", type=str, choices=["default", "pca"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--components", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
