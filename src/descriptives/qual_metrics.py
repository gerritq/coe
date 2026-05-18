import json
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
import torch
import math
from src.inference import Inference
import skdim
from datetime import datetime

BASE_DIR = os.getenv("BASE_COE", ".")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "qual_metrics")

# ============================================================================================
# METRICS
# ============================================================================================

def mean_and_pair_layers(hidden_states: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # zip
    layer_pairs = list(zip(hidden_states[:-1], hidden_states[1:]))
    return layer_pairs

def length(hidden_states: torch.Tensor) -> list[float]:
    scores = []
    for previous_state, current_state in mean_and_pair_layers(hidden_states):
        previous_state = previous_state.float().reshape(-1)
        current_state = current_state.float().reshape(-1)
        ratio = torch.norm(current_state, p=2) / torch.norm(previous_state, p=2)
        scores.append(ratio)

    return scores

def magnitude(hidden_states: torch.Tensor, 
              normalize: bool = True) -> list[float]:
    scores = []
    first_state = hidden_states[0]
    last_state = hidden_states[-1]
    total_change = torch.norm(last_state - first_state, p=2)
    denom = torch.clamp(total_change, min=1e-12)

    for previous_state, current_state in mean_and_pair_layers(hidden_states):
        previous_state = previous_state.float().reshape(-1)
        current_state = current_state.float().reshape(-1)
        score = torch.norm(current_state - previous_state, p=2)
        if normalize:
            score = score / denom
        scores.append(score)

    return scores

def angle(hidden_states: torch.Tensor, 
          normalize: bool = True) -> list[float]:
    def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = torch.dot(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2))
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)
        return torch.acos(cosine)
    
    scores = []
    first_state = hidden_states[0].float().reshape(-1)
    last_state = hidden_states[-1].float().reshape(-1)
    total_change = angle_between(first_state, last_state)
    denom = torch.clamp(total_change, min=1e-12)

    for previous_state, current_state in mean_and_pair_layers(hidden_states):
        previous_state = previous_state.float().reshape(-1)
        current_state = current_state.float().reshape(-1)
        score = angle_between(previous_state, current_state)
        if normalize:
            score = score / denom
        scores.append(score)

    return scores


def intrinsic_dimensionality(h: torch.Tensor, eps: float = 1e-12) -> float:
    X = h.detach().float().cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)

    estimator = skdim.id.TwoNN()
    id_score = estimator.fit_transform(X)

    return float(id_score)

def anisotropy(h: torch.Tensor, eps: float = 1e-12) -> float:

    X = h.float()
    # center
    X = X - X.mean(dim=0, keepdim=True)

    # Singular values: shape (k,), where k = min(n_samples, emb_dim)
    singular_values = torch.linalg.svdvals(X)
    squared = singular_values ** 2
    return squared[0] / (squared.sum() + eps)

def effective_rank(h: torch.Tensor) -> float:
    def normalize(R):
        # GQ check non finite values in A
        total = R.numel()
        finite = torch.isfinite(R).sum().item()
        non_finite = total - finite
        share_non_finite = non_finite / total if total > 0 else 0.0
        print(f"[effective_rank - h] non_finite={non_finite}/{total} ({share_non_finite:.6%})")
        
        with torch.no_grad():
            mean = R.mean(dim=0)
            R = R - mean
            norms = torch.norm(R, p=2, dim=1, keepdim=True)
            R = R/norms

            # GQ check non finite values in A
            total = R.numel()
            finite = torch.isfinite(R).sum().item()
            non_finite = total - finite
            share_non_finite = non_finite / total if total > 0 else 0.0
            print(f"[effective_rank - R] non_finite={non_finite}/{total} ({share_non_finite:.6%})")
        return R

    def cal_cov(R):
        with torch.no_grad():
            Z = torch.nn.functional.normalize(R, dim=1)
            A = torch.matmul(Z.T, Z)/Z.shape[0]
        return A

    def cal_erank(A):
        with torch.no_grad():

            # GQ check non finite values in A
            total = A.numel()
            finite = torch.isfinite(A).sum().item()
            non_finite = total - finite
            share_non_finite = non_finite / total if total > 0 else 0.0
            print(f"[effective_rank - A] non_finite={non_finite}/{total} ({share_non_finite:.6%})")

            # fix 
            A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

            eig_val = torch.svd(A / torch.trace(A))[1] 
            entropy = - (eig_val * torch.log(eig_val)).nansum().item()
            erank = math.exp(entropy)
        return erank

    
    return cal_erank(cal_cov(normalize(h)))
    

      
def von_neumann_entropy_1(h: torch.Tensor, eps: float = 1e-12) -> float:
    # h shape: (n_samples, hidden_dim)
    k = h @ h.T
    tr = torch.trace(k)
    if float(tr) <= 0.0:
        return float("nan")
    k = k / tr

    eigvals = torch.linalg.eigvalsh(k)
    eigvals = torch.clamp(eigvals, min=eps)
    entropy = -(eigvals * torch.log(eigvals)).sum()
    return float(entropy.item())

def von_neumann_entropy_2(h: torch.Tensor):

    K = h @ h.T
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]

    mek = mek/mek.sum()
    H = -1*torch.sum(mek*torch.log(mek))
    return H

# ============================================================================================
# OTHERS
# ============================================================================================

def load_d_m4_domain_items(domain: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, "d_m4_domains", "data.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    target = domain.lower()
    items = [x for x in all_items if str(x.get("source", "")).lower() == target]
    return items


def load_main_data_items(dataset_name: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, dataset_name, "train.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return items

def sample_balanced(items: list[dict[str, Any]], n_per_label: int, seed: int) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for item in items:
        label = int(item["label"])
        if label in by_label:
            by_label[label].append(item)

    if len(by_label[0]) < n_per_label or len(by_label[1]) < n_per_label:
        raise ValueError(
            f"Not enough samples for balanced subset: "
            f"human={len(by_label[0])}, machine={len(by_label[1])}, requested={n_per_label} each."
        )

    rng = random.Random(seed)
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    sampled = by_label[0][:n_per_label] + by_label[1][:n_per_label]
    rng.shuffle(sampled)
    return sampled


def collect_hidden_states(items: list[dict[str, Any]], model_name: str) -> tuple[np.ndarray, np.ndarray]:
    inference = Inference(model_name=model_name)
    infer_args = Namespace(mode="default", token_mode="last_token")

    all_hidden = []
    labels = []

    for item in items:
        out = inference.run(item=item, args=infer_args)
        hs = out["hidden_states"]
        sample_layers = [h.detach().to(torch.float32).cpu().numpy() for h in hs[1:]] # drop embedding layer
        all_hidden.append(np.stack(sample_layers, axis=0))  # (n_layers, d_model)
        labels.append(int(out["label"]))

    x = np.stack(all_hidden, axis=0)  # (n_samples, n_layers, d_model)
    y = np.asarray(labels, dtype=np.int32)  # (n_samples,)
    return x, y

def compute_layer_metric(x: np.ndarray, y: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    # x: (n_samples, n_layers, d_model), y: (n_samples,)
    human_mask = y == 0
    machine_mask = y == 1
    if not np.any(human_mask) or not np.any(machine_mask):
        raise ValueError("Expected both human(label=0) and machine(label=1) samples.")

    n_layers = x.shape[1]
    h_vals = np.zeros(n_layers, dtype=np.float64)
    m_vals = np.zeros(n_layers, dtype=np.float64)

    if metric in ['angle', 'magnitude', 'length']:

        mean_human_sample = torch.tensor(x[human_mask, :, :].mean(axis=0), dtype=torch.float32)  # (n_layers, d_model)
        mean_machine_sample = torch.tensor(x[machine_mask, :, :].mean(axis=0), dtype=torch.float32)  # (n_layers, d_model)


        if metric == "angle":
            h_vals = angle(mean_human_sample)
            m_vals = angle(mean_machine_sample)
        elif metric == "magnitude":
            h_vals = magnitude(mean_human_sample)
            m_vals = magnitude(mean_machine_sample)
        elif metric == "length":
            h_vals = length(mean_human_sample)
            m_vals = length(mean_machine_sample)
    else:
        for layer in range(n_layers):
            print(f"Computing {metric} for layer {layer}...")
            h_layer = torch.tensor(x[human_mask, layer, :], dtype=torch.float32)
            m_layer = torch.tensor(x[machine_mask, layer, :], dtype=torch.float32)

            if metric == "von_neumann_entropy":
                h_vals[layer] = float(von_neumann_entropy_2(h_layer))
                m_vals[layer] = float(von_neumann_entropy_2(m_layer))
            elif metric == "effective_rank":
                print(f"Human samples ")
                h_vals[layer] = effective_rank(h_layer)
                print(f"Machine samples ")
                m_vals[layer] = effective_rank(m_layer)
            elif metric == "anisotropy":
                h_vals[layer] = anisotropy(h_layer)
                m_vals[layer] = anisotropy(m_layer)
            elif metric == "intrinsic_dimensionality":
                h_vals[layer] = intrinsic_dimensionality(h_layer)
                m_vals[layer] = intrinsic_dimensionality(m_layer)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return h_vals, m_vals


def save_metric_json(
    dataset_name: str,
    metric: str,
    seed: int,
    h_vals: np.ndarray,
    m_vals: np.ndarray,
    out_dir: str,
) -> str:
    out_path = os.path.join(out_dir, f"qual_{dataset_name}_{metric}_s{seed}.json")
    payload = {
        "dataset": dataset_name,
        "date": datetime.now().isoformat(),
        "metric": metric,
        "seed": int(seed),
        "layers": list(range(len(h_vals))),
        "human": [float(v) for v in h_vals],
        "machine": [float(v) for v in m_vals],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen_06b")
    parser.add_argument("--n_per_label", type=int, default=250)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["von_neumann_entropy", "effective_rank", "anisotropy", "intrinsic_dimensionality", "angle", "length", "magnitude"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n_per_label = 25 if bool(args.smoke_test) else int(args.n_per_label)
    metric = str(args.metric)
    seed = int(args.seed)

    # d_m4_domains: run one balanced plot per source domain.
    d_m4_domains = ["wikipedia", "arxiv", "reddit", "peerread"]
    for domain in d_m4_domains:
        items = load_d_m4_domain_items(domain=domain)
        sampled = sample_balanced(items=items, n_per_label=n_per_label, seed=seed)
        print(f"Sampled {n_per_label} human + {n_per_label} machine from d_m4_domains:{domain}.")

        x, y = collect_hidden_states(items=sampled, model_name=args.model)
        dataset_name = f"d_m4_domains_{domain}"
        h_vals, m_vals = compute_layer_metric(x=x, y=y, metric=metric)
        out_path = save_metric_json(
            dataset_name=dataset_name,
            metric=metric,
            seed=seed,
            h_vals=h_vals,
            m_vals=m_vals,
            out_dir=OUT_DIR,
        )
        print(f"Saved json: {out_path}")

    # drlDomain_* datasets: use test split only, no rebalancing.
    # drl_datasets = ["drlDomain_arxiv", "tsm_first", "multisocial_en", "raidModel_gpt4"]
    # for dataset_name in drl_datasets:
    #     items = load_main_data_items(dataset_name=dataset_name)
    #     sampled = sample_balanced(items=items, n_per_label=n_per_label, seed=seed)
    #     print(f"Loaded {len(items)} train items from {dataset_name}, resampled to {n_per_label} per label.")

    #     x, y = collect_hidden_states(items=sampled, model_name=args.model)
    #     h_vals, m_vals = compute_layer_metric(x=x, y=y, metric=metric)
    #     out_path = save_metric_json(
    #         dataset_name=dataset_name,
    #         metric=metric,
    #         seed=seed,
    #         h_vals=h_vals,
    #         m_vals=m_vals,
    #         out_dir=OUT_DIR,
    #     )
    #     print(f"Saved json: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
