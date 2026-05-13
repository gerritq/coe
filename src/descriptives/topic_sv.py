import json
import os
from argparse import Namespace
from typing import Any

import numpy as np
from tqdm import tqdm

from src.inference import Inference
from src.sv.sv_main import SVBase
from src.utils import load_dataset

BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "output", "descriptives", "sandbox_sv_topic")
os.makedirs(OUT_DIR, exist_ok=True)


class TopicSVAnalyzer:
    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    def _collect_hidden_states(
        self,
        data: Any,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_hidden_states: list[np.ndarray] = []
        labels: list[int] = []

        for item in tqdm(data, desc="Topic-SV Inference", leave=False):
            out = self.inference.run(item, args=Namespace(mode=mode))
            hidden_states = out.get("hidden_states")
            if hidden_states is None:
                continue

            sample_layers: list[np.ndarray] = []
            for layer_tensor in hidden_states:
                layer_vec = layer_tensor.detach().float().cpu().numpy().reshape(-1)
                sample_layers.append(layer_vec.astype(np.float32))

            all_hidden_states.append(np.stack(sample_layers, axis=0))
            labels.append(int(item["label"]))

        if not all_hidden_states:
            raise ValueError("No hidden states collected for topic-vs-steering analysis.")

        return np.stack(all_hidden_states, axis=0), np.array(labels, dtype=np.int32)

    @staticmethod
    def topic_vector_pc1_per_layer(hidden_states: np.ndarray) -> np.ndarray:
        n_samples, n_layers, d_model = hidden_states.shape
        topic_vectors = np.zeros((n_layers, d_model), dtype=np.float32)

        for layer_idx in range(n_layers):
            layer_states = hidden_states[:, layer_idx, :]
            centered = layer_states - layer_states.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            topic_vectors[layer_idx] = vt[0].astype(np.float32)

        return topic_vectors

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
        return float(np.dot(a, b) / denom)

    def run(self, args: Namespace) -> dict[str, Any]:
        dataset = load_dataset(
            Namespace(
                dataset=args.data,
                prefix=bool(args.prefix),
                smoke_test=bool(args.smoke_test),
            )
        )

        val_data = dataset["val"]
        if getattr(args, "n", 0) > 0:
            val_data = val_data.select(range(min(len(val_data), int(args.n))))

        val_hidden, val_labels = self._collect_hidden_states(data=val_data, mode=args.mode)

        raw_sv = SVBase.raw_steering_vector(hidden_states=val_hidden, labels=val_labels)  # n_layers x d_model
        topic_vec = self.topic_vector_pc1_per_layer(val_hidden)  # n_layers x d_model

        dot_per_layer = np.einsum("ld,ld->l", raw_sv, topic_vec).astype(np.float32)
        cos_per_layer = np.array(
            [self._cosine(raw_sv[i], topic_vec[i]) for i in range(raw_sv.shape[0])],
            dtype=np.float32,
        )

        result: dict[str, Any] = {
            "model": args.model,
            "dataset": args.data,
            "split": "val",
            "mode": args.mode,
            "n_samples": int(val_hidden.shape[0]),
            "n_layers": int(val_hidden.shape[1]),
            "dot_product_per_layer": dot_per_layer.tolist(),
            "cosine_per_layer": cos_per_layer.tolist(),
            "dot_product_mean": float(dot_per_layer.mean()),
            "cosine_mean": float(cos_per_layer.mean()),
            "dot_product_last_layer": float(dot_per_layer[-1]),
            "cosine_last_layer": float(cos_per_layer[-1]),
        }

        out_path = os.path.join(
            OUT_DIR,
            f"sv_topic_{args.model}_{args.data}_{args.mode}_val.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        result["output_json"] = out_path
        return result
