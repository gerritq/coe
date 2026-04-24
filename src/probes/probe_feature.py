import json
import os
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.inference import Inference
from src.utils import evaluation, load_dataset

BASE_DIR = os.getenv("BASE_COE")


class FeatureProbeBase:
    """Independent feature probe with notebook-style feature sets per layer."""

    def __init__(self, model_name: str) -> None:
        self.inference = Inference(model_name=model_name)

    @staticmethod
    def _safe_entropy(x: np.ndarray) -> np.ndarray:
        """Entropy from row-wise softmax with numeric guards (matches notebook intent)."""
        x_clipped = np.clip(x, -50, 50)
        x_shifted = x_clipped - np.max(x_clipped, axis=1, keepdims=True)
        e_x = np.exp(x_shifted)
        probs = e_x / (e_x.sum(axis=1, keepdims=True) + 1e-12)
        log_probs = np.log(probs + 1e-12)
        entropies = -np.sum(probs * log_probs, axis=1)
        entropies = np.nan_to_num(entropies, nan=0.0, posinf=0.0, neginf=0.0)
        return entropies.reshape(-1, 1).astype(np.float32)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    @staticmethod
    def layer_minmax(val_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            val_scores.min(axis=0, keepdims=True),
            val_scores.max(axis=0, keepdims=True),
        )

    @staticmethod
    def apply_layer_minmax(
        scores: np.ndarray,
        layer_min: np.ndarray,
        layer_max: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        denom = np.clip(layer_max - layer_min, eps, None)
        norm = (scores - layer_min) / denom
        return np.clip(norm, 0.0, 1.0)

    @staticmethod
    def avg_projection_div_std(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mean_scores = scores.mean(axis=1)
        std_scores = scores.std(axis=1)
        return mean_scores / np.clip(std_scores, eps, None)

    @staticmethod
    def last_two_thirds_layers(scores: np.ndarray) -> np.ndarray:
        n_layers = scores.shape[1]
        start_idx = n_layers // 3
        return scores[:, start_idx:]

    @staticmethod
    def _output_dir(args: Namespace) -> str:
        subdir = "probe_ood" if bool(args.ood) else "probe_id"
        out_dir = os.path.join(BASE_DIR, "output", subdir, f"sandbox_{args.mode}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _extract_feature_cache(
        self,
        data: Any,
    ) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], np.ndarray]:
        pooled_rows_by_layer: list[dict[str, list[np.ndarray]]] = []
        attn_rows_by_layer: list[list[np.ndarray]] = []
        labels: list[int] = []
        layer_count: int | None = None

        for item in tqdm(data, desc="Feature Inference", leave=False):
            text = str(item["text"])
            if not text.strip():
                continue

            inputs = self.inference.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.inference.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.inference.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states[1:]
            attentions = outputs.attentions

            if layer_count is None:
                layer_count = len(hidden_states)
                pooled_rows_by_layer = [
                    {"mean": [], "last": [], "min": [], "max": [], "concat": []}
                    for _ in range(layer_count)
                ]
                attn_rows_by_layer = [[] for _ in range(layer_count)]

            seq_len = int(inputs["attention_mask"].sum(dim=1).item())
            last_idx = max(seq_len - 1, 0)

            for layer_idx, layer_h in enumerate(hidden_states):
                mean_pooled = layer_h.mean(dim=1).detach().cpu().numpy().reshape(-1)
                last_pooled = layer_h[:, last_idx, :].detach().cpu().numpy().reshape(-1)
                min_pooled = layer_h.min(dim=1).values.detach().cpu().numpy().reshape(-1)
                max_pooled = layer_h.max(dim=1).values.detach().cpu().numpy().reshape(-1)
                concat_pooled = np.hstack([min_pooled, max_pooled, mean_pooled]).astype(np.float32)

                pooled_rows_by_layer[layer_idx]["mean"].append(mean_pooled.astype(np.float32))
                pooled_rows_by_layer[layer_idx]["last"].append(last_pooled.astype(np.float32))
                pooled_rows_by_layer[layer_idx]["min"].append(min_pooled.astype(np.float32))
                pooled_rows_by_layer[layer_idx]["max"].append(max_pooled.astype(np.float32))
                pooled_rows_by_layer[layer_idx]["concat"].append(concat_pooled)

                # attention features cache: per-head entropy from attention probabilities
                a = attentions[layer_idx].detach().cpu().numpy()[0]  # (heads, S, S)
                a = np.clip(a, 1e-12, 1.0)
                ent = -(a * np.log(a)).sum(axis=-1).mean(axis=-1).astype(np.float32)  # (heads,)
                attn_rows_by_layer[layer_idx].append(ent)

            labels.append(int(item["label"]))

        if layer_count is None:
            return [], [], np.asarray([], dtype=np.int32)

        pooled_by_layer: list[dict[str, np.ndarray]] = []
        for layer_idx in range(layer_count):
            pooled_by_layer.append(
                {
                    key: np.vstack(value).astype(np.float32)
                    for key, value in pooled_rows_by_layer[layer_idx].items()
                }
            )
        attn_by_layer = [np.vstack(value).astype(np.float32) for value in attn_rows_by_layer]
        return pooled_by_layer, attn_by_layer, np.asarray(labels, dtype=np.int32)

    def _base_feature_matrix(
        self,
        pooled_layer: dict[str, np.ndarray],
        attn_layer: np.ndarray,
        pool: str,
        feature_set: str,
    ) -> np.ndarray:
        x_hidden = pooled_layer[pool]

        if feature_set == "raw":
            return self._sanitize(x_hidden)

        if feature_set == "pca":
            # PCA is fit later on val, but starts from pooled hidden representation.
            return self._sanitize(x_hidden)

        if feature_set == "stats":
            norms = np.linalg.norm(x_hidden, axis=1, keepdims=True)
            variances = np.var(x_hidden, axis=1, keepdims=True)
            entropies = self._safe_entropy(x_hidden)
            return self._sanitize(np.hstack([norms, variances, entropies]))

        if feature_set == "attention":
            attn_mean = attn_layer.mean(axis=1, keepdims=True)
            attn_std = attn_layer.std(axis=1, keepdims=True)
            attn_max = attn_layer.max(axis=1, keepdims=True)
            return self._sanitize(np.hstack([attn_mean, attn_std, attn_max]))

        raise ValueError(f"Unsupported feature_set: {feature_set}")

    def _fit_feature_transforms(
        self,
        pooled_by_layer: list[dict[str, np.ndarray]],
        attn_by_layer: list[np.ndarray],
        pool: str,
        feature_set: str,
        pca_components: int = 50,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        val_features: list[np.ndarray] = []
        transforms: list[dict[str, Any]] = []

        for layer_idx in range(len(pooled_by_layer)):
            base_x = self._base_feature_matrix(
                pooled_layer=pooled_by_layer[layer_idx],
                attn_layer=attn_by_layer[layer_idx],
                pool=pool,
                feature_set=feature_set,
            )

            variances = np.var(base_x, axis=0)
            non_constant_mask = variances > 1e-12
            if not np.any(non_constant_mask):
                x_clean = np.zeros((base_x.shape[0], 1), dtype=np.float32)
                mask = None
            else:
                x_clean = base_x[:, non_constant_mask].astype(np.float32)
                mask = non_constant_mask

            pca_model: PCA | None = None
            if feature_set == "pca":
                max_comp = min(int(pca_components), x_clean.shape[1], max(1, x_clean.shape[0] - 1))
                if max_comp >= 1 and x_clean.shape[0] >= 2 and x_clean.shape[1] >= 1:
                    pca_model = PCA(n_components=max_comp, random_state=42)
                    x_clean = pca_model.fit_transform(x_clean).astype(np.float32)

            val_features.append(x_clean)
            transforms.append({"mask": mask, "pca": pca_model})

        return val_features, transforms

    def _apply_feature_transforms(
        self,
        pooled_by_layer: list[dict[str, np.ndarray]],
        attn_by_layer: list[np.ndarray],
        pool: str,
        feature_set: str,
        transforms: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        features: list[np.ndarray] = []
        for layer_idx in range(len(pooled_by_layer)):
            base_x = self._base_feature_matrix(
                pooled_layer=pooled_by_layer[layer_idx],
                attn_layer=attn_by_layer[layer_idx],
                pool=pool,
                feature_set=feature_set,
            )

            transform = transforms[layer_idx]
            mask = transform["mask"]
            if mask is None:
                x_clean = np.zeros((base_x.shape[0], 1), dtype=np.float32)
            else:
                x_clean = base_x[:, mask].astype(np.float32)

            pca_model: PCA | None = transform["pca"]
            if pca_model is not None:
                x_clean = pca_model.transform(x_clean).astype(np.float32)

            features.append(x_clean)

        return features

    @staticmethod
    def _fit_feature_logistic_by_layer(
        feature_by_layer: list[np.ndarray],
        labels: np.ndarray,
    ) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for x in feature_by_layer:
            if np.unique(labels).size < 2:
                models.append(
                    {
                        "kind": "linear",
                        "scaler": None,
                        "w": np.zeros((x.shape[1],), dtype=np.float32),
                        "b": 0.0,
                    }
                )
                continue

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            clf = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                penalty="l2",
                random_state=42,
            )
            clf.fit(x_scaled, labels)
            models.append({"kind": "logistic", "scaler": scaler, "model": clf})

        return models

    @staticmethod
    def _score_feature_models(
        feature_by_layer: list[np.ndarray],
        models: list[dict[str, Any]],
    ) -> np.ndarray:
        if len(feature_by_layer) != len(models):
            raise ValueError("Layer-model count mismatch for feature probes.")

        n_samples = feature_by_layer[0].shape[0] if feature_by_layer else 0
        n_layers = len(feature_by_layer)
        scores = np.zeros((n_samples, n_layers), dtype=np.float32)

        for layer_idx, x in enumerate(feature_by_layer):
            entry = models[layer_idx]
            scaler = entry.get("scaler")
            x_use = scaler.transform(x) if scaler is not None else x

            if entry["kind"] == "logistic":
                clf: LogisticRegression = entry["model"]
                layer_scores = np.asarray(clf.decision_function(x_use), dtype=np.float32).reshape(-1)
            else:
                w = np.asarray(entry["w"], dtype=np.float32)
                b = float(entry["b"])
                layer_scores = (x_use @ w + b).astype(np.float32)

            scores[:, layer_idx] = layer_scores
        return scores

    @staticmethod
    def _save_eval_metrics(
        scores: np.ndarray,
        labels: np.ndarray,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        args: Namespace,
        steering_domain: str,
        eval_domain: str,
        out_dir: str,
    ) -> None:
        metrics_per_layer: dict[str, dict[str, float]] = {}
        for layer_idx in range(scores.shape[1]):
            metrics_per_layer[f"layer_{layer_idx + 1}"] = evaluation(
                y_true=labels,
                y_predict=scores[:, layer_idx],
                y_val_true=val_labels,
                y_val_predict=val_scores[:, layer_idx],
            )

        avg_metrics = evaluation(
            y_true=labels,
            y_predict=scores.mean(axis=1),
            y_val_true=val_labels,
            y_val_predict=val_scores.mean(axis=1),
        )

        avg_div_std_metrics = evaluation(
            y_true=labels,
            y_predict=FeatureProbeBase.avg_projection_div_std(scores),
            y_val_true=val_labels,
            y_val_predict=FeatureProbeBase.avg_projection_div_std(val_scores),
        )

        scores_last_2_3 = FeatureProbeBase.last_two_thirds_layers(scores)
        val_scores_last_2_3 = FeatureProbeBase.last_two_thirds_layers(val_scores)
        avg_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=scores_last_2_3.mean(axis=1),
            y_val_true=val_labels,
            y_val_predict=val_scores_last_2_3.mean(axis=1),
        )
        avg_div_std_last_2_3_metrics = evaluation(
            y_true=labels,
            y_predict=FeatureProbeBase.avg_projection_div_std(scores_last_2_3),
            y_val_true=val_labels,
            y_val_predict=FeatureProbeBase.avg_projection_div_std(val_scores_last_2_3),
        )

        result = {
            "args": vars(args),
            "steering_domain": steering_domain,
            "eval_domain": eval_domain,
            "n_samples": int(scores.shape[0]),
            "n_layers": int(scores.shape[1]),
            "metrics_per_layer": metrics_per_layer,
            "metrics_avg_projection": avg_metrics,
            "metrics_avg_projection_div_std": avg_div_std_metrics,
            "metrics_avg_projection_last_2_3_layers": avg_last_2_3_metrics,
            "metrics_avg_projection_div_std_last_2_3_layers": avg_div_std_last_2_3_metrics,
        }

        out_path = os.path.join(
            out_dir,
            f"psm_{args.mode}_{args.model}_{steering_domain}_on_{eval_domain}_M{int(getattr(args, 'manifold', False))}_P{int(getattr(args, 'pca_components', 0))}_NS{int(getattr(args, 'normalize_scores', False))}_AS{str(getattr(args, 'ablation_set', 'all'))}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)

        pools = ["mean", "last", "min", "max", "concat"]
        feature_sets = ["raw", "pca", "stats", "attention"]
        pca_feat_components = int(getattr(args, "feature_pca_components", 50))

        val_pooled, val_attn, val_labels = self._extract_feature_cache(dataset["val"])
        test_pooled, test_attn, test_labels = self._extract_feature_cache(dataset["test"])

        ood_cache: dict[str, tuple[list[dict[str, np.ndarray]], list[np.ndarray], np.ndarray]] = {}
        for ood_dataset_name in args.ood:
            if ood_dataset_name == args.dataset:
                continue
            ood_data_args = Namespace(
                dataset=ood_dataset_name,
                prefix=bool(args.prefix),
                smoke_test=bool(args.smoke_test),
            )
            ood_dataset = load_dataset(ood_data_args)
            ood_cache[ood_dataset_name] = self._extract_feature_cache(ood_dataset["test"])

        combo_summaries: list[dict[str, object]] = []
        n_layers = 0
        d_model = 0
        n_val = 0
        n_test = 0

        for pool in pools:
            for feature_set in feature_sets:
                combo_args = Namespace(**vars(args))
                combo_args.pool = pool
                combo_args.feature_set = feature_set
                combo_args.feature_pca_components = pca_feat_components
                combo_args.mode = f"{args.mode}_pool-{pool}_feat-{feature_set}"

                val_features, transforms = self._fit_feature_transforms(
                    pooled_by_layer=val_pooled,
                    attn_by_layer=val_attn,
                    pool=pool,
                    feature_set=feature_set,
                    pca_components=pca_feat_components,
                )
                test_features = self._apply_feature_transforms(
                    pooled_by_layer=test_pooled,
                    attn_by_layer=test_attn,
                    pool=pool,
                    feature_set=feature_set,
                    transforms=transforms,
                )

                models = self._fit_feature_logistic_by_layer(feature_by_layer=val_features, labels=val_labels)
                val_scores = self._score_feature_models(feature_by_layer=val_features, models=models)
                test_scores = self._score_feature_models(feature_by_layer=test_features, models=models)

                layer_min: np.ndarray | None = None
                layer_max: np.ndarray | None = None
                if args.normalize_scores:
                    layer_min, layer_max = self.layer_minmax(val_scores)
                    val_scores = self.apply_layer_minmax(val_scores, layer_min, layer_max)
                    test_scores = self.apply_layer_minmax(test_scores, layer_min, layer_max)

                self._save_eval_metrics(
                    scores=test_scores,
                    labels=test_labels,
                    val_scores=val_scores,
                    val_labels=val_labels,
                    args=combo_args,
                    steering_domain=args.dataset,
                    eval_domain=args.dataset,
                    out_dir=out_dir,
                )

                for ood_dataset_name, (ood_pooled, ood_attn, ood_labels) in ood_cache.items():
                    ood_features = self._apply_feature_transforms(
                        pooled_by_layer=ood_pooled,
                        attn_by_layer=ood_attn,
                        pool=pool,
                        feature_set=feature_set,
                        transforms=transforms,
                    )
                    ood_scores = self._score_feature_models(feature_by_layer=ood_features, models=models)
                    if args.normalize_scores:
                        if layer_min is None or layer_max is None:
                            raise ValueError("Layer min/max stats not initialized for score normalization.")
                        ood_scores = self.apply_layer_minmax(ood_scores, layer_min, layer_max)

                    self._save_eval_metrics(
                        scores=ood_scores,
                        labels=ood_labels,
                        val_scores=val_scores,
                        val_labels=val_labels,
                        args=combo_args,
                        steering_domain=args.dataset,
                        eval_domain=ood_dataset_name,
                        out_dir=out_dir,
                    )

                n_layers = len(val_features)
                d_model = int(val_features[0].shape[1]) if val_features else 0
                n_val = int(val_scores.shape[0])
                n_test = int(test_scores.shape[0])
                combo_summaries.append({"pool": pool, "feature_set": feature_set})

        return {
            "model": args.model,
            "dataset": args.dataset,
            "val_split": "val",
            "test_split": "test",
            "n_val": n_val,
            "n_test": n_test,
            "n_layers": int(n_layers),
            "d_model": d_model,
            "ood": bool(args.ood),
            "manifold": False,
            "combinations": combo_summaries,
        }
