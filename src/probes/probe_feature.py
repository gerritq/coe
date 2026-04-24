from argparse import Namespace
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.probes.probe_main import ProbeBase
from src.utils import load_dataset


class FeatureProbeBase(ProbeBase):
    """Feature-based probe using pooling features per layer + logistic classifiers."""

    @staticmethod
    def _safe_entropy(attn_scores: np.ndarray) -> np.ndarray:
        probs = np.clip(attn_scores, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        entropy = np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return entropy.astype(np.float32)

    @staticmethod
    def _clean_features(features: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        if x.ndim != 2 or x.shape[1] == 0:
            return x
        variances = np.var(x, axis=0)
        non_constant = variances > 1e-12
        if np.any(non_constant):
            return x[:, non_constant]
        return np.zeros((x.shape[0], 1), dtype=np.float32)

    def _collect_features_by_layer(
        self,
        data: Any,
        pool: str = "mean",
        use_attn: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        feature_by_layer: list[list[np.ndarray]] = []
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
                    output_attentions=use_attn,
                    use_cache=False,
                )

            # hidden_states[0] is embedding output; keep transformer layers only.
            hidden_states = outputs.hidden_states[1:]
            attentions = outputs.attentions if use_attn else None

            if layer_count is None:
                layer_count = len(hidden_states)
                feature_by_layer = [[] for _ in range(layer_count)]

            seq_len = int(inputs["attention_mask"].sum(dim=1).item())
            last_idx = max(seq_len - 1, 0)

            for layer_idx, layer_h in enumerate(hidden_states):
                mean_pooled = layer_h.mean(dim=1).detach().cpu().numpy().reshape(-1)
                last_pooled = layer_h[:, last_idx, :].detach().cpu().numpy().reshape(-1)
                min_pooled = layer_h.min(dim=1).values.detach().cpu().numpy().reshape(-1)
                max_pooled = layer_h.max(dim=1).values.detach().cpu().numpy().reshape(-1)

                if pool == "mean":
                    feats = mean_pooled
                elif pool == "last":
                    feats = last_pooled
                elif pool == "min":
                    feats = min_pooled
                elif pool == "max":
                    feats = max_pooled
                elif pool == "concat":
                    feats = np.concatenate([min_pooled, max_pooled, mean_pooled], axis=0)
                else:
                    raise ValueError(f"Unsupported pool method: {pool}")

                if use_attn and attentions is not None:
                    # (1, heads, S, S) -> per-head entropy averaged across query positions.
                    attn_layer = attentions[layer_idx].detach().cpu().numpy()[0]
                    ent = self._safe_entropy(attn_layer).mean(axis=-1).reshape(-1)
                    feats = np.concatenate([feats, ent], axis=0)

                feature_by_layer[layer_idx].append(feats.astype(np.float32))

            labels.append(int(item["label"]))

        if layer_count is None:
            return [], np.asarray([], dtype=np.int32)

        stacked = [self._clean_features(np.vstack(layer_feats)) for layer_feats in feature_by_layer]
        return stacked, np.asarray(labels, dtype=np.int32)

    @staticmethod
    def _fit_feature_logistic_by_layer(
        feature_by_layer: list[np.ndarray],
        labels: np.ndarray,
    ) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for x in feature_by_layer:
            if np.unique(labels).size < 2:
                models.append({"kind": "linear", "scaler": None, "w": np.zeros((x.shape[1],), dtype=np.float32), "b": 0.0})
                continue

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            try:
                clf = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    penalty="l2",
                    random_state=42,
                )
                clf.fit(x_scaled, labels)
                models.append({"kind": "logistic", "scaler": scaler, "model": clf})
                continue
            except Exception:
                pass

            machine_mean = x_scaled[labels == 1].mean(axis=0)
            human_mean = x_scaled[labels == 0].mean(axis=0)
            w = (machine_mean - human_mean).astype(np.float32)
            b = float(-0.5 * np.dot(machine_mean + human_mean, w))
            models.append({"kind": "linear", "scaler": scaler, "w": w, "b": b})
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

    def run(self, args: Namespace) -> dict[str, object]:
        dataset = load_dataset(args)
        out_dir = self._output_dir(args)
        pools = ["mean", "last", "min", "max", "concat"]
        attn_modes = [False, True]
        combo_summaries: list[dict[str, object]] = []
        n_layers = 0
        d_model = 0
        n_val = 0
        n_test = 0

        for pool in pools:
            for use_attn in attn_modes:
                combo_args = Namespace(**vars(args))
                combo_args.pool = pool
                combo_args.use_attn = bool(use_attn)
                combo_args.mode = f"{args.mode}_pool-{pool}_attn-{int(use_attn)}"

                val_features, val_labels = self._collect_features_by_layer(
                    data=dataset["val"],
                    pool=pool,
                    use_attn=use_attn,
                )
                models = self._fit_feature_logistic_by_layer(feature_by_layer=val_features, labels=val_labels)
                val_scores = self._score_feature_models(feature_by_layer=val_features, models=models)

                test_features, test_labels = self._collect_features_by_layer(
                    data=dataset["test"],
                    pool=pool,
                    use_attn=use_attn,
                )
                test_scores = self._score_feature_models(feature_by_layer=test_features, models=models)

                layer_min: np.ndarray | None = None
                layer_max: np.ndarray | None = None
                if args.normalize_scores:
                    layer_min, layer_max = self.layer_minmax(val_scores)
                    val_scores = self.apply_layer_minmax(val_scores, layer_min, layer_max)
                    test_scores = self.apply_layer_minmax(test_scores, layer_min, layer_max)

                # Metrics only (no plots for this probe).
                self._save_projection_metrics(
                    projections=test_scores,
                    labels=test_labels,
                    val_projections=val_scores,
                    val_labels=val_labels,
                    args=combo_args,
                    steering_domain=args.dataset,
                    eval_domain=args.dataset,
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
                    ood_features, ood_labels = self._collect_features_by_layer(
                        data=ood_dataset["test"],
                        pool=pool,
                        use_attn=use_attn,
                    )
                    ood_scores = self._score_feature_models(feature_by_layer=ood_features, models=models)
                    if args.normalize_scores:
                        if layer_min is None or layer_max is None:
                            raise ValueError("Layer min/max stats not initialized for score normalization.")
                        ood_scores = self.apply_layer_minmax(ood_scores, layer_min, layer_max)

                    self._save_projection_metrics(
                        projections=ood_scores,
                        labels=ood_labels,
                        val_projections=val_scores,
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
                combo_summaries.append({"pool": pool, "use_attn": bool(use_attn)})

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
