import json
import math
import os
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from src.coe.classifier import ScoreGMM, ScoreLogistic, ScoreMLP
from src.inference import Inference
from src.utils import evaluation, load_dataset

BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "output", "coe")
os.makedirs(OUT_DIR, exist_ok=True)


class CoEBASE:
    def __init__(self) -> None:
        pass

    def _angle_between(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = torch.dot(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2))
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)
        return torch.acos(cosine)

    def _states(self, hidden_states: tuple[torch.Tensor, ...], use_diff_vectors: bool) -> list[torch.Tensor]:
        if not use_diff_vectors:
            return list(hidden_states)
        return [hidden_states[i + 1] - hidden_states[i] for i in range(len(hidden_states) - 1)]

    @staticmethod
    def _layer_pairs(vectors: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(vectors[:-1], vectors[1:]))

    def length(self, states: list[torch.Tensor]) -> dict[str, float]:
        scores = []
        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)
            ratio = torch.norm(current_state, p=2) / torch.norm(previous_state, p=2)
            scores.append(ratio)

        return {
            "scores": [score.item() for score in scores],
            "mean": torch.tensor(scores).mean().item(),
            "std": torch.tensor(scores).std(unbiased=False).item(),
        }

    def magnitude(self, states: list[torch.Tensor], normalize: bool = False) -> dict[str, float]:
        scores = []
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = torch.norm(last_state - first_state, p=2)
        denom = torch.clamp(total_change, min=1e-12)

        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)
            score = torch.norm(current_state - previous_state, p=2)
            if normalize:
                score = score / denom
            scores.append(score)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def angle(self, states: list[torch.Tensor], normalize: bool = False) -> dict[str, float]:
        scores = []
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = self._angle_between(first_state, last_state)
        denom = torch.clamp(total_change, min=1e-12)

        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)
            score = self._angle_between(previous_state, current_state)
            if normalize:
                score = score / denom
            scores.append(score)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def _zero_shot_scores(self, states: list[torch.Tensor], normalize: bool) -> dict[str, dict[str, Any]]:
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_magnitude = torch.norm(last_state - first_state, p=2)
        total_angle = self._angle_between(first_state, last_state)
        denom_magnitude = torch.clamp(total_magnitude, min=1e-12)
        denom_angle = torch.clamp(total_angle, min=1e-12)

        magnitudes = []
        angles = []
        ratios = []
        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)
            mag = torch.norm(current_state - previous_state, p=2)
            ang = self._angle_between(previous_state, current_state)
            ratio = torch.norm(current_state, p=2) / torch.norm(previous_state, p=2)
            if normalize:
                mag = mag / denom_magnitude
                ang = ang / denom_angle
            magnitudes.append(mag)
            angles.append(ang)
            ratios.append(ratio)

        mag_tensor = torch.stack(magnitudes)
        ang_tensor = torch.stack(angles)
        ratio_tensor = torch.stack(ratios)

        diff_tensor = mag_tensor - ang_tensor
        diff_diff_tensor = mag_tensor - ang_tensor - ratio_tensor
        diff_add_tensor = mag_tensor - ang_tensor + ratio_tensor
        fs_layer_wise_tensor = torch.sqrt(mag_tensor**2 + ang_tensor**2 + ratio_tensor**2)
        fs_avg_tensor = torch.sqrt(mag_tensor.mean()**2 + ang_tensor.mean()**2 + ratio_tensor.mean()**2)

        def pack(scores: torch.Tensor) -> dict[str, Any]:
            return {
                "scores": scores.tolist(),
                "mean": scores.mean().item(),
                "std": scores.std(unbiased=False).item(),
                "max": scores.max().item(),
            }

        return {
            "diff": pack(diff_tensor),
            "diff_diff": pack(diff_diff_tensor),
            "diff_add": pack(diff_add_tensor),
            "fs_layer_wise": pack(fs_layer_wise_tensor),
            "fs_avg": pack(fs_avg_tensor),
        }

    def compute_metrics(self, hidden_states: tuple[torch.Tensor, ...], use_diff_vectors: bool, normalize: bool = False) -> dict[str, Any]:
        states = self._states(hidden_states=hidden_states, use_diff_vectors=use_diff_vectors)
        magnitude_scores = self.magnitude(states, normalize=normalize)
        angle_scores = self.angle(states, normalize=normalize)
        length_scores = self.length(states)
        cross_layer = self._zero_shot_scores(states, normalize=normalize)

        return {
            "magnitude_change_scores": magnitude_scores["scores"],
            "magnitude_change_mean": magnitude_scores["mean"],
            "magnitude_change_std": magnitude_scores["std"],
            "angle_change_scores": angle_scores["scores"],
            "angle_change_mean": angle_scores["mean"],
            "angle_change_std": angle_scores["std"],
            "length_change_scores": length_scores["scores"],
            "length_change_mean": length_scores["mean"],
            "length_change_std": length_scores["std"],
            "diff_change_scores": cross_layer["diff"]["scores"],
            "diff_change_mean": cross_layer["diff"]["mean"],
            "diff_change_std": cross_layer["diff"]["std"],
            "diff_change_max": cross_layer["diff"]["max"],
            "diff_diff_change_scores": cross_layer["diff_diff"]["scores"],
            "diff_diff_change_mean": cross_layer["diff_diff"]["mean"],
            "diff_diff_change_std": cross_layer["diff_diff"]["std"],
            "diff_diff_change_max": cross_layer["diff_diff"]["max"],
            "diff_add_change_scores": cross_layer["diff_add"]["scores"],
            "diff_add_change_mean": cross_layer["diff_add"]["mean"],
            "diff_add_change_std": cross_layer["diff_add"]["std"],
            "diff_add_change_max": cross_layer["diff_add"]["max"],
            "fs_layer_wise_change_scores": cross_layer["fs_layer_wise"]["scores"],
            "fs_layer_wise_change_mean": cross_layer["fs_layer_wise"]["mean"],
            "fs_layer_wise_change_std": cross_layer["fs_layer_wise"]["std"],
            "fs_layer_wise_change_max": cross_layer["fs_layer_wise"]["max"],
            "fs_avg_change_scores": cross_layer["fs_avg"]["scores"],
            "fs_avg_change_mean": cross_layer["fs_avg"]["mean"],
            "fs_avg_change_std": cross_layer["fs_avg"]["std"],
            "fs_avg_change_max": cross_layer["fs_avg"]["max"],
        }

    @staticmethod
    def _label_map(args: Namespace) -> tuple[dict[int, str], dict[int, str]]:
        if args.dataset in ["counterfact"]:
            label_names = {0: "correct", 1: "incorrect"}
        else:
            label_names = {0: "human", 1: "machine"}
        label_colors = {0: "tab:blue", 1: "tab:orange"}
        return label_names, label_colors

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(var)

    def plot_scores_by_label(self, out: list[dict], save_path: str | None = None) -> str:
        metric_specs = [
            ("angle_change_mean", "Angle Mean"),
            ("angle_change_std", "Angle Std"),
            ("magnitude_change_mean", "Magnitude Mean"),
            ("magnitude_change_std", "Magnitude Std"),
            ("length_change_mean", "Length Mean"),
            ("length_change_std", "Length Std"),
        ]

        label_names, label_colors = self._label_map(self.args)
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        labels = sorted({item["label"] for item in out})

        for axis, (metric_key, title) in zip(axes, metric_specs):
            for label in labels:
                values = [item[metric_key] for item in out if item["label"] == label]
                if not values:
                    continue
                label_name = label_names.get(label, str(label))
                label_color = label_colors.get(label, "tab:gray")
                mean_value = sum(values) / len(values)
                axis.hist(values, bins=20, alpha=0.5, color=label_color, label=label_name)
                axis.axvline(mean_value, color=label_color, linestyle="--", linewidth=1, label=f"{label_name} mean")
            axis.set_title(title)
            axis.set_xlabel("Score")
            axis.set_ylabel("Count")
            axis.legend()

        fig.suptitle(f"Angle, Magnitude, and Length Scores by Label {self.args.title_info}")
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(OUT_DIR, f"coe_dist_{self.args.suffix}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_layer_profiles(self, out: list[dict], save_path: str | None = None) -> str:
        label_names, label_colors = self._label_map(self.args)
        labels = sorted({item["label"] for item in out})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        for label in labels:
            label_out = [item for item in out if item["label"] == label]
            angle_scores = [item["angle_change_scores"] for item in label_out]
            magnitude_scores = [item["magnitude_change_scores"] for item in label_out]
            length_scores = [item["length_change_scores"] for item in label_out]

            min_layers = min(len(scores) for scores in angle_scores + magnitude_scores + length_scores)
            angle_scores = [scores[:min_layers] for scores in angle_scores]
            magnitude_scores = [scores[:min_layers] for scores in magnitude_scores]
            length_scores = [scores[:min_layers] for scores in length_scores]

            angle_means, angle_stds = [], []
            magnitude_means, magnitude_stds = [], []
            length_means, length_stds = [], []

            for layer_index in range(min_layers):
                angle_vals = [scores[layer_index] for scores in angle_scores]
                magnitude_vals = [scores[layer_index] for scores in magnitude_scores]
                length_vals = [scores[layer_index] for scores in length_scores]
                angle_mean, angle_std = self._mean_std(angle_vals)
                magnitude_mean, magnitude_std = self._mean_std(magnitude_vals)
                length_mean, length_std = self._mean_std(length_vals)
                angle_means.append(angle_mean)
                angle_stds.append(angle_std)
                magnitude_means.append(magnitude_mean)
                magnitude_stds.append(magnitude_std)
                length_means.append(length_mean)
                length_stds.append(length_std)

            layers = list(range(1, min_layers + 1))
            label_name = label_names.get(label, str(label))
            label_color = label_colors.get(label, "tab:gray")
            avg_angle = sum(angle_means) / len(angle_means)
            axes[0].plot(layers, angle_means, color=label_color, label=f"{label_name} mean")
            axes[0].axhline(avg_angle, linestyle="--", color=label_color, linewidth=1, label=f"{label_name} mean (avg)")
            axes[0].fill_between(layers, [m - s for m, s in zip(angle_means, angle_stds)], [m + s for m, s in zip(angle_means, angle_stds)], color=label_color, alpha=0.2, label=f"{label_name} std")

            avg_magnitude = sum(magnitude_means) / len(magnitude_means)
            axes[1].plot(layers, magnitude_means, color=label_color, label=f"{label_name} mean")
            axes[1].axhline(avg_magnitude, linestyle="--", color=label_color, linewidth=1, label=f"{label_name} mean (avg)")
            axes[1].fill_between(layers, [m - s for m, s in zip(magnitude_means, magnitude_stds)], [m + s for m, s in zip(magnitude_means, magnitude_stds)], color=label_color, alpha=0.2, label=f"{label_name} std")

            avg_length = sum(length_means) / len(length_means)
            axes[2].plot(layers, length_means, color=label_color, label=f"{label_name} mean")
            axes[2].axhline(avg_length, linestyle="--", color=label_color, linewidth=1, label=f"{label_name} mean (avg)")
            axes[2].fill_between(layers, [m - s for m, s in zip(length_means, length_stds)], [m + s for m, s in zip(length_means, length_stds)], color=label_color, alpha=0.2, label=f"{label_name} std")

        axes[0].set_title("Angle by Layer")
        axes[0].set_ylabel("Score")
        axes[0].legend()
        axes[1].set_title("Magnitude by Layer")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[2].set_title("Length by Layer")
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Score")
        axes[2].legend()
        fig.suptitle(f"Trajectory | {self.args.title_info}")
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(OUT_DIR, f"trajectory_{self.args.suffix}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def pair_plot(self, out: list[dict], save_path: str | None = None) -> str:
        df = pd.DataFrame(
            [
                {
                    "label": item["label"],
                    "Angle": item["angle_change_mean"],
                    "Magnitude": item["magnitude_change_mean"],
                    "Length": item["length_change_mean"],
                }
                for item in out
            ]
        )
        if self.args.dataset in ["counterfact"]:
            df["label"] = df["label"].map({0: "correct", 1: "incorrect"}).fillna(df["label"])
        else:
            df["label"] = df["label"].map({0: "human", 1: "machine"}).fillna(df["label"])
        grid = sns.pairplot(
            df,
            vars=["Angle", "Magnitude", "Length"],
            hue="label",
            kind="kde",
            diag_kind="kde",
            corner=True,
            plot_kws={"levels": 8, "fill": False},
        )
        grid.fig.suptitle(f"Pair Plot (Means) | {self.args.title_info}", y=1.02)
        if save_path is None:
            save_path = os.path.join(OUT_DIR, f"pp_{self.args.suffix}")
        grid.fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(grid.fig)
        return save_path


class ScoreEvaluator:
    SCORE_KEYS = [
        "magnitude_change_mean",
        "angle_change_mean",
        "length_change_mean",
        "diff_change_mean",
        "diff_diff_change_mean",
        "diff_add_change_mean",
        "fs_layer_wise_change_mean",
        "fs_avg_change_mean",
    ]

    @staticmethod
    def run(out: list[dict[str, Any]], args: Namespace) -> dict[str, Any]:
        labels = [int(item["label"]) for item in out]

        metrics: dict[str, Any] = {}
        for key in ScoreEvaluator.SCORE_KEYS:
            values: list[float] = []
            usable_labels: list[int] = []
            for item in out:
                value = item.get(key)
                if value is None:
                    continue
                values.append(float(value))
                usable_labels.append(int(item["label"]))

            if not values:
                continue

            metrics[key] = evaluation(y_true=usable_labels, y_predict=values)

        payload = {"args": vars(args), "n": len(labels), "metrics": metrics}
        out_path = os.path.join(OUT_DIR, f"eval_{args.suffix.replace('.pdf', '.json')}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload


class COEAnalyzer(CoEBASE):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.inference = Inference(model_name=args.model)

    def _infer_hidden(self, item: dict[str, Any]) -> dict[str, Any]:
        return self.inference.run(item=item, args=Namespace(mode=self.args.token_mode))

    def _postprocess_hidden_states(
        self,
        hidden_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return hidden_records

    def _metrics_from_hidden(self, hidden_record: dict[str, Any]) -> dict[str, Any]:
        out = {
            "model_id": hidden_record["model_id"],
            "text": hidden_record["text"],
            "label": hidden_record["label"],
        }
        metrics_dict = self.compute_metrics(
            hidden_states=hidden_record["hidden_states"],
            use_diff_vectors=self.args.diff_vectors,
            normalize=self.args.normalize,
        )
        out.update(metrics_dict)
        return out

    def run(self) -> dict[str, Any]:
        data = load_dataset(args=self.args)
        data = list(data["test"])
        self.args.n = len(data)

        suffix = (
            f"{self.args.model}_{self.args.dataset}_TM{self.args.token_mode}_CM{self.args.mode}"
            f"_DV{int(self.args.diff_vectors)}_PF{int(self.args.prefix)}_NO{int(self.args.normalize)}"
            f"{'_ST' if self.args.smoke_test else ''}.pdf"
        )
        title_info = (
            f"{self.args.model} | {self.args.dataset} | N={len(data)} | TokenMode {self.args.token_mode}"
            f" | CoeMode {self.args.mode} | DV {int(self.args.diff_vectors)} | Pre {int(self.args.prefix)}"
            f" | Norm {int(self.args.normalize)}"
        )
        self.args.suffix = suffix
        self.args.title_info = title_info

        hidden_records: list[dict[str, Any]] = []
        for item in tqdm(data, desc="Inference ..."):
            hidden_records.append(self._infer_hidden(item))

        hidden_records = self._postprocess_hidden_states(hidden_records)
        out = [self._metrics_from_hidden(record) for record in hidden_records]

        if self.args.save_viz:
            self.plot_scores_by_label(out=out)
            self.pair_plot(out=out)
            if self.args.token_mode != "horizontal":
                self.plot_layer_profiles(out=out)

        eval_payload = ScoreEvaluator.run(out=out, args=self.args)

        classifier_payload = None
        if self.args.classifier:
            gmm = ScoreGMM()
            logreg = ScoreLogistic()
            mlp = ScoreMLP()
            classifier_payload = {
                "gmm": gmm.run(out=out, suffix=self.args.suffix, args=self.args),
                "logistic": logreg.run(out=out, suffix=self.args.suffix, args=self.args),
                "mlp": mlp.run(out=out, suffix=self.args.suffix, args=self.args),
            }

        return {
            "n": len(out),
            "eval_saved": eval_payload is not None,
            "classifier_ran": classifier_payload is not None,
        }
