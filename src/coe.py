import torch
import sys
from typing import Any

class Metrics:
    def __init__(self):
        pass

    def _angle_between(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = torch.dot(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2))
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)
        return torch.acos(cosine)

    def length(self, 
               states: list[torch.Tensor]
                ) -> dict[str, float]:
        
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


    def _states(self, 
                hidden_states: tuple[torch.Tensor, ...], 
                use_diff_vectors: bool
                ) -> list[torch.Tensor]:
        if not use_diff_vectors:
            return list(hidden_states)

        return [hidden_states[i + 1] - hidden_states[i] for i in range(len(hidden_states) - 1)]

    def _layer_pairs(
        self, vectors: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(vectors[:-1], vectors[1:]))

    def magnitude(
        self,
        states: list[torch.Tensor],
        normalize: bool = False,
    ) -> dict[str, float]:
        
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

    def angle(
        self,
        states: list[torch.Tensor],
        normalize: bool = False,
    ) -> dict[str, float]:
        
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

    def _zero_shot_scores(
        self,
        states: list[torch.Tensor],
        normalize: bool,
    ) -> dict[str, dict[str, Any]]:
    
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

        # this is a vector of size 28
        mag_tensor = torch.stack(magnitudes)
        ang_tensor = torch.stack(angles)
        ratio_tensor = torch.stack(ratios)

        # scores
        diff_diff_tensor = mag_tensor - ang_tensor - ratio_tensor
        diff_add_tensor = mag_tensor - ang_tensor + ratio_tensor
        feature_space_tensor = torch.sqrt(mag_tensor**2 + ang_tensor**2 + ratio_tensor**2)
        feature_avg_tensor = torch.sqrt(mag_tensor.mean()**2 + ang_tensor.mean()**2 + ratio_tensor.mean()**2)

        def pack(scores: torch.Tensor) -> dict[str, Any]:
            return {
                "scores": scores.tolist(),
                "mean": scores.mean().item(),
                "std": scores.std(unbiased=False).item(),
                "max": scores.max().item(),
            }

        return {
            "difference": pack(diff_diff_tensor),
            "addition": pack(diff_add_tensor),
            "feature_space": pack(feature_space_tensor),
            "feature_average": pack(feature_avg_tensor),
        }

    def run(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        use_diff_vectors: bool,
        normalize: bool = False,
    ) -> dict[str, Any]:
        
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
            "diff_diff_change_scores": cross_layer["difference"]["scores"],
            "diff_diff_change_mean": cross_layer["difference"]["mean"],
            "diff_diff_change_std": cross_layer["difference"]["std"],
            "diff_diff_change_max": cross_layer["difference"]["max"],
            "diff_add_change_scores": cross_layer["addition"]["scores"],
            "diff_add_change_mean": cross_layer["addition"]["mean"],
            "diff_add_change_std": cross_layer["addition"]["std"],
            "diff_add_change_max": cross_layer["addition"]["max"],
            "feature_space_change_scores": cross_layer["feature_space"]["scores"],
            "feature_space_change_mean": cross_layer["feature_space"]["mean"],
            "feature_space_change_std": cross_layer["feature_space"]["std"],
            "feature_space_change_max": cross_layer["feature_space"]["max"],
            "feature_average_change_scores": cross_layer["feature_average"]["scores"],
            "feature_average_change_mean": cross_layer["feature_average"]["mean"],
            "feature_average_change_std": cross_layer["feature_average"]["std"],
            "feature_average_change_max": cross_layer["feature_average"]["max"],
        }
