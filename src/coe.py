from typing import Any

import torch


class Metrics:
    def __init__(self):
        pass

    def _angle_between(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cosine = torch.dot(a, b) / (
            torch.norm(a, p=2).clamp_min(1e-12) * torch.norm(b, p=2).clamp_min(1e-12)
        )
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)
        return torch.acos(cosine)

    def _transformer_states(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        states = hidden_states[1:]
        if len(states) < 2:
            raise ValueError(
                "Need at least two transformer hidden states after excluding the embedding layer."
            )

        return states

    def _layer_pairs(
        self, hidden_states: tuple[torch.Tensor, ...]
        ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        states = self._transformer_states(hidden_states)
        return list(zip(states[:-1], states[1:]))

    def magnitude(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states = self._transformer_states(hidden_states)
        scores = []
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = torch.norm(last_state - first_state, p=2).clamp_min(1e-12)

        for previous_state, current_state in self._layer_pairs(hidden_states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)

            score = torch.norm(current_state - previous_state, p=2) / total_change
            scores.append(score)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def angle(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states = self._transformer_states(hidden_states)
        scores = []
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = self._angle_between(first_state, last_state).clamp_min(1e-12)

        for previous_state, current_state in self._layer_pairs(hidden_states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)

            score = self._angle_between(previous_state, current_state)
            scores.append(score / total_change)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def run(self, hidden_states: tuple[torch.Tensor, ...]) -> dict[str, Any]:
        magnitude_scores = self.magnitude(hidden_states)
        angle_scores = self.angle(hidden_states)

        return {
            "magnitude_change_scores": magnitude_scores["scores"],
            "magnitude_change_mean": magnitude_scores["mean"],
            "magnitude_change_std": magnitude_scores["std"],
            "angle_change_scores": angle_scores["scores"],
            "angle_change_mean": angle_scores["mean"],
            "angle_change_std": angle_scores["std"],
        }
