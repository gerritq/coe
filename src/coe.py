from typing import Any

import torch
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

    def magnitude(self, 
                  states: list[torch.Tensor]
                    ) -> dict[str, float]:
        
        scores = []

        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = torch.norm(last_state - first_state, p=2)

        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)

            score = torch.norm(current_state - previous_state, p=2) # / total_change
            scores.append(score)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def angle(self, 
              states: list[torch.Tensor]
                ) -> dict[str, float]:
        
        scores = []
        first_state = states[0].float().reshape(-1)
        last_state = states[-1].float().reshape(-1)
        total_change = self._angle_between(first_state, last_state)

        for previous_state, current_state in self._layer_pairs(states):
            previous_state = previous_state.float().reshape(-1)
            current_state = current_state.float().reshape(-1)

            score = self._angle_between(previous_state, current_state)
            # scores.append(score / total_change)
            scores.append(score)

        score_tensor = torch.stack(scores)
        return {
            "scores": score_tensor.tolist(),
            "mean": score_tensor.mean().item(),
            "std": score_tensor.std(unbiased=False).item(),
        }

    def run(self, 
            hidden_states: tuple[torch.Tensor, ...], 
            use_diff_vectors: bool
            ) -> dict[str, Any]:
        
        states = self._states(hidden_states=hidden_states, use_diff_vectors=use_diff_vectors)
        magnitude_scores = self.magnitude(states)
        angle_scores = self.angle(states)
        length_scores = self.length(states)

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
        }
