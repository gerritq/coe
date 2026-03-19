from typing import Any

import torch
import sys

class Metrics:
    def __init__(self, top_k: int = 1000) -> None:
        self.top_k = top_k

    def _entropy_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # print(probs.shape) # shape 1 x sequence length x vocab size
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)
        # print(entropy.shape) # dim 1 x sequence length
        return entropy

    def _entropy_full(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return self._entropy_from_probs(probs)

    def _entropy_topk(self, logits: torch.Tensor) -> torch.Tensor:
        topk_vals, _ = torch.topk(logits, k=self.top_k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        return self._entropy_from_probs(probs)

    def _tvd_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        diffs = (probs[:, 1:] - probs[:, :-1]).abs()
        tvd = 0.5 * diffs.sum(dim=-1)
        return tvd.squeeze(0)

    def _tvd_full(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return self._tvd_from_probs(probs)

    def _tvd_topk(self, logits: torch.Tensor) -> torch.Tensor:
        topk_vals, _ = torch.topk(logits, k=self.top_k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        return self._tvd_from_probs(probs)

    def _per_token_entropy(self, logits: torch.Tensor, topk: bool) -> torch.Tensor:
        if topk:
            ent = self._entropy_topk(logits)
        else:
            ent = self._entropy_full(logits)
        
        # dim 1 x sequence length
        ent = ent.squeeze(0)  # remove batch dimension
        # dim sequence length
        return ent


    def _change_scores(self, values: torch.Tensor) -> torch.Tensor:
        diffs = (values[1:] - values[:-1]).abs()
        # norm by total
        # total = (values[-1] - values[0]).abs().clamp_min(1e-12)
        # norm by mean
        mean = values.mean().abs().clamp_min(1e-12)
        return diffs / mean

    def _stats(self, values: torch.Tensor) -> dict[str, Any]:
        return {
            "scores": values.tolist(),
            "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(),
        }

    def run(self, logits: torch.Tensor) -> dict[str, Any]:
        # return vecgor of entropies of sequence length
        vocab_entropy = self._per_token_entropy(logits, topk=False)
        topk_entropy = self._per_token_entropy(logits, topk=True)

        vocab_tvd = self._tvd_full(logits)
        topk_tvd = self._tvd_topk(logits)

        vocab_change = self._change_scores(vocab_entropy)
        topk_change = self._change_scores(topk_entropy)

        vocab_stats = self._stats(vocab_entropy)
        topk_stats = self._stats(topk_entropy)

        vocab_change_stats = self._stats(vocab_change)
        topk_change_stats = self._stats(topk_change)

        vocab_tvd_stats = self._stats(vocab_tvd)
        topk_tvd_stats = self._stats(topk_tvd)

        return {
            "vocab_entropy_scores": vocab_stats["scores"],
            "vocab_entropy_mean": vocab_stats["mean"],
            "vocab_entropy_std": vocab_stats["std"],
            "vocab_entropy_change_scores": vocab_change_stats["scores"],
            "vocab_entropy_change_mean": vocab_change_stats["mean"],
            "vocab_entropy_change_std": vocab_change_stats["std"],
            "vocab_tvd_scores": vocab_tvd_stats["scores"],
            "vocab_tvd_mean": vocab_tvd_stats["mean"],
            "vocab_tvd_std": vocab_tvd_stats["std"],
            "topk_entropy_scores": topk_stats["scores"],
            "topk_entropy_mean": topk_stats["mean"],
            "topk_entropy_std": topk_stats["std"],
            "topk_entropy_change_scores": topk_change_stats["scores"],
            "topk_entropy_change_mean": topk_change_stats["mean"],
            "topk_entropy_change_std": topk_change_stats["std"],
            "topk_tvd_scores": topk_tvd_stats["scores"],
            "topk_tvd_mean": topk_tvd_stats["mean"],
            "topk_tvd_std": topk_tvd_stats["std"],
        }
