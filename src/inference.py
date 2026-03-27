import os
import sys
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = {
    "qwen_06b": "Qwen/Qwen3-0.6B",
    "qwen_8b": "Qwen/Qwen3-8B",
    "qwen_32b": "Qwen/Qwen3-32B",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct"

}


class Inference:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        if model_name not in MODEL_DIR:
            available = ", ".join(sorted(MODEL_DIR))
            raise ValueError(
                f"Unknown model_name '{model_name}'. Available models: {available}"
            )

        self.model_id = MODEL_DIR[model_name]
        self.device = device or self._resolve_device()

        config = AutoConfig.from_pretrained(self.model_id)
        config.tie_word_embeddings = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def run(self, item: dict, args) -> dict[str, Any]:
        text = item["text"]
        if not text.strip():
            raise ValueError("Input text must be non-empty.")

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, 
                                 output_hidden_states=True, 
                                 use_cache=False)

        if args.mode == "last_token":
            hidden_states = tuple(
                layer[:, -1, :].detach().cpu() for layer in outputs.hidden_states
            )
            return {
                "model_id": self.model_id,
                "text": text,
                "label": item["label"],
                "hidden_states": hidden_states,
            }
        elif args.mode == "pooling":
            hidden_states = tuple(
                layer.mean(dim=1).detach().cpu() for layer in outputs.hidden_states
            )
            return {
                "model_id": self.model_id,
                "text": text,
                "label": item["label"],
                "hidden_states": hidden_states,
            }
        elif args.mode == "horizontal":
            last_layer = outputs.hidden_states[-1].detach().cpu()
            hidden_states = tuple(
                last_layer[:, t, :] for t in range(last_layer.shape[1])
            )
            return {
                "model_id": self.model_id,
                "text": text,
                "label": item["label"],
                "hidden_states": hidden_states,
            }
        elif args.mode == "logits":
            logits = outputs.logits.detach().cpu()
            return {
                "model_id": self.model_id,
                "text": text,
                "label": item["label"],
                "logits": logits,
            }
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
