import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = {
    "qwen_06b": "Qwen/Qwen3-0.6B",
    "qwen_8b": "Qwen/Qwen3-8B",
    "qwen_32b": "Qwen/Qwen3-32B",
    "llama_8b": "meta-llama/Meta-Llama-3-8B"

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
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
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        
        if args.last_token:
            hidden_states = tuple(
                layer[:, -1, :].detach().cpu() for layer in outputs.hidden_states
            )
        else:
            hidden_states = tuple(
                layer.mean(dim=1).detach().cpu() for layer in outputs.hidden_states
            )

    
        # print(hidden_states_last[0].shape)
        return {
            "model_id": self.model_id,
            "text": text,
            "label": item["label"],
            "hidden_states": hidden_states,
        }
