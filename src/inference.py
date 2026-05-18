from typing import Any
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.utils import return_device
from argparse import Namespace

MODEL_DIR = {
    "smol": "HuggingFaceTB/SmolLM-135M",
    "qwen_06b": "Qwen/Qwen3-0.6B",
    "qwen_8b": "Qwen/Qwen3-8B",
    "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct"
}

class Inference:
    def __init__(self, 
                 model_name: str) -> None:
        
        self.model_id = MODEL_DIR[model_name]
        self.device = return_device()

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, 
                                                          attn_implementation="eager",)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def run(self, 
            item: dict, 
            args: Namespace) -> dict[str, Any]:
        
    
        text = item["text"]
        
        if not text.strip():
            raise ValueError("Input text must be non-empty.")

        inputs = self.tokenizer(text, 
                                truncation=True,
                                max_length=1024,
                                return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, 
                                 output_hidden_states=True, 
                                 output_attentions=False,
                                 use_cache=False)
            
        # attentions/hidden states are tuples of len layer

        # extract hidden states
        if args.token_mode == "last_token":
            hidden_states = tuple(
                layer[:, -1, :].detach().cpu().squeeze(0) for layer in outputs.hidden_states
            )
        if args.token_mode == "pooling":
            hidden_states = tuple(
                layer.mean(dim=1).detach().cpu().squeeze(0) for layer in outputs.hidden_states
            )


        return {
            "model_id": self.model_id,
            "text": text,
            "label": item["label"],
            "hidden_states": hidden_states,
        }

