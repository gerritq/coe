import torch
from argparse import Namespace
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.utils import return_device

class MLModels:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = return_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "editlens" in self.model_name:
            peft_config = PeftConfig.from_pretrained(self.model_name)
            base_model_name = peft_config.base_model_name_or_path
            
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name,trust_remote_code=True)
            self.model = PeftModel.from_pretrained(base_model, self.model_name).to(self.device)
        else:
            attn_impl = "eager" if "radar" in self.model_name.lower() else None
            self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    attn_implementation=attn_impl,
                    trust_remote_code=True
                ).to(self.device)
        self.model.eval()

    def run(
        self,
        texts: list[str],
        args: Namespace
    ) -> list[float]:
        if not texts:
            return []

        batch_size = 24
        scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits

                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, 1].detach().cpu().tolist()

                scores.extend(float(s) for s in batch_scores)

        return scores
