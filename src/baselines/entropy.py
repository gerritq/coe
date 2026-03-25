import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class Entropy:
    
    def __init__(self, 
                 model_name: str, 
                 device: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                          trust_remote_code=True)
        self.model.eval()

    def _get_entropy(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.model.device) # input_ids + mask
            logits = self.model(**tokenized).logits[:, :-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            return -neg_entropy.sum(-1).mean().item()
        
    def run(self, texts: list[str]):
        '''wrapper function to run get_entropy for list of txts'''
        scores = []
        for text in texts:
            score = self._get_entropy(text)
            scores.append(score)
        return scores

