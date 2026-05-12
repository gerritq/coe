import os
import json
from random import sample
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset
from argparse import Namespace
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


from src.utils import return_device, metrics, return_args
BASE_DIR = os.getenv("BASE_COE")

COMPLETION_PROMPT_ONLY = "Complete the following text: "
COMPLETION_PROMPT = "Given the summary:\n{prompt}\n Complete the following text: "

class BiScope:

    def __init__(self, args: Namespace):
        
        args.summary_model = args.base_model_2
        args.detect_model = args.base_model_2

        self.summary_model = AutoModelForCausalLM.from_pretrained(
                args.summary_model,
                torch_dtype=torch.float16,
                device_map='auto'
            ).eval()
        self.summary_tokenizer = AutoTokenizer.from_pretrained(
                args.summary_model, padding_side='left'
            )
        self.summary_tokenizer.pad_token = self.summary_tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
                args.detect_model,
                torch_dtype=torch.float16,
                device_map='auto'
            ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
                args.detect_model, padding_side='left'
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = return_device()
        args.sample_clip = 2000
        self.sample_clip = 2000
        args.seed = 42

    def generate(self, model, tokenizer, input_ids, trigger_length, target_length):
        """
        Generate additional tokens using the model's generation API.
        
        Parameters:
        model: the language model for generation.
        tokenizer: associated tokenizer.
        input_ids: input token IDs (either 1D or 2D).
        trigger_length: the length of the prompt (number of tokens to skip in the output).
        target_length: the number of new tokens to generate.
        
        Returns:
        Generated tokens (as a 2D tensor) after removing the trigger tokens.
        """
        config = model.generation_config
        config.max_new_tokens = target_length
        # If input_ids is 1D, add a batch dimension; otherwise, assume it's already 2D.
        if input_ids.dim() == 1:
            input_ids = input_ids.to(model.device).unsqueeze(0)
        else:
            input_ids = input_ids.to(model.device)
        # Create an attention mask of the same shape.
        attn_masks = torch.ones(input_ids.shape, device=input_ids.device)
        # Generate new tokens.
        out = model.generate(
            input_ids, 
            attention_mask=attn_masks,
            generation_config=config,
            pad_token_id=tokenizer.pad_token_id
        )[0]
        # Return output tokens after the prompt (slice along dimension 1).
        return out[trigger_length:]


    def compute_fce_loss(self, logits, targets, text_slice):
        """
        Compute the FCE loss by shifting indices by 1.
        Returns a NumPy array of loss values.
        """
        loss = CrossEntropyLoss(reduction='none')(
            logits[0, text_slice.start-1:text_slice.stop-1, :],
            targets
        )
        return loss.detach().cpu().numpy()

    def compute_bce_loss(self, logits, targets, text_slice):
        """
        Compute the BCE loss without shifting indices.
        Returns a NumPy array of loss values.
        """
        loss = CrossEntropyLoss(reduction='none')(
            logits[0, text_slice, :],
            targets
        )
        return loss.detach().cpu().numpy()

    def detect_single_sample(self,
                             args, 
                             item: str
                             ) -> list[float]:

        # SUMMARY MODEL GENERATION
        # sample = item['text']

        # summary_input = f"Write a title for this text: {sample}\nJust output the title:"
        # summary_ids = self.summary_tokenizer(summary_input, return_tensors='pt',
        #                                 max_length=self.sample_clip, truncation=True).input_ids.to(self.device)
        # summary_ids = summary_ids[:, 1:]  # Remove start token.
        # gen_ids = self.generate(self.summary_model, self.summary_tokenizer, summary_ids, summary_ids.shape[1], 64)
        # summary_text = self.summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
        
        sample = item['text']
        
        summary_text = item['summary_text']
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)


        # Tokenize the prompt and sample with token-level clipping.
        prompt_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids.to(self.device)
        text_ids = self.tokenizer(sample, return_tensors='pt', max_length=args.sample_clip, truncation=True).input_ids.to(self.device)
        combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
        text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])
        outputs = self.model(input_ids=combined_ids)
        logits = outputs.logits
        targets = combined_ids[0][text_slice]
        
        # Compute loss features from FCE and BCE losses.
        fce_loss = self.compute_fce_loss(logits, targets, text_slice)
        bce_loss = self.compute_bce_loss(logits, targets, text_slice)
        features = []
        for p in range(1, 10):
            split = len(fce_loss) * p // 10
            features.extend([
                np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
                np.min(fce_loss[split:]), np.std(fce_loss[split:]),
                np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
                np.min(bce_loss[split:]), np.std(bce_loss[split:])
            ])

        # guard against nan and inf
        clip = 1e6
        features = np.asarray(features, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=clip, neginf=-clip)
        return features


    def generate_summaries_batch(self, items: list[dict]) -> list[dict]:
        """"
        GQ: we generate this function for faster inference of summaries
        takes ages w/o batching 
        """
        # SUMMARY MODEL GENERATION
        # sample = item['text']

        # summary_input = f"Write a title for this text: {sample}\nJust output the title:"
        # summary_ids = self.summary_tokenizer(summary_input, return_tensors='pt',
        #                                 max_length=self.sample_clip, truncation=True).input_ids.to(self.device)
        # summary_ids = summary_ids[:, 1:]  # Remove start token.
        # gen_ids = self.generate(self.summary_model, self.summary_tokenizer, summary_ids, summary_ids.shape[1], 64)
        # summary_text = self.summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
        batch_size = 24

        for start in tqdm(range(0, len(items), batch_size)):
            batch = items[start:start + batch_size]
            summary_inputs = [
                f"Write a title for this text: {item['text']}\nJust output the title:"
                for item in batch
            ]
            tokenized = self.summary_tokenizer(
                summary_inputs,
                return_tensors='pt',
                padding=True,
                max_length=self.sample_clip,
                truncation=True
            )
            summary_ids = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)
            summary_ids = summary_ids[:, 1:]  # Remove start token.
            attention_mask = attention_mask[:, 1:]

            config = self.summary_model.generation_config
            config.max_new_tokens = 64
            outputs = self.summary_model.generate(
                summary_ids,
                attention_mask=attention_mask,
                generation_config=config,
                pad_token_id=self.summary_tokenizer.pad_token_id
            )

            input_len = summary_ids.shape[1]
            for idx, item in enumerate(batch):
                gen_ids = outputs[idx, input_len:]
                summary_text = self.summary_tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True
                ).strip().split('\n')[0]
                item['summary_text'] = summary_text
        return items

    def data_generation(self,
                        args, 
                        human_data: list[str],
                        machine_data: list[str]) -> tuple[list[float], list[float]]:
        """
        Generate loss-based features for both human and GPT samples and save them to disk.
        
        Parameters:
        out_dir: Output directory.
        dataset_type: 'paraphrased' or 'nonparaphrased'.
        task: Task name (e.g., Arxiv, Code, Essay).
        generative_model: Key for the GPT samples.
        
        Returns:
        The output directory.
        """

        # GQ: add summaries to the keys
        human_features = self.generate_summaries_batch(human_data)
        machine_features = self.generate_summaries_batch(machine_data)

        
        human_features = [self.detect_single_sample(args, item) for item in tqdm(human_data)]
        machine_features = [self.detect_single_sample(args, item) for item in tqdm(machine_data)]
        
        return human_features, machine_features
    
    def run(self, 
            args: Namespace, 
            training_data: Dataset,
            ood_data: list[Dataset]) -> None:
        
        set_seed(42)

        # Set args for their code

        args.summary_model = args.base_model_2
        args.detect_model = args.base_model_2

        
        # TRAINGING DATA
        train_machine = [item for item in training_data['train'] if item['label'] == 1]
        train_human = [item for item in training_data['train'] if item['label'] == 0]
        
        train_human_features, train_machine_features = self.data_generation(args, train_human, train_machine)

        # TRAIN
        train_feats = np.concatenate([train_human_features, train_machine_features], axis=0)
        train_labels = np.concatenate([np.zeros(len(train_human_features)), np.ones(len(train_machine_features))], axis=0)
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        clf.fit(train_feats, train_labels)

        # OOD EVAL
        for ood_ds in ood_data:
            ood_machine = [item for item in ood_ds['data']['test'] if item['label'] == 1]
            ood_human = [item for item in ood_ds['data']['test'] if item['label'] == 0]
            ood_human_features, ood_machine_features = self.data_generation(args, ood_human, ood_machine)
            
            test_feats = np.concatenate([ood_human_features, ood_machine_features], axis=0)
            test_labels = np.concatenate([np.zeros(len(ood_human_features)), np.ones(len(ood_machine_features))], axis=0)
            ood_probs = clf.predict_proba(test_feats)[:, 1]

            metrics_results = metrics(y_true=test_labels,
                                      y_predict=ood_probs,
                                        acc_threshold=.5,
                                        f1_threshold=.5)
            
            file_name = f"{args.model}_{args.dataset}_2_{ood_ds['name']}_N{args.training_size}.json"

            args_copy = Namespace(**vars(args))  
            out_args = return_args(args_copy)
            out_args['target_dataset'] = ood_ds['name']
            out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            out = {"args": out_args, 
                   "metrics": metrics_results}
            self.out_dir = os.path.join(BASE_DIR, "output", "baseline", args.folder)
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, file_name), "w") as f:
                json.dump(out, f, indent=2)
    
