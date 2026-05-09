from argparse import Namespace
from ctypes import Union
import torch
import json
from datasets import Dataset
from src.utils import (return_args, 
                       metrics, 
                       optimal_thresholds)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import os
from datetime import datetime

BASE_DIR = os.getenv("BASE_COE")

MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

class EncoderBaseline:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        seed: int = SEED,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

    def run(self, 
            args: Namespace, 
            training_data: Dataset,
            ood_data: list[Dataset]
            ) -> None:
        
        self.out_dir = os.path.join(BASE_DIR, "output", "baseline", args.folder)
        os.makedirs(self.out_dir, exist_ok=True)

        set_seed(self.seed)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        model.to(self.device)

        def tokenize_function(examples):
                    return tokenizer(examples["text"], truncation=True, max_length=self.max_length)

        ds_tok = training_data.map(tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=None,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=50,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tok["train"],
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        val_predictions = trainer.predict(ds_tok["val"])
        val_logits = torch.from_numpy(val_predictions.predictions)
        val_probs = torch.nn.functional.softmax(val_logits, dim=-1).numpy()

        optimal_thresholds_dict = optimal_thresholds(y_true=ds_tok["val"]["label"], 
                                                    y_predict=val_probs[:, 1])

        for ood_dict in ood_data:
            ood_ds = ood_dict['data']
            ood_name = ood_dict['name']

            ood_ds_tok = ood_ds.map(tokenize_function, batched=True)
            ood_predictions = trainer.predict(ood_ds_tok["test"])
            ood_logits = torch.from_numpy(ood_predictions.predictions)
            ood_probs = torch.nn.functional.softmax(ood_logits, dim=-1).numpy()

            ds_metrics = metrics(y_true=ood_ds['test']["label"],
                                  y_predict=ood_probs[:, 1],
                                  acc_threshold=optimal_thresholds_dict["threshold_acc"],
                                  f1_threshold=optimal_thresholds_dict["threshold_f1"])
            
            file_name = f"{args.model}_{args.dataset}_2_{ood_name}.json"

            args_copy = Namespace(**vars(args))  
            out_args = return_args(args_copy)
            out_args['target_dataset'] = ood_name
            out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            out = {"args": out_args, 
                   "metrics": ds_metrics}
            with open(os.path.join(self.out_dir, file_name), "w") as f:
                json.dump(out, f, indent=2)
