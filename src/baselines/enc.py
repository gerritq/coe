from argparse import Namespace
import numpy as np
from utils import compute_metrics
from typing import Any
import torch
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MAX_LENGTH = 256
TEST_SIZE = 0.2
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

    def run(self, args: Namespace,
            data: Dataset
            ) -> dict[str, Any]:
        set_seed(self.seed)

    
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        model.to(self.device)

        def tokenize_function(examples):
                    return tokenizer(examples["text"], truncation=True, max_length=self.max_length)

        ds_tok = data.map(tokenize_function, batched=True)

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

        
        predictions = trainer.predict(ds_tok["test"])
        # precited labels --- what to do with them
        y_pred = np.argmax(predictions.predictions, axis=1)
        metrics = compute_metrics(ds_tok["test"]["labels"], y_pred)

        # get probs
        logits = torch.from_numpy(predictions.predictions)
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        y_score = probs[:, 1]

        return y_score.tolist()
