from argparse import Namespace
import numpy as np
from utils import compute_metrics
from typing import Any
import torch


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

def build_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    dataset = []
    for text, label in zip(texts, labels, strict=True):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = int(label)
        dataset.append(encoded)
    return dataset


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
            splits: tuple[Any, Any, Any, Any]
            ) -> dict[str, Any]:
        set_seed(self.seed)

        x_train, x_test, y_train, y_test = splits

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        model.to(self.device)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataset = build_dataset(x_train, y_train, tokenizer, self.max_length)
        test_dataset = build_dataset(x_test, y_test, tokenizer, self.max_length)

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
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        
        predictions = trainer.predict(test_dataset)
        # precited labels --- what to do with them
        y_pred = np.argmax(predictions.predictions, axis=1)
        metrics = compute_metrics(y_test, y_pred)

        # get probs
        logits = torch.from_numpy(predictions.predictions)
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        y_score = probs[:, 1]

        return y_score.tolist()
