import os
import json
from argparse import ArgumentParser, Namespace
import numpy as np
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

from sklearn.model_selection import train_test_split
from utils import load_dataset, return_args, compute_metrics

BASE_DIR = os.getenv("BASE_COE")
BASELINE_DIR = os.path.join(BASE_DIR, "baselines", "test")
os.makedirs(BASELINE_DIR, exist_ok=True)

MAX_LENGTH = 256
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

MODE_DICT = {"bert": "google-bert/bert-base-uncased"}

def prepare_dataset(args: Namespace) -> list[dict[str, Any]]:
    data = load_dataset(args)
    
    # train and test split
    return train_test_split(
        [item["text"] for item in data],
        [item["label"] for item in data],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=[item["label"] for item in data],
    )

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

class BERTBaseline:
    def __init__(
        self,
        model_name: str,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        num_epochs: int = EPOCHS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        seed: int = SEED,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

    def run(self, args: Namespace) -> dict[str, Any]:
        set_seed(self.seed)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # prep data
        x_train, x_test, y_train, y_test = prepare_dataset(args)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        model.to(device)

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
        y_pred = np.argmax(predictions.predictions, axis=1)
        metrics = compute_metrics(y_test, y_pred)

        os.makedirs(BASELINE_DIR, exist_ok=True)
        out_path = os.path.join(BASELINE_DIR, f"bert_{args.dataset}.json")
        
        payload = {
            "metrics": metrics,
            "args": return_args(args),
        }
        
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--prefix", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    args.smoke_test = bool(args.smoke_test)
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    args.prefix = bool(args.prefix)

    
    
    baseline = BERTBaseline(model_name=MODE_DICT[args.model])
    baseline.run(args)

if __name__ == "__main__":
    main()
