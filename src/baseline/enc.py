from argparse import Namespace
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

    def run(self, args: Namespace, data: Dataset) -> dict[str, list[float]]:
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

        val_predictions = trainer.predict(ds_tok["val"])
        test_predictions = trainer.predict(ds_tok["test"])

        val_logits = torch.from_numpy(val_predictions.predictions)
        test_logits = torch.from_numpy(test_predictions.predictions)

        val_probs = torch.nn.functional.softmax(val_logits, dim=-1).numpy()
        test_probs = torch.nn.functional.softmax(test_logits, dim=-1).numpy()

        return {
            "val_scores": val_probs[:, 1].tolist(),
            "test_scores": test_probs[:, 1].tolist(),
        }
