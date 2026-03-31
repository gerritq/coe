import json
import os
import random
from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import Any

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")
M4_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4")
MULTISOCIAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multisocial")
TSM_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "tsm")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(TSM_RAW_DATA_DIR, exist_ok=True)

"""
M4
- reddit cohere machine text is almost completely empty
- this is parallel corpora; we keep it parallel bc we may wanna try mean difference steering vectors

Multisocial
- 

"""

TRAINING_N = 2000
VALIDATION_N = 500
TESTING_N = 500
SEED=42

random.seed(SEED)

def prepare_TSM_data() -> None:
    
    train_n = TRAINING_N // 2
    val_n = VALIDATION_N // 2
    test_n = TESTING_N // 2 

    random.seed(SEED)
    
    def pairs(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in subset:
            human_text = item.get("trgt", None)
            machine_text = item.get("mgt", None)
            if human_text and machine_text:
                result.append({"text": human_text, "label": 0})
                result.append({"text": machine_text, "label": 1})
            else:
                print(f"[Warning] Missing human_text or machine_text in item.")
        return result

    if not os.path.isdir(TSM_RAW_DATA_DIR):
        raise FileNotFoundError(f"Could not find TSM folder at: {TSM_RAW_DATA_DIR}")

    summaries: dict[str, dict[str, int]] = {}

    for filename in sorted(os.listdir(TSM_RAW_DATA_DIR)):
        if not filename.endswith(".jsonl"):
            continue

        print("=" * 50)
        print(f"Processing file: {filename}")
        print("=" * 50)

        input_path = os.path.join(TSM_RAW_DATA_DIR, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = []
            for line in f:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    print(f"[Error]: {error}")
                    continue

        random.shuffle(raw_data)

        train_raw_data = raw_data[:train_n]
        val_raw_data = raw_data[train_n : train_n + val_n]
        test_raw_data = raw_data[train_n + val_n : train_n + val_n + test_n]

        train_data = pairs(train_raw_data)
        val_data = pairs(val_raw_data)
        test_data = pairs(test_raw_data)

        dataset_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        dataset = DatasetDict({split: Dataset.from_list(data) for split, data in dataset_data.items()})
        summary = {split: len(data) for split, data in dataset.items()}

        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(DATA_DIR, stem)
        dataset.save_to_disk(output_path)
        summaries[stem] = summary

    print("\nSummary of dataset splits:")
    for dataset_name, summary in summaries.items():
        print(f"{dataset_name}: {summary}")

def prepare_M4_data() -> None:


    train_n = TRAINING_N // 2
    val_n = VALIDATION_N // 2
    test_n = TESTING_N // 2 

    random.seed(SEED)
    
    def pairs(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in subset:
            human_text = item.get("human_text", item.get("abstract", item.get("text", item.get('trgt'))))
            machine_text = item.get("machine_text", item.get("machine_abstract", item.get("machine_answer", item.get("mgt"))))
            if human_text and machine_text:
                result.append({"text": human_text, "label": 0})
                result.append({"text": machine_text, "label": 1})
            else:
                print(f"[Warning] Missing human_text or machine_text in item.")
        return result

    if not os.path.isdir(M4_RAW_DATA_DIR):
        raise FileNotFoundError(f"Could not find M4 folder at: {M4_RAW_DATA_DIR}")

    summaries: dict[str, dict[str, int]] = {}

    for filename in sorted(os.listdir(M4_RAW_DATA_DIR)):
        if not filename.endswith(".jsonl"):
            continue

        print("=" * 50)
        print(f"Processing file: {filename}")
        print("=" * 50)

        input_path = os.path.join(M4_RAW_DATA_DIR, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = []
            for line in f:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    print(f"[Error]: {error}")
                    continue

        random.shuffle(raw_data)

        train_raw_data = raw_data[:train_n]
        val_raw_data = raw_data[train_n : train_n + val_n]
        test_raw_data = raw_data[train_n + val_n : train_n + val_n + test_n]

        train_data = pairs(train_raw_data)
        val_data = pairs(val_raw_data)
        test_data = pairs(test_raw_data)

        dataset_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        dataset = DatasetDict({split: Dataset.from_list(data) for split, data in dataset_data.items()})
        summary = {split: len(data) for split, data in dataset.items()}

        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(DATA_DIR, stem)
        dataset.save_to_disk(output_path)
        summaries[stem] = summary

    print("\nSummary of dataset splits:")
    for dataset_name, summary in summaries.items():
        print(f"{dataset_name}: {summary}")


def prepare_multisocial_data() -> None:
    """prep mutlisocial data. two subsets
    1) random sample from full train, test sets
    2) random samples by language
    
    """

    random.seed(SEED)
    
    input_path = os.path.join(MULTISOCIAL_RAW_DATA_DIR, "multisocial_anonymized.csv")

    dataset = Dataset.from_csv(input_path)

    def build_subset(rows: Dataset, 
                     name: str
                     ) -> None:
        
        train_rows = rows.filter(lambda x: x["split"] == "train")
        test_rows = rows.filter(lambda x: x["split"] == "test")

        train_per_label = TRAINING_N // 2
        val_per_label = VALIDATION_N // 2
        test_per_label = TESTING_N // 2

        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        test_parts: list[Dataset] = []

        for label in ["0", "1"]:
            train_label_rows = train_rows.filter(lambda x, label=label: str(x["label"]) == label).shuffle(seed=SEED)
            test_label_rows = test_rows.filter(lambda x, label=label: str(x["label"]) == label).shuffle(seed=SEED)

            train_parts.append(train_label_rows.select(range(train_per_label)))
            val_parts.append(train_label_rows.select(range(train_per_label, train_per_label + val_per_label)))
            test_parts.append(test_label_rows.select(range(test_per_label)))

        final_train = concatenate_datasets(train_parts).shuffle(seed=SEED)
        final_val = concatenate_datasets(val_parts).shuffle(seed=SEED)
        final_test = concatenate_datasets(test_parts).shuffle(seed=SEED)

        output = DatasetDict({
            "train": final_train,
            "val": final_val,
            "test": final_test,
        })

        print(f"SUMMARY for {name}:")
        for split, data in output.items():
            print(f"  {split.capitalize()}: {len(data)}")
        output.save_to_disk(os.path.join(DATA_DIR, name))

    build_subset(dataset, "multisocial_full")

    for lang in ["en", "de", "ar", "nl", "pt"]:
        lang_rows = dataset.filter(lambda x, lang=lang: x["language"] == lang)
        build_subset(lang_rows, f"multisocial_{lang}")


# def counterfact_data() -> None:
#     """Valid function to gen the counterfact data, but we hasn't use it iin the end."""

#     def build_claim(prompt: str, subject: str, target: str) -> str:
#         base = prompt.format(subject) if "{}" in prompt else f"{subject} {prompt}".strip()
#         separator = "" if base.endswith((" ", "\n", "\t")) else " "
#         return f"{base}{separator}{target}".strip()

#     def convert_counterfact_to_jsonl(input_path: str, output_path: str) -> int:
#         with open(input_path, "r", encoding="utf-8") as f:
#             records = json.load(f)

#         written = 0
#         with open(output_path, "w", encoding="utf-8") as out:
#             for item in records:
#                 if not item or "requested_rewrite" not in item:
#                     continue

#                 rewrite = item["requested_rewrite"]
#                 case_id = item.get("case_id")
#                 subject = rewrite.get("subject", "")
#                 prompt = rewrite.get("prompt", "")
#                 target_true = rewrite.get("target_true", {}).get("str", "")
#                 target_new = rewrite.get("target_new", {}).get("str", "")

#                 if case_id is None or not prompt or not subject or not target_true or not target_new:
#                     continue

#                 row = {
#                     "case_id": case_id,
#                     "correct": build_claim(prompt, subject, target_true),
#                     "incorrect": build_claim(prompt, subject, target_new),
#                 }
#                 out.write(json.dumps(row, ensure_ascii=False) + "\n")
#                 written += 1
#         return written

#     convert_counterfact_to_jsonl(
#         input_path=os.path.join(DATA_DIR, "counterfact.json"),
#         output_path=os.path.join(DATA_DIR, "counterfact.jsonl"),
#     )


def main() -> None:
    
    
    # M4
    # need to halve bc one instance contains both machine and human text
    print("Preparing M4 data...")
    prepare_M4_data()
    
    # MULTISOCIAL
    print("Preparing Multisocial data...")
    prepare_multisocial_data()

    # TSM
    prepare_TSM_data()


if __name__ == "__main__":
    main()
