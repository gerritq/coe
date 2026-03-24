from fileinput import filename
import json
import os
import random
from datasets import Dataset, DatasetDict
from typing import Any

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

"""reddit cohere machine text is almost completely empty"""

def prepare_M4_data(
    input_path: str,
    filename: str,
    train_n: int = 1000,
    val_n: int = 250,
    test_n: int = 250,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    def pairs(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in subset:
            human_text = item.get("human_text", item.get("abstract", item.get("text")))
            machine_text = item.get("machine_text", item.get("machine_abstract", item.get("machine_answer")))
            if human_text and machine_text:
                result.append({"text": human_text, "label": 0})
                result.append({"text": machine_text, "label": 1})
            else:
                print(f"[Warning] Missing human_text or machine_text in item: {item.get("human_text")}, {item.get("machine_text")}")
        return result

    random.seed(seed)

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
    output_path = os.path.join(DATA_DIR, f"{stem}")
    dataset.save_to_disk(output_path)

    return {stem: summary}

	


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
    summaries: dict[str, dict[str, int]] = {}

    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        print("=" * 50)
        print(f"Processing file: {filename}")
        print("=" * 50)
        if not filename.endswith(".jsonl"):
            continue
        if "counterfact"  in filename:
            continue

        input_path = os.path.join(RAW_DATA_DIR, filename)
        result = prepare_M4_data(input_path=input_path, filename=filename)
        summaries.update(result)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
