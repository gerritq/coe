import json
import os
import random
from datasets import Dataset, concatenate_datasets
from typing import Any

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")

M4_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4")
M4_MULTILINGUAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4_multi")
MULTISOCIAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multisocial")
TSM_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "tsm")
DETECT_RL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl")
MULTITUDE_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multitude")
EDITLENS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "editlens")

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

def prepare_editlens() -> None:
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    def sample_split(csv_path: str, per_label: int) -> list[dict[str, Any]]:
        rows_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

        dataset = Dataset.from_csv(csv_path)
        for row in dataset:
            text = str(row.get("text", "")).strip()
            text_type = str(row.get("text_type", "")).strip().lower()
            if not text:
                continue

            if text_type == "human_written":
                label = 0
            elif text_type in {"ai_generated", "ai_edited"}:
                label = 1
            else:
                continue

            rows_by_label[label].append({"text": text, "label": label})

        rng = random.Random(SEED)
        rng.shuffle(rows_by_label[0])
        rng.shuffle(rows_by_label[1])

        take = min(per_label, len(rows_by_label[0]), len(rows_by_label[1]))
        sampled = rows_by_label[0][:take] + rows_by_label[1][:take]
        rng.shuffle(sampled)
        return sampled

    val_path = os.path.join(EDITLENS_RAW_DATA_DIR, "val.csv")
    test_path = os.path.join(EDITLENS_RAW_DATA_DIR, "test.csv")

    split_data = {
        "val": sample_split(val_path, val_per_label),
        "test": sample_split(test_path, test_per_label),
    }
    save_jsonl_splits("editlens", split_data)

    print("\nSummary of EditLens splits:")
    print({split: len(rows) for split, rows in split_data.items()})

def prepare_multitude()-> None:
    languages = ["de", "en", "uk", "pt", "ro", "nl"]
    allowed_models = {"human", "gpt-3.5-turbo-0125"}
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    input_path = os.path.join(MULTITUDE_RAW_DATA_DIR, "multitude_v3_clean.csv")
    dataset = Dataset.from_csv(input_path)

    print("\nPreparing MULTITUDE subsets:")

    for language in languages:
        lang_rows = dataset.filter(lambda x, lang=language: str(x["language"]) == lang)

        selected_rows = lang_rows.filter(
            lambda x: str(x["multi_label"]) in allowed_models and str(x.get("text", "")).strip() != ""
        )

        train_rows = selected_rows.filter(lambda x: str(x["split"]) == "train").shuffle(seed=SEED)
        test_rows = selected_rows.filter(lambda x: str(x["split"]) == "test").shuffle(seed=SEED)

        train_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        test_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

        for row in train_rows:
            label = 0 if str(row["multi_label"]) == "human" else 1
            train_by_label[label].append({"text": row["text"], "label": label})

        for row in test_rows:
            label = 0 if str(row["multi_label"]) == "human" else 1
            test_by_label[label].append({"text": row["text"], "label": label})

        rng = random.Random(SEED)
        rng.shuffle(train_by_label[0])
        rng.shuffle(train_by_label[1])
        rng.shuffle(test_by_label[0])
        rng.shuffle(test_by_label[1])

        # First sample a balanced validation set from train.
        val_take = min(val_per_label, len(train_by_label[0]), len(train_by_label[1]))
        val_data = train_by_label[0][:val_take] + train_by_label[1][:val_take]

        # Remaining train rows after carving out validation.
        rem_train_h = train_by_label[0][val_take:]
        rem_train_m = train_by_label[1][val_take:]
        train_take = min(len(rem_train_h), len(rem_train_m))
        train_data = rem_train_h[:train_take] + rem_train_m[:train_take]

        # Keep test as a balanced random sample of up to 500.
        test_take = min(test_per_label, len(test_by_label[0]), len(test_by_label[1]))
        test_data = test_by_label[0][:test_take] + test_by_label[1][:test_take]

        rng.shuffle(train_data)
        rng.shuffle(val_data)
        rng.shuffle(test_data)

        dataset_name = f"multitude_{language}"
        split_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        save_jsonl_splits(dataset_name, split_data)

        summary = {
            split: len(rows)
            for split, rows in split_data.items()
        }
        print(f"{dataset_name}: {summary}")

def to_binary_label(label: Any) -> int:
    label_str = str(label).strip().lower()
    if label_str in {"human", "0"}:
        return 0
    if label_str in {"llm", "machine", "1"}:
        return 1
    raise ValueError(f"Unsupported label value: {label}")

def save_jsonl_splits(name: str, splits: dict[str, list[dict[str, Any]]]) -> None:
    output_dir = os.path.join(DATA_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    for split_name, rows in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

def prepare_detectrl_task_1()-> None:
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    task_dir = os.path.join(DETECT_RL_RAW_DATA_DIR, "task_1")
    variants = ["paraphrase", "perturbation"]

    summaries: dict[str, dict[str, int]] = {}

    for variant in variants:
        train_path = os.path.join(task_dir, f"{variant}_attacks_llm_train.json")
        test_path = os.path.join(task_dir, f"{variant}_attacks_llm_test.json")

        with open(train_path, "r", encoding="utf-8") as f:
            train_rows: list[dict[str, Any]] = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_rows: list[dict[str, Any]] = json.load(f)

        train_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        test_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

        for row in train_rows:
            text = row.get("text")
            if not text:
                continue
            label = to_binary_label(row.get("label"))
            train_by_label[label].append({"text": text, "label": label})

        for row in test_rows:
            text = row.get("text")
            if not text:
                continue
            label = to_binary_label(row.get("label"))
            test_by_label[label].append({"text": text, "label": label})

        random.Random(SEED).shuffle(train_by_label[0])
        random.Random(SEED).shuffle(train_by_label[1])
        random.Random(SEED).shuffle(test_by_label[0])
        random.Random(SEED).shuffle(test_by_label[1])

        train_data = train_by_label[0][:train_per_label] + train_by_label[1][:train_per_label]
        val_data = (
            train_by_label[0][train_per_label:train_per_label + val_per_label]
            + train_by_label[1][train_per_label:train_per_label + val_per_label]
        )
        test_data = test_by_label[0][:test_per_label] + test_by_label[1][:test_per_label]

        random.Random(SEED).shuffle(train_data)
        random.Random(SEED).shuffle(val_data)
        random.Random(SEED).shuffle(test_data)

        dataset_name = f"drl_t1_{variant}"
        split_data = {"train": train_data, "val": val_data, "test": test_data}
        save_jsonl_splits(dataset_name, split_data)
        summaries[dataset_name] = {split: len(rows) for split, rows in split_data.items()}

    print("\nSummary of DetectRL task_1 attack splits:")
    for name, summary in summaries.items():
        print(f"{name}: {summary}")

def prepare_tsm_multi()-> None:
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    for filename in sorted(os.listdir(TSM_RAW_DATA_DIR)):
        if not filename.endswith(".jsonl"):
            continue

        input_path = os.path.join(TSM_RAW_DATA_DIR, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    print(f"[Error] {filename}: {error}")
                    continue

                human_text = row.get("trgt")
                machine_text = row.get("mgt")
                if not human_text or not machine_text:
                    continue

                by_label[0].append({"text": human_text, "label": 0})
                by_label[1].append({"text": machine_text, "label": 1})

    needed = train_per_label + val_per_label + test_per_label

    random.Random(SEED).shuffle(by_label[0])
    random.Random(SEED).shuffle(by_label[1])

    train_data = by_label[0][:train_per_label] + by_label[1][:train_per_label]
    val_data = (
        by_label[0][train_per_label:train_per_label + val_per_label]
        + by_label[1][train_per_label:train_per_label + val_per_label]
    )
    test_data = (
        by_label[0][train_per_label + val_per_label:train_per_label + val_per_label + test_per_label]
        + by_label[1][train_per_label + val_per_label:train_per_label + val_per_label + test_per_label]
    )

    random.Random(SEED).shuffle(train_data)
    random.Random(SEED).shuffle(val_data)
    random.Random(SEED).shuffle(test_data)

    split_data = {"train": train_data, "val": val_data, "test": test_data}
    save_jsonl_splits("tsm_multi", split_data)
    print("\nSummary of tsm_multi splits:")
    print({split: len(data) for split, data in split_data.items()})

def prepare_M4_multi_data() -> None:
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    train_path = os.path.join(M4_MULTILINGUAL_RAW_DATA_DIR, "subtaskA_train_multilingual.jsonl")

    train_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[Error] train: {error}")
                continue

            text = row.get("text")
            if not text:
                continue

            label = to_binary_label(row.get("label"))
            train_by_label[label].append({"text": text, "label": label})

    random.Random(SEED).shuffle(train_by_label[0])
    random.Random(SEED).shuffle(train_by_label[1])

    train_data = train_by_label[0][:train_per_label] + train_by_label[1][:train_per_label]
    val_data = (
        train_by_label[0][train_per_label:train_per_label + val_per_label]
        + train_by_label[1][train_per_label:train_per_label + val_per_label]
    )
    test_data = (
        train_by_label[0][train_per_label + val_per_label:train_per_label + val_per_label + test_per_label]
        + train_by_label[1][train_per_label + val_per_label:train_per_label + val_per_label + test_per_label]
    )

    random.Random(SEED).shuffle(train_data)
    random.Random(SEED).shuffle(val_data)
    random.Random(SEED).shuffle(test_data)

    output_name = "m4_multi"
    split_data = {"train": train_data, "val": val_data, "test": test_data}
    save_jsonl_splits(output_name, split_data)

    print("\nSummary of M4 multilingual splits:")
    print({split: len(data) for split, data in split_data.items()})

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

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        split_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        summary = {split: len(data) for split, data in split_data.items()}

        stem = "tsm_" + os.path.splitext(filename)[0]
        save_jsonl_splits(stem, split_data)
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

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        split_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        summary = {split: len(data) for split, data in split_data.items()}

        stem = "m4_" + os.path.splitext(filename)[0]
        save_jsonl_splits(stem, split_data)
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

        split_data = {
            "train": [final_train[i] for i in range(len(final_train))],
            "val": [final_val[i] for i in range(len(final_val))],
            "test": [final_test[i] for i in range(len(final_test))],
        }

        print(f"SUMMARY for {name}:")
        for split, data in split_data.items():
            print(f"  {split.capitalize()}: {len(data)}")
        save_jsonl_splits(name, split_data)

    build_subset(dataset, "multisocial_full")

    for lang in ["en", "de", "ar", "nl", "pt"]:
        lang_rows = dataset.filter(lambda x, lang=lang: x["language"] == lang)
        build_subset(lang_rows, f"multisocial_{lang}")


def main() -> None:
    
    
    # # M4
    # # need to halve bc one instance contains both machine and human text
    # # print("Preparing M4 data...")
    # prepare_M4_data()

    # # M4 multilingual
    # # need to halve bc one instance contains both machine and human text
    # print("Preparing M4 data...")
    # prepare_M4_multi_data()
    
    # # MULTISOCIAL
    # # print("Preparing Multisocial data...")
    # prepare_multisocial_data()

    # # TSM
    # # print("Preparing TSM task_2 data...")
    # prepare_TSM_data()
    
    # # DetectRL task_1
    # print("Preparing DetectRL task_1 data...")
    # prepare_detectrl_task_1()

    # # tsm multi
    # print("Preparing TSM multi data...")
    # prepare_tsm_multi()

    # prepare multitude
    # print("Preparing MULTITUDE data...")
    # prepare_multitude()

    # prepare editlens
    print("Preparing EditLens data...")
    prepare_editlens()


if __name__ == "__main__":
    main()
