import json
import os
import random
import re
import pandas as pd

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

M4_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4")
M4_MULTILINGUAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4_multi")
MULTISOCIAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multisocial")
TSM_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "tsm")
DETECT_RL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl" , "task_2")
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

TRAINING_N = 1500
VALIDATION_N = 500
TESTING_N = 500
SEED=42

random.seed(SEED)

def process_tsm():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    languages = ['en', "pt", "vi"]
    subsets = ["sums", "paras"]
    models = ["deepseek", "gemini", "gpt_4o_mini", "gpt4o"]

    for lang in languages:
        for subset in subsets:
            all_pos = []
            all_neg = []
            for model in models:
                if subset == "sums":
                    file_name = f"{subset}_{lang}_{model}.jsonl"
                else:
                    file_name = f"{subset}_{lang}_first_{model}.jsonl"

                # load jsonl
                with open(os.path.join(TSM_RAW_DATA_DIR, file_name), "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]

                pos = [{"text": x["mgt"], "label": 1} for x in data if x['mgt'].strip()]
                neg = [{"text": x["trgt"], "label": 0} for x in data if x['trgt'].strip()]

                all_pos.extend(pos)
                all_neg.extend(neg)
            
            # shuffle and split
            random.shuffle(all_pos)
            # rm duplicates
            print(f"Before deduplication: {len(all_neg)} negatives")
            all_neg_unique = set()
            deduplicated_neg = []
            for item in all_neg:
                if item["text"] not in all_neg_unique:
                    all_neg_unique.add(item["text"])
                    deduplicated_neg.append(item)
            all_neg = deduplicated_neg
            print(f"After deduplication: {len(all_neg)} negatives")
            all_neg = list(all_neg)
            random.shuffle(all_neg)

            data_out = {"train": [], "val": [], "test": []}
            data_out["train"].extend(all_pos[:train_per_label])
            data_out["train"].extend(all_neg[:train_per_label])
            data_out["val"].extend(all_pos[train_per_label:train_per_label + val_per_label])
            data_out["val"].extend(all_neg[train_per_label:train_per_label + val_per_label])
            data_out["test"].extend(all_pos[train_per_label + val_per_label:train_per_label + val_per_label + test_per_label])
            data_out["test"].extend(all_neg[train_per_label + val_per_label:train_per_label + val_per_label + test_per_label])

            output_dir = os.path.join(DATA_DIR, f"tsm_{subset}_{lang}")
            os.makedirs(output_dir, exist_ok=True)
            for split in ["train", "val", "test"]:
                with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                    for item in data_out[split]:
                        f.write(json.dumps(item) + "\n")


def process_detectrl():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    pattern = re.compile(r"^multi_domains_(.+)_(train|test)\.json$")
    domains = set()
    for filename in os.listdir(DETECT_RL_RAW_DATA_DIR):
        match = pattern.match(filename)
        if match:
            domain = match.group(1)
            domains.add(domain)

    label_map = {"human": 0, "llm": 1}

    for domain in sorted(domains):
        data_out = {"train": [], "val": [], "test": []}
        for subset in ["train", "test"]:
            file_name = f"multi_domains_{domain}_{subset}.json"
            with open(os.path.join(DETECT_RL_RAW_DATA_DIR, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            # exclude claude
            data = [x for x in data if x.get("llm_type") != "Claude-instant"]

            # keep only text+label keys
            pos = [{"text": x["text"], "label": label_map["llm"]} for x in data if x.get("label") == "llm" and x.get("text")]
            neg = [{"text": x["text"], "label": label_map["human"]} for x in data if x.get("label") == "human" and x.get("text")]

            random.shuffle(pos)
            random.shuffle(neg)

            if subset == "train":
                data_out["train"].extend(pos[:train_per_label])
                data_out["train"].extend(neg[:train_per_label])
                data_out["val"].extend(pos[train_per_label:train_per_label + val_per_label])
                data_out["val"].extend(neg[train_per_label:train_per_label + val_per_label])

            if subset == "test":
                data_out["test"].extend(pos[:test_per_label])
                data_out["test"].extend(neg[:test_per_label])

        # shuffle splits
        random.shuffle(data_out["train"])
        random.shuffle(data_out["val"])
        random.shuffle(data_out["test"])

        # save jsonl
        output_dir = os.path.join(DATA_DIR, f"detectrl_{domain}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")


def process_multisocial():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2
    selected_languages = ["en", "es", "de", "pl", "pt", "ar", "zh", "ru"]

    input_path = os.path.join(MULTISOCIAL_RAW_DATA_DIR, "multisocial_anonymized.csv")
    by_language = {lang: {"train": {0: [], 1: []}, 
                          "test": {0: [], 1: []}} for lang in selected_languages}

    df = pd.read_csv(input_path)
    df = df[["text", "split", "language", "label"]].copy()

    data = df.to_dict("records")
    data = [row for row in data if row["language"] in selected_languages and row["text"].strip()]
    for row in data:
        text = row["text"]
        split = row["split"]
        language = row["language"]
        label = int(row["label"])

        by_language[language][split][label].append({"text": text, "label": label})

    for language in sorted(by_language):
        train_neg = by_language[language]["train"][0]
        train_pos = by_language[language]["train"][1]
        test_neg = by_language[language]["test"][0]
        test_pos = by_language[language]["test"][1]

        random.shuffle(train_neg)
        random.shuffle(train_pos)
        random.shuffle(test_neg)
        random.shuffle(test_pos)

        data_out = {"train": [], "val": [], "test": []}
        data_out["train"].extend(train_pos[:train_per_label])
        data_out["train"].extend(train_neg[:train_per_label])
        data_out["val"].extend(train_pos[train_per_label:train_per_label + val_per_label])
        data_out["val"].extend(train_neg[train_per_label:train_per_label + val_per_label])
        data_out["test"].extend(test_pos[:test_per_label])
        data_out["test"].extend(test_neg[:test_per_label])

        random.shuffle(data_out["train"])
        random.shuffle(data_out["val"])
        random.shuffle(data_out["test"])

        output_dir = os.path.join(DATA_DIR, f"multisocial_{language}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

def main() -> None:

    # DetectRL task_2
    # print("Preparing DetectRL task_2 data...")
    # process_detectrl()
    
    # # Multisocial
    # print("Preparing Multisocial data...")
    # process_multisocial()

    # TSM
    print("Preparing TSM data...")
    process_tsm()

if __name__ == "__main__":
    main()
