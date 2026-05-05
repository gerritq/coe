import json
import os
import random
import re
import pandas as pd

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

DETECT_RL_DOMAINS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl" , "task_2_domains")
DETECT_RL_ATTACKS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl" , "task_2_attacks")
MULTISOCIAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multisocial")
TSM_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "tsm")
EDITLENS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "editlens")

os.makedirs("data/raw/atp", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_N = 1500
VALIDATION_N = 500
TESTING_N = 500
SEED=42

random.seed(SEED)

def check_for_duplicates(data: dict[str, list[dict]]) -> None:
    # check for duplicates within each split
    
    print("\nDATASET SUMMARY:")
    for split in data:
        print(f"{split}: {len(data[split])} items")
    print("")
    
    for split in data:
        print(f"Checking for duplicates within {split} ...")
        seen = set()
        count = 0
        for item in data[split]:
            text = item["text"]
            if text in seen:
                count += 1
            else:
                seen.add(text)
        print(f"Found {count} duplicates in {split}")

    # check for duplicates across splits
    print("Checking for duplicates across splits ...")
    seen = set()
    count = 0
    for split in data:
        for item in data[split]:
            text = item["text"]
            if text in seen:
                count += 1
            else:
                seen.add(text)
    print(f"Found {count} duplicates across splits")


def process_atp():
    
    # SAVE DATASET TO CSV
    # from datasets import load_dataset
    # dataset = load_dataset("smksaha/apt-eval", data_files={
    #     "test": "merged_apt_eval_dataset.csv",
    #     "original": "original.csv"
    # })

    # dataset = load_dataset(
    # "smksaha/apt-eval",
    # data_files={
    #     "test": "merged_apt_eval_dataset.csv",
    #     "original": "original.csv",
    # },
    # )

    # out_dir = "data/raw/atp"
    # for split, ds in dataset.items():
    #     ds.to_csv(f"{out_dir}/{split}.csv", index=False)
    
    n_machine = 200

    
    data_out = {"train": [], "val": [], "test": []}
    
    # Load human data -> used only in training
    human_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "atp", "original.csv"))
    human_data = human_df.to_dict("records")
    random.shuffle(human_data)
    
    human_data = [{"text": x["generation"], "label": 0} for x in human_data if x["generation"].strip()]
    data_out["train"].extend(human_data)

    # Load machine-generated data
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "atp", "test.csv"))
    test_data = test_df.to_dict("records")
    test_data = [x for x in test_data if x["polish_type"] == "degree-based"]

    editing_types = set(x["polishing_degree"] for x in test_data)
    print(f"Editing types in ATP dataset: {editing_types}")

    for editing_degree in editing_types:
    
        subset = [x for x in test_data if x["polishing_degree"] == editing_degree and x["generation"].strip()]
        random.shuffle(subset)
        machine_test = [{"text": x["generation"], 
                         "label": 1,
                         "sem_similarity": x["sem_similarity"],
                         "levenshtein_distance": x["levenshtein_distance"],
                         "jaccard_distance": x["jaccard_distance"]} for x in subset]
        data_out["test"].extend(machine_test[:n_machine])

        if editing_degree == "major":
            data_out["train"].extend(machine_test[n_machine:n_machine*4])
    
    # shuffle splits
    # we dont need val
    data_out["val"] = data_out["train"]
    random.shuffle(data_out["train"])
    random.shuffle(data_out["test"])

    output_dir = os.path.join(DATA_DIR, f"atp")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_out[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_out)
    

    



def process_editlens():
    train_per_label = 1000 // 2
    val_per_label = 500 // 2
    test_per_label = 500 // 2

    # VAL
    file_path_val = os.path.join(EDITLENS_RAW_DATA_DIR, "val.csv")
    file_path_test = os.path.join(EDITLENS_RAW_DATA_DIR, "test.csv")

    # merge two pds
    ds_val = pd.read_csv(file_path_val)
    ds_test = pd.read_csv(file_path_test)

    df = pd.concat([ds_val, ds_test], ignore_index=True)
    df = df[["text", "cosine_score", "soft_ngrams_score", "split", "text_type"]].copy()
    data = df.to_dict("records")
    
    data_out = {"train": [], "val": [], "test": []}
    for subset in ['val', "test"]:
        if subset == "val":
            n = train_per_label
        else:
            n = test_per_label
        subset_data = [x for x in data if x["split"] == subset]
        pos = [x for x in subset_data if x["text_type"] == "ai_edited" and x["text"].strip()][:n]
        neg = [x for x in subset_data if x["text_type"] == "human_written" and x["text"].strip()][:n]

        data_out[subset].extend([{"text": x["text"], "label": 1} for x in pos])
        data_out[subset].extend([{"text": x["text"], "label": 0} for x in neg])

    # shuffle splits
    random.shuffle(data_out["val"])
    random.shuffle(data_out["test"])
    
    # we do not need the val split
    data_out["train"] = data_out["val"]
    
    output_dir = os.path.join(DATA_DIR, f"editlens")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_out[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_out)
    

        



def process_tsm():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    subsets = ["sums", "extend", "first", "tst"]
    models = ["deepseek", "gemini", "gpt_4o_mini", "gpt4o", "qwen"]

    for subset in subsets:
        all_pos = []
        all_neg = []
        for model in models:
            if subset == "sums":
                file_name = f"{subset}_en_{model}.jsonl"

            if subset in ["extend", "first"]:
                file_name = f"paras_en_{subset}_{model}.jsonl"

            if subset == "tst":
                file_name = f"{subset}_en_paragraphs_{model}.jsonl"
            
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

        output_dir = os.path.join(DATA_DIR, f"tsm_{subset}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

        check_for_duplicates(data_out)


def process_detectrl_attacks():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    pattern = re.compile(r"^(.+)_(train|test)\.json$")
    domains = set()
    for filename in os.listdir(DETECT_RL_ATTACKS_RAW_DATA_DIR):
        match = pattern.match(filename)
        if match:
            domain = match.group(1)
            domains.add(domain)

    label_map = {"human": 0, "llm": 1}

    for domain in sorted(domains):
        print(f"Processing attack: {domain}")
        data_out = {"train": [], "val": [], "test": []}
        for subset in ["train", "test"]:
            file_name = f"{domain}_{subset}.json"
            with open(os.path.join(DETECT_RL_ATTACKS_RAW_DATA_DIR, file_name), "r", encoding="utf-8") as f:
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
        output_dir = os.path.join(DATA_DIR, f"drlAttack_{domain}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

        check_for_duplicates(data_out)

def process_detectrl_domains():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    pattern = re.compile(r"^multi_domains_(.+)_(train|test)\.json$")
    domains = set()
    for filename in os.listdir(DETECT_RL_DOMAINS_RAW_DATA_DIR):
        match = pattern.match(filename)
        if match:
            domain = match.group(1)
            domains.add(domain)

    label_map = {"human": 0, "llm": 1}

    for domain in sorted(domains):
        print(f"Processing domain: {domain}")
        data_out = {"train": [], "val": [], "test": []}
        for subset in ["train", "test"]:
            file_name = f"multi_domains_{domain}_{subset}.json"
            with open(os.path.join(DETECT_RL_DOMAINS_RAW_DATA_DIR, file_name), "r", encoding="utf-8") as f:
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
        output_dir = os.path.join(DATA_DIR, f"drlDomain_{domain}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

        check_for_duplicates(data_out)

    return domains

def process_multisocial():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2
    selected_languages = ["en", "de", "zh", "ru"]

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
        print(f"Processing language: {language}")
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

        check_for_duplicates(data_out)


def cross_benchmark_datasets():

    n_train_per_subset = 250 // 2
    n_val_per_subset = 125 // 2
    n_test_per_subset = 125 // 2

    datasets = ["drlDomain", "multisocial", "tsm"]

    for ds in datasets:
        
        print(f"Processing dataset: {ds}")
        # find all ds sets
        if ds == "drlDomain":
            sets = sorted(["drlDomain_" + x for x in ["arxiv", "writing_prompt", "xsum", "yelp_review"]])
        else:
            sets = sorted([d for d in os.listdir(DATA_DIR) if d.startswith(ds)])

        print(f"Processing dataset: {sets}")
             

        train_split = []
        val_split = []
        test_split = []

        for set_name in sets:
            file_path = os.path.join(DATA_DIR, set_name, f"train.jsonl")
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            

            pos = [x for x in data if x["label"] == 1]
            neg = [x for x in data if x["label"] == 0]

            train_split.extend(pos[:n_train_per_subset])
            train_split.extend(neg[:n_train_per_subset])

            val_split.extend(pos[n_train_per_subset:n_train_per_subset + n_val_per_subset])
            val_split.extend(neg[n_train_per_subset:n_train_per_subset + n_val_per_subset])

            test_split.extend(pos[n_train_per_subset + n_val_per_subset:n_train_per_subset + n_val_per_subset + n_test_per_subset])
            test_split.extend(neg[n_train_per_subset + n_val_per_subset:n_train_per_subset + n_val_per_subset + n_test_per_subset])

        
        
        random.shuffle(train_split)
        random.shuffle(val_split)
        random.shuffle(test_split)

        data_out = {"train": train_split, "val": val_split, "test": test_split}

        output_dir = os.path.join(DATA_DIR, f"CB_{ds}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

        check_for_duplicates(data_out)

def main() -> None:

    # DetectRL task_2 - domains
    print("="*60)
    print("="*60)
    print("DETECTRL - task_2 DOMAINS...")
    process_detectrl_domains()

    # DetectRL task_2 - attacks
    print("="*60)
    print("="*60)
    print("DETECTRL - task_2 ATTACKS...")
    process_detectrl_attacks()
    
    # # Multisocial
    # print("Preparing Multisocial data...")
    # process_multisocial()

    # TSM
    # print("Preparing TSM data...")
    # process_tsm()


    # CROSS BENCHMARK DATASETS
    print("="*60)
    print("="*60)
    print("Preparing CROSS BENCHMARK data...")
    cross_benchmark_datasets()

    # EDITLENS DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing EDITLENS data...")
    # process_editlens()

    # ATP DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing ATP data...")
    # process_atp()


if __name__ == "__main__":
    main()
