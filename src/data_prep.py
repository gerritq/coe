import json
import ast
import os
import random
import re
import pandas as pd
from utils import SimCalculator

BASE_DIR = os.getenv("BASE_COE")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")

DETECT_RL_DOMAINS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl" , "task_2_domains")
DETECT_RL_ATTACKS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "detectrl" , "task_2_attacks")
MULTISOCIAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "multisocial")
TSM_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "tsm")
M4_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "m4_multi")
BEEMO_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "beemo")
APT_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "apt")
EDITLENS_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "editlens")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BEEMO_RAW_DATA_DIR, exist_ok=True)
os.makedirs(APT_RAW_DATA_DIR, exist_ok=True)
os.makedirs(EDITLENS_RAW_DATA_DIR, exist_ok=True)

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
        labels = set([x["label"] for x in data[split]])
        print(f"  Labels: {sorted(labels)}")
        for label in sorted(labels):
            count = sum(1 for x in data[split] if x["label"] == label)
            print(f"    Label {label}: {count} items")
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

def process_editlens():
    # SAVE DATASET TO CSV
    # from datasets import load_dataset
    # ds = load_dataset("pangram/editlens_iclr")

    # os.makedirs(EDITLENS_RAW_DATA_DIR, exist_ok=True)
    # for split, subset in ds.items():
    #     subset.to_csv(f"{EDITLENS_RAW_DATA_DIR}/{split}.csv", index=False)

    similarity_calculator = SimCalculator()

    training_size = 500 # 500 per labels
    test_size = 500 # 500 machine in total

    # load data
    data = pd.read_csv(os.path.join(EDITLENS_RAW_DATA_DIR, "train.csv"))
    data = data.to_dict("records")

    random.shuffle(data)
    
    # TRAIN SET
    # Create the training dataset from non-edited text
    train = []
    human_train = [x for x in data if x['text_type'] == "human_written"]
    machine_train = [x for x in data if x['text_type'] == "ai_generated"]
    
    train.extend([{"text_id": x['text_id'], "text": x['text'], "label": 0} for x in human_train[:training_size]])
    train.extend([{"text_id": x['text_id'], "text": x['text'], "label": 1} for x in machine_train[:training_size]])

    random.shuffle(train)

    # TEST SET
    # Approach try random sample
    test = []
    subset = [x for x in data if x['text_type'] == "ai_edited"]
    random.shuffle(subset)

    subset = subset[:test_size]

    # add sim score
    test = []
    for item in subset:
        human = item["source_text"].strip()
        machine = item["text"].strip()

        sim = similarity_calculator.cal_similarity(human, machine)

        test.append({"text_id": item["text_id"], 
                     "text": machine, 
                     "label": 1,
                     "sem_similarity": sim['sem_similarity'],
                     "levenshtein_distance": sim['levenshtein_distance'],
                     "jaccard_distance": sim['jaccard_distance']})
        
    # we do not need val
    # we do not need train
    data_out = {"train": train, "val": train, "test": test}

    output_dir = os.path.join(DATA_DIR, f"editlens")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_out[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_out)

def process_beemo():
    # SAVE DATASET TO CSV
    # from datasets import load_dataset
    # ds = load_dataset("toloka/beemo")

    # os.makedirs(BEEMO_RAW_DATA_DIR, exist_ok=True)
    # for split, subset in ds.items():
    #     subset.to_csv(f"{BEEMO_RAW_DATA_DIR}/{split}.csv", index=False)

    similarity_calculator = SimCalculator()
    
    training_size = 500 # 500 per labels
    test_size = 500 # 500 machine in total
    
    data = pd.read_csv(os.path.join(BEEMO_RAW_DATA_DIR, "train.csv"))
    data = data.to_dict("records")

    random.shuffle(data)
    
    # TRAIN SET
    # Create the training dataset from non-edited text
    train = []
    for i in range(training_size):
        human = data[i]['human_output']
        machine = data[i]['model_output']

        train.append({"text": human, "label": 0})
        train.append({"text": machine, "label": 1})

    random.shuffle(train)

    remaining = data[training_size:]

    # TEST SET - Human edits
    test_human_edits = []
    for i in range(test_size):
        human = remaining[i]['human_output'].strip()
        machine = remaining[i]['human_edits'].strip() # machine edited by human

        sim = similarity_calculator.cal_similarity(human, machine)

        test_human_edits.append({"text": machine, 
                                 "label": 1,
                                 "sem_similarity": sim['sem_similarity'],
                                 "levenshtein_distance": sim['levenshtein_distance'],
                                 "jaccard_distance": sim['jaccard_distance']})
        
    # we do not need train
    data_human_edits = {"train": train, "val": train, "test": test_human_edits}

    output_dir = os.path.join(DATA_DIR, f"beemo_human_edits")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_human_edits[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_human_edits)

    # TEST SET - Machine edits
    test_machine_edits = []
    for i in range(test_size):
        human = remaining[i]['human_output'].strip()
        machine = ast.literal_eval(remaining[i]['gpt-4o_edits'])
        machine = machine[0]["P1"].strip()
        
        sim = similarity_calculator.cal_similarity(human, machine)

        test_machine_edits.append({"text": machine, 
                                   "label": 1,
                                   "sem_similarity": sim['sem_similarity'],
                                   "levenshtein_distance": sim['levenshtein_distance'],
                                   "jaccard_distance": sim['jaccard_distance']})

    # we do not need train
    data_machine_edits = {"train": train, "val": train, "test": test_machine_edits}

    output_dir = os.path.join(DATA_DIR, f"beemo_machine_edits")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_machine_edits[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_machine_edits)


def process_d_M4():
    """This is M4 for descriptives, 
    """
    n_per_label = 2000 // 2 

    # load data
    file_path = os.path.join(M4_RAW_DATA_DIR, f"SubtaskB.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # check generators
    # all_generators = set(x["model"] for x in data)
    # print(all_generators)

    # fixing dolly in case there is a naming issue
    for item in data:
            if item["model"] == "dolly-v2-12b":
                item["model"] = "dolly"
                
    # check domains
    all_domains = set(x["source"] for x in data)
    print(all_domains)

    generators = ["cohere", "gpt4", "bloomz", "dolly"]
    domains = ["wikipedia", "arxiv", "reddit", "peerread"]

    fix_generator = "gpt4"
    fix_domain = "wikipedia"

    # GENERATOR SPLIT
    generator_data = []
    for idx, generator in enumerate(generators, start=1):
        subset = [x for x in data if x["model"] == generator and x["source"] == fix_domain and x["text"].strip()]

        pos = [{"text": x["text"], "label": idx, "source": x["source"], "model": x["model"]} for x in subset if x["model"] == generator]
        generator_data.extend(pos[:n_per_label])

    neg = [{"text": x["text"], "label": 0, "source": fix_domain, "model": "human"} for x in data if x["model"] == "human" and x["source"] == fix_domain and x["text"].strip()]
    generator_data.extend(neg[:n_per_label])

    random.shuffle(generator_data)
    
    output_dir = os.path.join(DATA_DIR, f"d_m4_generators")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"data.jsonl"), "w", encoding="utf-8") as f:
        for item in generator_data:
            f.write(json.dumps(item) + "\n")

    check_for_duplicates({"train": generator_data, "val": [], "test": []})

    # DOMAIN SPLIT
    domain_data = []
    for domain in domains:
        
        pos = [{"text": x["text"], "label": 1, "source": domain, "model": fix_generator} for x in data if x["source"] == domain and x["model"] == fix_generator and x["text"].strip()]
        neg = [{"text": x["text"], "label": 0, "source": domain, "model": "human"} for x in data if x["source"] == domain and x["model"] == "human" and x["text"].strip()]

        domain_data.extend(pos[:n_per_label])
        domain_data.extend(neg[:n_per_label])

        
    output_dir = os.path.join(DATA_DIR, f"d_m4_domains")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"data.jsonl"), "w", encoding="utf-8") as f:
        for item in domain_data:
            f.write(json.dumps(item) + "\n")

    check_for_duplicates({"train": domain_data, "val": [], "test": []})


def process_m4():
    train_per_label = TRAINING_N // 2
    val_per_label = VALIDATION_N // 2
    test_per_label = TESTING_N // 2

    # load data
    file_path = os.path.join(M4_RAW_DATA_DIR, f"SubtaskB.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # check generators
    # all_generators = set(x["model"] for x in data)
    # print(all_generators)

    # correct dolly to 'dolly-v2-12b'
    for item in data:
        if item["model"] == "dolly-v2-12b":
            item["model"] = "dolly"
    
    generators = ["cohere", "gpt4", 'dolly', "bloomz"]

    for generator in generators:

        print(f"Processing generator: {generator}")

        subset = [x for x in data if x["model"] in [generator, 'human'] and x["text"].strip()]
        pos = [{"text": x["text"], "label": 1} for x in subset if x["model"] == generator]
        neg = [{"text": x["text"], "label": 0} for x in subset if x["model"] == "human"]

        # suffle ensure mix of domains
        random.shuffle(pos)
        random.shuffle(neg)

        train_split = []
        val_split = []
        test_split = []
        
        train_split.extend(pos[:train_per_label])
        train_split.extend(neg[:train_per_label])

        val_split.extend(pos[train_per_label:train_per_label + val_per_label])
        val_split.extend(neg[train_per_label:train_per_label + val_per_label])

        test_split.extend(pos[train_per_label + val_per_label:train_per_label + val_per_label + test_per_label])
        test_split.extend(neg[train_per_label + val_per_label:train_per_label + val_per_label + test_per_label])

        data_out = {"train": train_split, "val": val_split, "test": test_split}

        output_dir = os.path.join(DATA_DIR, f"m4_{generator}")
        os.makedirs(output_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for item in data_out[split]:
                    f.write(json.dumps(item) + "\n")

        check_for_duplicates(data_out)


# {"text":"We consider a system of many polymers in solution that interact via an external force that is applied to each pair of polymers. We study the statistical equilibrium of this system, and find that the polymers form clusters whose sizes are given by a power law distribution. This is in contrast to the traditional picture of polymers in solution, where the thermodynamic equilibrium is described by a mean-field theory based on the solution of the mean-field Boltzmann equation. We show that this difference is due to a breakdown of the assumptions that were used to derive the mean-field theory. In particular, we show that the polymer-polymer interactions in the system considered are non-local, and are thus not described by the mean-field theory. We then derive a new theory for the statistical equilibrium in the presence of an external force, which includes a correction to the mean-field theory. The new theory predicts that the polymer clusters become less dense as the external force increases, in clear contrast to the predictions of the mean-field theory. We analyze this disagreement, and show that it is due to the fact that the mean-field theory predicts a non-monotonic dependence of the polymer-polymer interaction strength on the external force, while our theory predicts a strictly monotonic dependence. We then consider the limit of our theory as the number of polymers in the system tends to infinity, and show that it describes a model of polymer quantum mechanics in a Box, which is a system with a large number of infinitely-lived polymers that interact via a non-local potential, and are in statistical equilibrium in the presence of an external force. We analyze this model, and show that it describes a system with anomalous diffusion and ballistic transport, which is analogous to the anomalous behavior observed in recent experiments on pedestrians.","label":2,"source":"arxiv","model":"cohere","arxiv_0":"test","wikipedia_0":"train","wikihow_0":"train","reddit_0":"train","peerread_0":"train","outfox_0":"train","no_0":"train","arxiv_10":"test","wikipedia_10":"train","wikihow_10":"train","reddit_10":"train","peerread_10":"train","outfox_10":"train","no_10":"train","arxiv_30":"test","wikipedia_30":"train","wikihow_30":"train","reddit_30":"train","peerread_30":"valid","outfox_30":"train","no_30":"train","arxiv_55":"test","wikipedia_55":"train","wikihow_55":"train","reddit_55":"train","peerread_55":"train","outfox_55":"train","no_55":"test","arxiv_75":"test","wikipedia_75":"train","wikihow_75":"train","reddit_75":"train","peerread_75":"train","outfox_75":"train","no_75":"test"}

def process_apt():
    
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

    # for split, ds in dataset.items():
    #     ds.to_csv(f"{APT_RAW_DATA_DIR}/{split}.csv", index=False)
    
    n_machine = 200

    
    data_out = {"train": [], "val": [], "test": []}
    
    # Load human data -> used only in training
    human_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "apt", "original.csv"))
    human_data = human_df.to_dict("records")
    random.shuffle(human_data)
    
    human_data = [{"text": x["generation"], "label": 0} for x in human_data if x["generation"].strip()]
    data_out["train"].extend(human_data)

    print(f"Loaded {len(human_data)} human-written examples for training.")

    # Load machine-generated data
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "apt", "test.csv"))
    test_data = test_df.to_dict("records")
    test_data = [x for x in test_data if x["polish_type"] == "degree-based"]

    editing_types = set(x["polishing_degree"] for x in test_data)
    print(f"Editing types in apt dataset: {editing_types}")

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
    random.shuffle(data_out["train"])
    random.shuffle(data_out["test"])

    # do not need val
    data_out["val"] = data_out["train"]

    output_dir = os.path.join(DATA_DIR, f"apt")
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for item in data_out[split]:
                f.write(json.dumps(item) + "\n")

    check_for_duplicates(data_out)

def process_apt_with_m4_train():
    
    n_apt_per_editing_type = 200
    n_train=1500

    data_out = {"train": [], "val": [], "test": []}
    
    # A TEST SET ONLY APT
    # Load machine-generated data
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "apt", "test.csv"))
    test_data = test_df.to_dict("records")
    test_data = [x for x in test_data if x["polish_type"] == "degree-based"]

    editing_types = set(x["polishing_degree"] for x in test_data)
    print(f"Editing types in apt dataset: {editing_types}")

    for editing_degree in editing_types:    
        subset = [x for x in test_data if x["polishing_degree"] == editing_degree and x["generation"].strip()]
        random.shuffle(subset)
        machine_test = [{"text": x["generation"], 
                         "label": 1,
                         "sem_similarity": x["sem_similarity"],
                         "levenshtein_distance": x["levenshtein_distance"],
                         "jaccard_distance": x["jaccard_distance"]} for x in subset]
        data_out["test"].extend(machine_test[:n_apt_per_editing_type])

    # A TRAIN BASED ON M4
    file_path = os.path.join(M4_RAW_DATA_DIR, f"SubtaskB.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    human = [{"text": x["text"], "label": 0} for x in data if x["model"] == "human" and x["text"].strip()]
    machine = [{"text": x["text"], "label": 1} for x in data if x["model"] != "human" and x["text"].strip()]

    random.shuffle(human)
    random.shuffle(machine)

    data_out["train"].extend(human[:n_train//2])
    data_out["train"].extend(machine[:n_train//2])

    # shuffle splits
    random.shuffle(data_out["train"])
    random.shuffle(data_out["test"])

    # we do not need val    
    data_out["val"] = data_out["train"]

    output_dir = os.path.join(DATA_DIR, f"apt_m4_train")
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

    datasets = ["drlDomain", "multisocial", "tsm", "m4"]

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
    # print("="*60)
    # print("="*60)
    # print("DETECTRL - task_2 DOMAINS...")
    # process_detectrl_domains()

    # DetectRL task_2 - attacks
    # print("="*60)
    # print("="*60)
    # print("DETECTRL - task_2 ATTACKS...")
    # process_detectrl_attacks()
    
    # # Multisocial
    # print("Preparing Multisocial data...")
    # process_multisocial()

    # TSM
    # print("Preparing TSM data...")
    # process_tsm()


    # CROSS BENCHMARK DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing CROSS BENCHMARK data...")
    # cross_benchmark_datasets()

    # EDITLENS DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing EDITLENS data...")
    # process_editlens()

    # apt DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing apt data...")
    # process_apt()

    # apt DATASETS with M4 train
    # print("="*60)
    # print("="*60)
    # print("Preparing apt data with m4 train...")
    # process_apt_with_m4_train()

    # M4 - generators
    # print("="*60)
    # print("="*60)
    # print("M4 - generators...")
    # process_m4()

    # D M4
    # print("="*60)
    # print("="*60)
    # print("D M4 - generators, domains, languages...")
    # process_d_M4()

    # beemo DATASETS
    # print("="*60)
    # print("="*60)
    # print("Preparing beemo data...")
    # process_beemo()

    # editlens DATASETS
    print("="*60)
    print("="*60)
    print("Preparing editlens data...")
    process_editlens()

if __name__ == "__main__":
    main()
