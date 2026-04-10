
import os
import json

OUT_DIR = "/scratch/prj/inf_nlg_ai_detection/tsm_data"
os.makedirs(OUT_DIR, exist_ok=True)

def process_tst():
    BASE_PATH = "/scratch/users/k21157437/neutral_new/data"
    OUT_DIR_TST = os.path.join(OUT_DIR, "tst")

    LANGUAGES = ["en", "pt", "vi"]
    MODELS = ["deepseek", "gemini", "gpt4o", "gpt", "mistral", "qwen"]
    SETTINGS = ["_default", "_paras"]

    for language in LANGUAGES:
        
        base_dir = os.path.join(BASE_PATH, language, "datasets", "mgt")

        for model in MODELS:
            for setting in SETTINGS:

                if language != "en":
                    setting = ""

                input_file = f"{language}{setting}_mgt_few5_{model}.jsonl"
                
                new_model = "gpt_4o_mini" if model == "gpt4o" else model
                new_setting = "sentence" if setting == "default" else "paragraphs"
                output_file = f"tst_{language}_{new_setting}_{new_model}.jsonl"

                input_path = os.path.join(base_dir, input_file)
                output_path = os.path.join(OUT_DIR_TST, output_file)
                os.makedirs(OUT_DIR_TST, exist_ok=True)

                if not os.path.exists(input_path):
                    print(f"[ERROR] Input path does not exist: {input_path}")
                    continue

                with open(input_path, "r", encoding="utf-8") as fin, \
                    open(output_path, "w", encoding="utf-8") as fout:

                    for line in fin:
                        data = json.loads(line)

                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                print(f"Saved: {output_path}")

def process_sums():
    BASE_PATH = "/scratch/users/k21157437/sums/data/"
    OUT_DIR_SUMS = os.path.join(OUT_DIR, "sums")
    # /scratch/users/k21157437/sums/data/en/ds/en_sums_mgt_few1_deepseek copy.jsonl

    LANGUAGES = ["en", "pt", "vi"]
    MODELS = ["deepseek", "gemini", "gpt4o", "gpt", "mistral", "qwen"]
    
    for language in LANGUAGES:
        base_dir = os.path.join(BASE_PATH, language, "ds")

        for model in MODELS:
            input_file = f"{language}_sums_mgt_few1_{model}.jsonl"
            output_file = f"sums_{language}_{model}.jsonl"

            input_path = os.path.join(base_dir, input_file)
            output_path = os.path.join(OUT_DIR_SUMS, output_file)
            os.makedirs(OUT_DIR_SUMS, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"[ERROR] Input path does not exist: {input_path}")
                continue

            with open(input_path, "r", encoding="utf-8") as fin, \
                open(output_path, "w", encoding="utf-8") as fout:

                for line in fin:
                    data = json.loads(line)

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"Saved: {output_path}")

def process_paragraphs():
    BASE_PATH = "/scratch/users/k21157437/paras/data/"
    OUT_DIR_PARAS = os.path.join(OUT_DIR, "paras")
    # /scratch/users/k21157437/paras/data/en/ds/mgt/en_paras_rag_extend_deepseek.jsonl
    # /scratch/users/k21157437/paras/data/en/ds/en_paras_rag_first_deepseek.jsonl

    LANGUAGES = ["en", "pt", "vi"]
    MODELS = ["deepseek", "gemini", "gpt4o", "gpt", "mistral", "qwen"]
    SETTINGS = ["first", "extend"]
    for language in LANGUAGES:
        base_dir = os.path.join(BASE_PATH, language, "ds", "mgt")

        for model in MODELS:
            for setting in SETTINGS:
                input_file = f"{language}_paras_rag_{setting}_{model}.jsonl"
                output_file = f"paras_{language}_{setting}_{model}.jsonl"

                input_path = os.path.join(base_dir, input_file)
                output_path = os.path.join(OUT_DIR_PARAS, output_file)
                os.makedirs(OUT_DIR_PARAS, exist_ok=True)

                print("Input", input_path)

                if not os.path.exists(input_path):
                    print(f"[ERROR] Input path does not exist: {input_path}")
                    continue

                with open(input_path, "r", encoding="utf-8") as fin, \
                    open(output_path, "w", encoding="utf-8") as fout:

                    skipped = 0
                    for line_no, line in enumerate(fin, start=1):
                        try:
                            data = json.loads(line)
                            data_clean = data.copy()
                            
                            if setting == "first":
                                keep_keys = {"id", "revid", "page_title", "section_title", "word_tertile", "trgt", "mgt"}
                                data_clean = {k: v for k, v in data.items() if k in keep_keys}

                            if setting == "extend":
                                keep_keys = {"id", "revid", "page_title", "section_title", "word_tertile", "mgt_new", "tgrt_new"}
                                data_clean['mgt_new'] = data['trgt_first'] + " " + data['mgt']
                                data_clean['tgrt_new'] = data['trgt_first'] + " " + data['trgt']
                                data_clean = {k: v for k, v in data_clean.items() if k in keep_keys}
                                #rename _new
                                data_clean['mgt'] = data_clean.pop('mgt_new')
                                data_clean['trgt'] = data_clean.pop('tgrt_new')
                                


                        except json.JSONDecodeError as error:
                            skipped += 1
                            print(f"[WARN] Skipping malformed JSON in {input_path} at line {line_no}: {os.error}")
                            continue

                        fout.write(json.dumps(data_clean, ensure_ascii=False) + "\n")
                print(f"Saved: {output_path} (skipped={skipped})")

if __name__ == "__main__":
    process_tst()
    process_sums()
    process_paragraphs()
