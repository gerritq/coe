import os
import json
import os
import numpy as np
import torch

from argparse import ArgumentParser, Namespace
from datasets import Dataset
from fuzzywuzzy import fuzz

import json
import matplotlib.pyplot as plt

from fuzzywuzzy import fuzz


from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          set_seed)

from src.utils import (return_device, 
                       metrics,
                       return_args)

BASE_DIR = os.getenv("BASE_COE")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
os.makedirs(BASELINE_DIR, exist_ok=True)


class RAIDAR:
    def __init__(self, args):
        self.args = args
        self.model_path = args.base_model_1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way'][:2]
        
        self.ngram_num = 4
        self.cutoff_start = 0
        self.cutoff_end = 6000000

        self.device = return_device()

    def GPT_self_prompt(self, prompt_str, content_to_be_detected):
        prompts = f"{prompt_str}: \"{content_to_be_detected}\""
        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        model_inputs.pop("token_type_ids", None)

        output = self.model.generate(**model_inputs, max_new_tokens=len(self.tokenize_and_normalize(prompts)))

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print('length', len(self.tokenize_and_normalize(prompts)), len(prompts))
        print(decoded_output)

        return decoded_output

    def rewrite_json(self, texts, prompt_list, human=False):
        all_data = []
        for i, data in enumerate(texts):
            # print(cc, len(input_json))
            tmp_dict ={}

            tmp_dict = data.copy()

            for ep in prompt_list:
                tmp_dict[ep] = self.GPT_self_prompt(ep, tmp_dict['input'])
            
            all_data.append(tmp_dict)

            if i % 50 == 0:
                print(f"Processed {i} samples")

        return all_data

    def rewrite_json_batch(self, texts, prompt_list, batch_size: int = 16):
        all_data = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_data = [x.copy() for x in batch]
            input_texts = [x["input"] for x in batch_data]

            for ep in prompt_list:
                prompt_batch = [f"{ep}: \"{txt}\"" for txt in input_texts]
                model_inputs = self.tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                model_inputs.pop("token_type_ids", None)

                max_new_tokens = max(
                    1,
                    max(len(self.tokenize_and_normalize(p)) for p in prompt_batch),
                )

                with torch.no_grad():
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                    )

                decoded_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for item, decoded in zip(batch_data, decoded_batch):
                    item[ep] = decoded

            all_data.extend(batch_data)
            print(f"Processed {min(start + len(batch), len(texts))} samples")

        return all_data
    
    def tokenize_and_normalize(self, sentence):
        # Tokenization and normalization
        return [word.lower().strip() for word in sentence.split()]

    def extract_ngrams(self, tokens, n):
        # Extract n-grams from the list of tokens
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def common_elements(self, list1, list2):
        # Find common elements between two lists
        return set(list1) & set(list2)

    def calculate_sentence_common(self, sentence1, sentence2):
        tokens1 = self.tokenize_and_normalize(sentence1)
        tokens2 = self.tokenize_and_normalize(sentence2)

        # Find common words
        common_words = self.common_elements(tokens1, tokens2)

        # Find common n-grams (let's say up to 3-grams for this example)
        common_ngrams = set()
        

        number_common_hierarchy = [len(list(common_words))]

        for n in range(2, 5):  # 2-grams to 3-grams
            ngrams1 = self.extract_ngrams(tokens1, n)
            ngrams2 = self.extract_ngrams(tokens2, n)
            common_ngrams = self.common_elements(ngrams1, ngrams2)
            number_common_hierarchy.append(len(list(common_ngrams)))

        return number_common_hierarchy

    def sum_for_list(self, a, b):
        return [aa+bb for aa, bb in zip(a,b)]

    def get_data_stat(self, data_json):
        total_len = len(data_json)
        for idxx, each in enumerate(data_json):
            
            original = each['input']

            # remove too short ones
            
            # import pdb; pdb.set_trace()
            raw = self.tokenize_and_normalize(each['input'])
            if len(raw)<self.cutoff_start or len(raw)>self.cutoff_end:
                continue
            else:
                print(idxx, total_len)

            statistic_res = {}
            ratio_fzwz = {}
            all_statistic_res = [0 for i in range(self.ngram_num)]
            cnt = 0
            whole_combined=''
            for pp in each.keys(): 
                if pp != 'common_features':
                
                    whole_combined += (' ' + each[pp])
                    

                    res = self.calculate_sentence_common(original, each[pp])
                    statistic_res[pp] = res
                    all_statistic_res = self.sum_for_list(all_statistic_res, res)

                    ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                    cnt += 1
            
            each['fzwz_features'] = ratio_fzwz
            each['common_features'] = statistic_res
            each['avg_common_features'] = [a/cnt for a in all_statistic_res]

            each['common_features_ori_vs_allcombined'] = self.calculate_sentence_common(original, whole_combined)

            if idxx == 400:
                break

        return data_json


    def xgboost_classifier(self, gpt, human, gpt_ada, gpt_davinci, gpt4, llama, ood_data):

        def get_feature_vec(input_json):
            all_list = []
            for idxx, each in enumerate(input_json):
                
                try:
                    raw = self.tokenize_and_normalize(each['input'])
                    r_len = len(raw)*1.0
                except:
                    import pdb; pdb.set_trace()
                each_data_fea  = []

                if r_len ==0:
                    continue
                if len(raw)<self.cutoff_start or len(raw)>self.cutoff_end:
                    continue

                # each_data_fea  = [len(raw) / 100.]
                
                each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
                for ek in each['common_features'].keys():
                    each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
                
                each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

                for ek in each['fzwz_features'].keys():
                    each_data_fea.extend(each['fzwz_features'][ek])

                all_list.append(np.array(each_data_fea))

                if idxx == 400:
                    break


            all_list = np.vstack(all_list)

            return all_list
        
        machine_all = get_feature_vec(gpt_davinci)
        human_all = get_feature_vec(human)

        X_train = np.concatenate((human_all, machine_all), axis=0)
        y_train = np.concatenate((np.zeros(human_all.shape[0]), np.ones(machine_all.shape[0])), axis=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
        clf.fit(X_train, y_train)


        # Eval
        for ds in ood_data:

            machine_all = get_feature_vec(ds['ood_machine_stat'])
            human_all = get_feature_vec(ds['ood_human_stat'])

            X_ood = np.concatenate((human_all, machine_all), axis=0)
            X_ood = scaler.transform(X_ood)
            y_ood = np.concatenate((np.zeros(human_all.shape[0]), np.ones(machine_all.shape[0])), axis=0)

            y_predict_probs = clf.predict_proba(X_ood)[:, 1]

            metrics_res = metrics(y_true=y_ood,
                                  y_predict=y_predict_probs,
                                  f1_threshold=0.5,
                                  acc_threshold=0.5)

            file_name = f"{self.args.model}_{self.args.dataset}_2_{ds['name']}.json"
            out = {"args": return_args(self.args), "metrics": metrics_res}
            with open(os.path.join(BASELINE_DIR, file_name), "w") as f:
                json.dump(out, f, indent=2)

        
        # y_pred = clf.predict(X_test)

        # from sklearn.metrics import accuracy_score, classification_report, f1_score

        # print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score", f1_score(y_test, y_pred))
        # print(classification_report(y_test, y_pred))


    def run(self, 
            args: Namespace, 
            training_data: Dataset,
            ood_data: list[Dataset]
            ) -> None:
        
        set_seed(42)

        # TRAINING DATA
        machine, human = [], []
        for x in training_data['train']:
            if x['label'] == 1:
                machine.append({'input': x['text']})
            else:
                human.append({'input': x['text']})

        machine = self.rewrite_json_batch(machine, self.prompt_list)
        human = self.rewrite_json_batch(human, self.prompt_list)
            
        # get stats
        machine_stat = self.get_data_stat(machine)
        human_stat = self.get_data_stat(human)

        # TEST DATA
        ood_data_stat = []
        for ood in ood_data:
            ood_machine, ood_human = [], []
            for x in ood['data']['test']:
                if x['label'] == 1:
                    ood_machine.append({'input': x['text']})
                else:
                    ood_human.append({'input': x['text']})
            
            # rewrite ood
            ood_machine = self.rewrite_json_batch(ood_machine, self.prompt_list)
            ood_human = self.rewrite_json_batch(ood_human, self.prompt_list)
            
            # get stats
            ood['ood_machine_stat']  = self.get_data_stat(ood_machine)
            ood['ood_human_stat']  = self.get_data_stat(ood_human)
            
            ood_data_stat.append(ood)

        # train classifier
        self.xgboost_classifier(None, human_stat, None, machine_stat, None, None, ood_data_stat)

        print("done")
    
