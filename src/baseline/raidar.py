import os
import json
import os
import numpy as np

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']  
        
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
                tmp_dict[ep] = self.GPT_self_prompt(ep, tmp_dict['text'])
            
            all_data.append(tmp_dict)

            if i % 50 == 0:
                print(f"Processed {i} samples")

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
        
        gpt_davinci_all = get_feature_vec(gpt_davinci)
        # gpt_ada_all = get_feature_vec(gpt_ada)
        # gpt_all = get_feature_vec(gpt)
        human_all = get_feature_vec(human)

        # gpt4_all = get_feature_vec(gpt4)
        # llama_all = get_feature_vec(llama)
        # import pdb; pdb.set_trace() # dim 112,28 

        # # random split, may have content similarity   
        # gpt_all = np.concatenate((gpt_all, gpt_all), axis=0) 

        # ### Original
        # X = np.concatenate((gpt_all, human_all), axis=0)
        # Y = np.concatenate((np.ones(gpt_all.shape[0]), np.zeros(human_all.shape[0])), axis=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


        # reblanced
        # g_train, g_test, yg_train, yg_test = train_test_split(gpt_all, np.ones(gpt_all.shape[0]), test_size=0.2, random_state=42)
        h_train, h_test, yh_train, yh_test = train_test_split(human_all, np.zeros(human_all.shape[0]), test_size=0.2, random_state=42)

        # ada_g_train, ada_g_test, ada_yg_train, ada_yg_test = train_test_split(gpt_ada_all, np.ones(gpt_ada_all.shape[0]), test_size=0.2, random_state=42)
        davinci_g_train, davinci_g_test, davinci_yg_train, davinci_yg_test = train_test_split(gpt_davinci_all, np.ones(gpt_davinci_all.shape[0]), test_size=0.2, random_state=42)

        # g4_train, g4_test, yg4_train, yg4_test = train_test_split(gpt4_all, np.ones(gpt4_all.shape[0]), test_size=0.2, random_state=42)
        # llama_g_train, llama_g_test, llama_yg_train, llama_yg_test = train_test_split(llama_all, np.ones(llama_all.shape[0]), test_size=0.2, random_state=42)


        X_train = np.concatenate((davinci_g_train, h_train), axis=0)
        y_train = np.concatenate((davinci_yg_train, yh_train), axis=0)

        X_test = np.concatenate((davinci_g_test, h_test), axis=0)
        y_test = np.concatenate((davinci_yg_test, yh_test), axis=0)


        # # Neural network
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 75.83, using fuzzywazzy, get 78.5% acc.

        clf.fit(X_train, y_train)
        
        for ds in ood_data:
            ood_all = get_feature_vec(ds['stats'])
            ood_all = scaler.transform(ood_all)
            # ood_pred = clf.predict(ood_all)
            
            y_predict_probs = clf.predict_proba(ood_all)[:, 1]

            metrics_res = metrics(y_true=ds['test']["label"],
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

        # Rewrite training data
        train = []
        for x in training_data['train']:
            train.append({'input': x['text'], 'label': x['label']})
        train_rewritten = self.rewrite_json(train, self.prompt_list)
        machine = [x for x in train_rewritten if x['label'] == 1]
        human = [x for x in train_rewritten if x['label'] == 0]
            
        # get stats
        machine_stat = self.get_data_stat(machine)
        human_stat = self.get_data_stat(human)

        ood_data_stat = []
        for ood in ood_data:
            ood['stats']  = self.get_data_stat(ood['data']['test'])
            ood_data_stat.append(ood)

        # train classifier
        self.xgboost_classifier(None, human_stat, None, machine_stat, None, None, ood_data_stat)

        print("done")
    