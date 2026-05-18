import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import RobertaTokenizer, RobertaModel

from tqdm import tqdm

from IntrinsicDim import PHD

import os
from datetime import datetime
from argparse import Namespace
from datasets import Dataset
import json

from transformers import (set_seed)
from src.utils import metrics, return_device, return_args,metrics


BASE_DIR = os.getenv("BASE_COE")

class IDEstimator:
    def __init__(self):
        self.model_path = 'FacebookAI/xlm-roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        self.model = RobertaModel.from_pretrained(self.model_path)

        self.MIN_SUBSAMPLE = 40 
        self.INTERMEDIATE_POINTS = 7

    def preprocess_text(self, text):
        return text.replace('\n', ' ').replace('  ', ' ')

    def get_phd_single(self, text, solver):
        inputs = self.tokenizer(self.preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.model(**inputs)
        
        # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
        mx_points = inputs['input_ids'].shape[1] - 2

        
        mn_points = self.MIN_SUBSAMPLE
        step = ( mx_points - mn_points ) // self.INTERMEDIATE_POINTS
            
        return solver.fit_transform(outp[0][0].numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, \
                                    point_jump=step)
    
    def get_phd(self, items: list[str], alpha=1.0):
        dims = []
        PHD_solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
        for x in tqdm(items):
            dims.append(self.get_phd_single(x, PHD_solver))

        return np.array(dims).reshape(-1, 1)
    
    def learn_logistic_regression(self, 
                                 x: dict, 
                                 y: list[int]
                                 ) -> None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_scaled, y)

        self.scaler = scaler
        self.classifier = classifier

        return None

    def run(self, 
            args: Namespace, 
            training_data: Dataset,
            ood_data: list[Dataset]
            ) -> None:
        
        self.out_dir = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
        os.makedirs(self.out_dir, exist_ok=True)

        set_seed(args.seed)

        # Get PHD features for training
        x = training_data['train']["text"]
        y = training_data['train']["label"]
        X_phd = self.get_phd(x)

        # Train logistic regression on PHD features
        self.learn_logistic_regression(X_phd, y)

        for ood_dict in ood_data:
            ood_name = ood_dict['name']
            ood_dataset = ood_dict['dataset']

            x_ood = ood_dataset["text"]
            y_ood = ood_dataset["label"]

            X_ood_phd = self.get_phd(x_ood)
            X_ood_scaled = self.scaler.transform(X_ood_phd)

            y_scores = self.classifier.predict_proba(X_ood_scaled)[:, 1]

            metrics_res = metrics(y_true=y_ood,
                                  y_predict=y_scores,
                                  f1_threshold=0.5,
                                  acc_threshold=0.5)

            args_copy = Namespace(**vars(args))  
            out_args = return_args(args_copy)
            out_args['target_dataset'] = ood_name
            out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            file_name = f"{args.model}_{args.dataset}_2_{ood_name}.json"

            out = {"args": out_args, 
                   "metrics": metrics_res}
            with open(os.path.join(self.out_dir, file_name), "w") as f:
                json.dump(out, f, indent=2)
    