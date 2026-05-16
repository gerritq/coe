from argparse import Namespace
from typing import Any
import os
import json
import torch
import numpy as np
import torch.nn as nn
from src.inference import Inference
from src.utils import metrics, OOD, return_args
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from datetime import datetime
from argparse import ArgumentParser
from transformers import set_seed
import random

BASE_DIR = os.getenv("BASE_COE")
DATA_DIR = os.path.join(BASE_DIR, "data", "sets")
OUT_DIR = os.path.join(BASE_DIR, "output", "desc")
os.makedirs(OUT_DIR, exist_ok=True)

def load_d_m4_domain_with_generators_items(args:Namespace, domain: str) -> list[dict[str, Any]]:
    
    set_seed(args.seed)
    
    path = os.path.join(DATA_DIR, "d_m4_domains_mixed_generators", "data.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    target = domain.lower()
    items = [x for x in all_items if x["source"] == target]

    assert len(items) == 2000, "Expected 2000 items for domain {}, but got {}".format(domain, len(items))
    
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]

    training_items = pos[:args.n_per_label] + neg[:args.n_per_label]
    test_items = pos[args.n_per_label:args.n_per_label+250] + neg[args.n_per_label:args.n_per_label+250] 

    random.shuffle(training_items)
    random.shuffle(test_items)

    assert len(training_items) == 2 * args.n_per_label
    assert len(test_items) == 500

    print(f"Loaded {len(training_items)} train items and {len(test_items)} test items for domain {domain}.")
    
    return {
        "train": training_items,
        "test": test_items,
    }

class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 depth: int, 
                 dropout: float = 0.1):
        super().__init__()
        layers = []

        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPProbe:
    def __init__(self, 
                 depth: int = None) -> None:
        self.model = None
        self.depth = depth

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        input_dim = x.shape[1]
        hidden_dim = max(32, min(256, input_dim // 2))
        model = MLP(input_dim=input_dim, 
                     hidden_dim=hidden_dim,
                     depth=self.depth)
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        model.eval()
        self.model = model

        return


    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


class LogisticProbe:
    def __init__(self, degree: int = 1) -> None:
        self.degree = degree
        self.pca = PCA(n_components=50, random_state=42)
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.poly_scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, random_state=42)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_pca = self.pca.fit_transform(x)
        x_poly = self.poly.fit_transform(x_pca)
        x_poly_scaled = self.poly_scaler.fit_transform(x_poly)
        self.model.fit(x_poly_scaled, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_pca = self.pca.transform(x)
        x_poly = self.poly.transform(x_pca)
        x_poly_scaled = self.poly_scaler.transform(x_poly)
        return self.model.predict_proba(x_poly_scaled)[:, 1]

class Probing:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.inference = Inference(self.args.model_name)

    def _collect_model_states(self, 
                              items: list[dict]
                              ) -> dict[str, Any]:
        
        all_hidden_states = []
        labels = []
        meta = []
        
        for item in items:
            out = self.inference.run(item=item, 
                                     args=self.args)
        
            # A HIDDEN STATES
            hidden_states = out["hidden_states"] # tuple of len layer
            hs_per_layer = []
            for l in hidden_states:
                # l: (d_model) 
                hs_per_layer.append(l.detach().to(torch.float32).cpu().numpy())
            
            # append: layer x d_model
            all_hidden_states.append(np.stack(hs_per_layer, axis=0))

            # LABELS
            labels.append(int(out["label"]))


        # STACK HS
        x = np.stack(all_hidden_states, axis=0)  # (n_samples, n_layers, d_model)
        x = np.transpose(x, (1, 0, 2))  # (n_layers, n_samples, d_model)
        y = np.asarray(labels, dtype=np.int32)   # (n_samples,)

        return {
            "hidden_x": x,
            "y": y,
            "meta": meta,
        }
    
    def _train_probe(self,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          ) -> dict[str, Any]:
        if self.args.mode == "mlp":
            scalers_by_layer = []
            models_by_layer = []

            for layer in range(x_train.shape[0]):

                # get data and scale
                x_layer_train = x_train[layer]  # (n_samples, d_model)

                # fit scaler
                scaler = StandardScaler()
                x_layer_train_scaled = scaler.fit_transform(x_layer_train)

                mlp_probe = MLPProbe(depth=self.args.complexity)
                mlp_probe.fit(x_layer_train_scaled, y_train)

                # collect
                scalers_by_layer.append(scaler)
                models_by_layer.append(mlp_probe)

            return {
                "scalers_by_layer": scalers_by_layer,
                "models_by_layer": models_by_layer,
            }

        # mode == "log"
        mid_layer = x_train.shape[0] // 2
        x_layer_train = x_train[mid_layer]  # (n_samples, d_model)
        scaler = StandardScaler()
        x_layer_train_scaled = scaler.fit_transform(x_layer_train)
        log_probe = LogisticProbe(degree=self.args.complexity)
        log_probe.fit(x_layer_train_scaled, y_train)
        return {
            "mid_layer": mid_layer,
            "scaler": scaler,
            "model": log_probe,
        }

    def _evaluate(self, 
                  train_out: dict[str, Any],
                  test: dict[str, Any]) -> dict[str, Any]:

        y_true = test["y"]

        if self.args.mode == "mlp":
            # run eval by layer
            all_projections = []
            for layer in range(len(train_out["models_by_layer"])):

                # Get scaler and model
                scaler = train_out["scalers_by_layer"][layer]
                model = train_out["models_by_layer"][layer]

                # Get test data for this layer
                x_layer_test = test["hidden_x"][layer]  # (n_samples, d_model)
                x_layer_test_scaled = scaler.transform(x_layer_test)

                x_pred = model.predict_proba(x_layer_test_scaled)

                # for small N; replace nan/and inf
                x_pred = np.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)

                all_projections.append(x_pred)

            # A aggregate projection score
            all_projections = np.stack(all_projections, axis=0)  # (n_layers, n_samples)

            # for small N; replace nan/and inf
            all_projections = np.nan_to_num(all_projections, nan=0.0, posinf=0.0, neginf=0.0)

            aggregate_projection = all_projections.mean(axis=0)  # (n_samples,)
        else:
            mid_layer = train_out["mid_layer"]
            scaler = train_out["scaler"]
            model = train_out["model"]
            x_layer_test = test["hidden_x"][mid_layer]  # (n_samples, d_model)
            x_layer_test_scaled = scaler.transform(x_layer_test)
            aggregate_projection = model.predict_proba(x_layer_test_scaled)
            aggregate_projection = np.nan_to_num(
                aggregate_projection, nan=0.0, posinf=0.0, neginf=0.0
            )

        mean_projection_metrics = metrics(
            y_true=y_true,
            y_predict=aggregate_projection,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )
        return {
            "mean_projection_metrics": mean_projection_metrics,

            }


    def run(self, args: Namespace) -> None:
        
        ALL_DOMAINS = ["wikipedia", "arxiv", "reddit", "peerread"]
        training_items = load_d_m4_domain_items(args=args,
                                                domain=args.domain)
        train = self._collect_model_states(training_items["train"])
        
        # TRAINING THE PROBE
        train_out = self._train_probe(x_train=train["hidden_x"],
                                      y_train=train["y"])
        # RUNNING EVAL
        for target_dataset in ALL_DOMAINS:
            if not args.ood:
                if args.domain != target_dataset:
                    continue
            
            print("="*60)
            print(f"    Evaluating on target dataset: {target_dataset}")
            print("="*60)

            target_items = load_d_m4_domain_items(args=args, 
                                                  domain=target_dataset)
            test = self._collect_model_states(target_items["test"])

            test_metrics = self._evaluate(train_out=train_out, test=test)
            
            # SAVE OUTPUT
            filename = (
                f"{args.mode}_{args.token_mode}_C{args.complexity}_"
                f"{args.domain}_2_{target_dataset}.json"
            )
            
            args_copy = Namespace(**vars(args))  
            out_args = return_args(args_copy)
            out_args['target_dataset'] = target_dataset
            out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            out = {'args': out_args,
                   'test_metrics': test_metrics,}
            
            with open(os.path.join(OUT_DIR, filename), "w") as f:
                json.dump(out, f, indent=4)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="mlp", choices=["mlp", "log"])
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--ood", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--complexity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_per_label", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.token_mode not in {"last_token", "pooling"}:
        raise ValueError("token_mode must be one of: last_token, pooling")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")

    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.model_name = args.model

    analyzer = Probing(args=args)
    analyzer.run(args)


if __name__ == "__main__":
    main()
