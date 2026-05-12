from argparse import Namespace
from typing import Any
import os
import json
import torch
import numpy as np
import torch.nn as nn
from src.inference import Inference
from src.utils import load_dataset, optimal_thresholds, metrics, OOD, return_args
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from argparse import ArgumentParser

BASE_DIR = os.getenv("BASE_COE")

class MLPProbe(nn.Module):
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



class Probe:
    def __init__(self, 
                 kind: str, 
                 c: float, 
                 depth: int = None) -> None:
        self.kind = kind
        self.c = c
        self.model = None
        self.depth = depth

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.kind == "mlp":
            input_dim = x.shape[1]
            hidden_dim = max(32, min(256, input_dim // 2))
            model = MLPProbe(input_dim=input_dim, 
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

        model = LogisticRegression(max_iter=2000, random_state=42, C=self.c)
        model.fit(x, y)
        self.model = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "mlp":
            x_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(x_t)
                probs = torch.sigmoid(logits).cpu().numpy()
            return probs

        return self.model.predict_proba(x)[:, 1]

class Probing:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.out_dir = os.path.join(BASE_DIR, "output", "probe", args.folder)
        os.makedirs(self.out_dir, exist_ok=True)

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
    

    def _train_meta_probe(self, 
                          x_hidden_train: np.ndarray, 
                          x_attn_train: np.ndarray, 
                          y_train: np.ndarray) -> dict[str, Any]:
        
        scalers_by_layer = []
        pca_by_layer = []
        hidden_pca_features_concatenated = []

        for layer in range(x_hidden_train.shape[0]):

            # get data
            x_hidden_layer_train = x_hidden_train[layer]  # (n_samples, d_model)

            # Hidden states scaling and pca
            scaler = StandardScaler()
            x_hidden_layer_train_scaled = scaler.fit_transform(x_hidden_layer_train)
            
            if self.args.mode in ["pca"]:
                layer_pca = PCA(n_components=self.args.components, random_state=42)
                x_hidden_layer_train_scaled = layer_pca.fit_transform(x_hidden_layer_train_scaled)
            else:
                layer_pca = None

            hidden_pca_features_concatenated.append(x_hidden_layer_train_scaled)

            # add items
            scalers_by_layer.append(scaler)
            pca_by_layer.append(layer_pca)

        # Combine pca-ed hidden states and attention features
        # (n_samples, n_layers * pca_dim)
        hidden_pca_features_concatenated = np.concatenate(hidden_pca_features_concatenated, axis=1) 

        # Fit the meta probe
        meta_probe = LogisticRegression(max_iter=2000, random_state=42, C=self.args.C)
        meta_probe.fit(hidden_pca_features_concatenated, y_train)

        return {
            "scalers_by_layer": scalers_by_layer,
            "pca_by_layer": pca_by_layer,
            "meta_probe": meta_probe,
        }
    

    def _train_linear_probe(self,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          x_val: np.ndarray,
                          y_val: np.ndarray
                          ) -> dict[str, Any]:

        scalers_by_layer = []
        pca_by_layer = []
        models_by_layer = []
        optimal_thresholds_by_layer = []
        val_metrics_by_layer = []
    

        for layer in range(x_train.shape[0]):

            # get data and scale
            x_layer_train = x_train[layer]  # (n_samples, d_model)

            scaler = StandardScaler()
            x_layer_train_scaled = scaler.fit_transform(x_layer_train)

            if self.args.mode in ["pca"]:
                pca = PCA(n_components=self.args.components, random_state=42)
                x_layer_train_scaled = pca.fit_transform(x_layer_train_scaled)

            probe_kind = "linear"
            if self.args.mode == "mlp":
                probe_kind = "mlp"
            clf_binary = Probe(
                kind=probe_kind,
                c=self.args.C,
                depth=self.args.mlp_depth,
            )
            clf_binary.fit(x_layer_train_scaled, y_train)

            # find optimal thresholds on val set
            x_layer_val = x_val[layer]  # (n_samples, d_model)
            x_layer_val_scaled = scaler.transform(x_layer_val)

            if self.args.mode in ["pca"]:
                x_layer_val_scaled = pca.transform(x_layer_val_scaled)
            y_val_score = clf_binary.predict_proba(x_layer_val_scaled)
            thresholds = optimal_thresholds(y_true=y_val, y_predict=y_val_score)
            
            # collect
            scalers_by_layer.append(scaler)
            if self.args.mode in ["pca"]:
                pca_by_layer.append(pca)
            models_by_layer.append(clf_binary)
            optimal_thresholds_by_layer.append(thresholds)

            # metrics on val set
            val_metrics = metrics(y_true=y_val, 
                                  y_predict=y_val_score,
                                  f1_threshold=thresholds["threshold_f1"],
                                  acc_threshold=thresholds["threshold_acc"])
            val_metrics_by_layer.append(val_metrics)

        return {
            "scalers_by_layer": scalers_by_layer,
            "pca_by_layer": pca_by_layer,
            "models_by_layer": models_by_layer,
            "optimal_thresholds_by_layer": optimal_thresholds_by_layer,
            "val_metrics_by_layer": val_metrics_by_layer,
        }

    def _evaluate_meta_probe(self,
                             train_out: dict[str, Any],
                             test: dict[str, Any]) -> dict[str, Any]:
        
        y_true = test["y"]
        hidden_pca_features_concatenated = []

        for layer in range(len(train_out["scalers_by_layer"])):
            scaler = train_out["scalers_by_layer"][layer]
            
            x_layer_test = test["hidden_x"][layer]
            x_layer_test_scaled = scaler.transform(x_layer_test)

            # apply pca only for those
            if self.args.mode in ["meta"]:
                pca = train_out["pca_by_layer"][layer]
                x_layer_test_scaled = pca.transform(x_layer_test_scaled)
            
            hidden_pca_features_concatenated.append(x_layer_test_scaled)


        # Combine (pca-ed) hidden states and attention features
        hidden_pca_features_concatenated = np.concatenate(hidden_pca_features_concatenated, axis=1) 

        # run the meta probe
        meta_probe = train_out["meta_probe"]
        meta_scores = meta_probe.predict_proba(hidden_pca_features_concatenated)[:, 1]

        meta_metrics = metrics(
            y_true=y_true,
            y_predict=meta_scores,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        return {
            "meta_metrics": meta_metrics,
            "scores": {"meta_scores": meta_scores.tolist()},
        }


    def _evaluate(self, 
                  train_out: dict[str, Any], 
                  test: dict[str, Any]) -> dict[str, Any]:

        test_metrics_by_layer = []
        y_true = test["y"]

        # run eval by layer
        all_projections = []
        for layer in range(len(train_out["models_by_layer"])):
            
            # Get scaler, pca, model, and thresholds per layer
            scaler = train_out["scalers_by_layer"][layer]
            if self.args.mode in ["pca"]:
                pca = train_out["pca_by_layer"][layer]
            else:
                pca=None
            model = train_out["models_by_layer"][layer]
            thresholds = train_out["optimal_thresholds_by_layer"][layer]

            # Get test data for this layer
            x_layer_test = test["hidden_x"][layer]  # (n_samples, d_model)
            x_layer_test_scaled = scaler.transform(x_layer_test)
            if self.args.mode in ["pca"]:
                x_layer_test_scaled = pca.transform(x_layer_test_scaled)

            x_proj = model.predict_proba(x_layer_test_scaled)

            # for small N; replace nan/and inf
            x_proj = np.nan_to_num(x_proj, nan=0.0, posinf=0.0, neginf=0.0)

            all_projections.append(x_proj)

            # layer metrics
            layer_metrics = metrics(
                y_true=y_true,
                y_predict=x_proj,
                f1_threshold=0.5, # these are wrong
                acc_threshold=0.5, # these are wrong
            )
            test_metrics_by_layer.append(layer_metrics)


        # A aggregate projection score
        all_projections = np.stack(all_projections, axis=0)  # (n_layers, n_samples)
        
        # for small N; replace nan/and inf
        all_projections = np.nan_to_num(all_projections, nan=0.0, posinf=0.0, neginf=0.0)

        if self.args.mode == "first_layer":
            aggregate_projection = all_projections[0]
        elif self.args.mode == "last_layer":
            aggregate_projection = all_projections[-1]
        else:
            aggregate_projection = all_projections.mean(axis=0)  # (n_samples,)

        mean_projection_metrics = metrics(
            y_true=y_true,
            y_predict=aggregate_projection,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        # B Aggreagtion using auroc as weight
        if self.args.mode in {"first_layer", "last_layer"}:
            weighted_projection = aggregate_projection
        else:
            auroc_val_by_layer = np.array([m["auroc"] for m in train_out["val_metrics_by_layer"]], dtype=np.float64)
            # sets a
            auroc_val_by_layer = np.clip(auroc_val_by_layer - 0.5, a_min=0.0, a_max=None)

            # softmax with temperature (lower temp -> more focus on top layers)
            temp = 0.5
            z = auroc_val_by_layer / max(temp, 1e-8)
            z = z - z.max()  # stability
            w = np.exp(z)
            w = w / (w.sum() + 1e-12)  # shape (n_layers,)

            # weighted aggregate score per sample
            weighted_projection = np.sum(all_projections * w[:, None], axis=0)  # (n_samples,)

        weighted_projection_metrics = metrics(
            y_true=y_true,
            y_predict=weighted_projection,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        return {
            "test_metrics_by_layer": test_metrics_by_layer,
            "mean_projection_metrics": mean_projection_metrics,
            "weighted_projection_metrics": weighted_projection_metrics,
            "scores": {
                "mean_projection": aggregate_projection.tolist(),
                "weighted_projection": weighted_projection.tolist(),
            }
        }


    def correlate_apt(self, 
                      test_data: dict[str, Any],
                      scores: dict[str, list[float]]) -> dict[str, Any]:
        meta = test_data["meta"]

        sem_sim = np.asarray([m.get("sem_similarity") for m in meta], dtype=np.float64)
        lev_dist = np.asarray([m.get("levenshtein_distance") for m in meta], dtype=np.float64)
        jac_dist = np.asarray([m.get("jaccard_distance") for m in meta], dtype=np.float64)
    
        def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
            return float(np.corrcoef(x, y)[0, 1])
        
        out = {}
        for name, pred_scores in scores.items():
            scores = np.asarray(pred_scores, dtype=np.float64)
            out[name] = {
                "sem_similarity": safe_corr(scores, sem_sim),
                "levenshtein_distance": safe_corr(scores, lev_dist),
                "jaccard_distance": safe_corr(scores, jac_dist),
            }    

        return out


    def run(self, args: Namespace) -> None:
        source_data = load_dataset(args=args)
        train = self._collect_model_states(source_data["train"])
        val = self._collect_model_states(source_data["val"])
        dataset_group = args.dataset.split("_")[0]
        target_datasets = OOD.get(dataset_group, [args.dataset])



        # TRAINING THE PROBE
        # layer wise probe
        if self.args.mode in ["default", "pca", "mlp", "first_layer", "last_layer"]:
            train_out = self._train_linear_probe(x_train=train["hidden_x"], 
                                                y_train=train["y"],
                                                x_val=val["hidden_x"], 
                                                y_val=val["y"])
        # meta probe
        else:
            train_out = self._train_meta_probe(x_hidden_train=train["hidden_x"],
                                    x_attn_train=train["attentions"],
                                    y_train=train["y"])

        # RUNNING EAL
        for target_dataset in target_datasets:
            if not args.ood:
                if args.dataset != target_dataset:
                    continue
            
            print("="*60)
            print(f"Evaluating on target dataset: {target_dataset}")
            print("="*60)

            target_data = load_dataset(args=Namespace(dataset=target_dataset, 
                                                      smoke_test=args.smoke_test,
                                                      training_size=args.training_size,
                                                      seed=args.seed))['test']
            test = self._collect_model_states(target_data)

            if self.args.mode in ["default", "pca", "mlp", "first_layer", "last_layer"]:
                test_metrics = self._evaluate(train_out=train_out, test=test)
            else:
                test_metrics = self._evaluate_meta_probe(train_out=train_out, test=test)


            if target_dataset in ["apt", "apt_m4_train", "beemo_human_edits", "beemo_machine_edits", "editlens"]:
                apt_correlations = self.correlate_apt(test_data=test, 
                                                        scores=test_metrics['scores'])
            else:
                apt_correlations = None
        
            # del scores
            del test_metrics['scores']

            # SAVE OUTPUT
            if args.mode == "mlp":
                filename = f"{args.mode}_{args.token_mode}_N{args.training_size}_D{args.mlp_depth}_{args.dataset}_2_{target_dataset}.json"
            else:
                filename = f"{args.mode}_{args.token_mode}_N{args.training_size}_PCA{args.components}_C{args.C}_{args.dataset}_2_{target_dataset}_S{args.seed}.json"

            args_copy = Namespace(**vars(args))  
            out_args = return_args(args_copy)
            out_args['target_dataset'] = target_dataset
            out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            out = {'args': out_args, 
                    'test_metrics': test_metrics,
                    'sim_correlations': apt_correlations}
            
            
            with open(os.path.join(self.out_dir, filename), "w") as f:
                json.dump(out, f, indent=4)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--ood", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--training_size", type=int, default=None)
    parser.add_argument("--folder", type=str, default="sandbox")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--mlp_depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.token_mode not in {"last_token", "pooling"}:
        raise ValueError("token_mode must be one of: last_token, pooling")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    if args.mode not in {"default", "pca", "meta", "meta_attn", "meta_no_pca", "mlp", "first_layer", "last_layer"}:
        raise ValueError("mode must be one of: default, pca, meta, meta_attn, meta_no_pca, mlp, first_layer, last_layer")

    if args.training_size == -1:
        args.training_size = None


    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.model_name = args.model

    analyzer = Probing(args=args)
    analyzer.run(args)


if __name__ == "__main__":
    main()
