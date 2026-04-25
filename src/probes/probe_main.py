from argparse import Namespace
from typing import Any
import os
import json
import numpy as np
from src.inference import Inference
from src.utils import load_dataset, optimal_thresholds, metrics, OOD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")

class LinearProbing:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(self.args.model_name)


    def _collect_hidden_states(self, 
                               items: list[dict]) -> dict[str, Any]:
        
        all_hidden_states = []
        labels = []

        for item in items:
            out = self.inference.run(item=item, args=self.args)
            hidden_states = out["hidden_states"]

            hs_per_layer = []
            for l in hidden_states:
                hs_per_layer.append(np.asarray(l, dtype=np.float32))

            all_hidden_states.append(np.stack(hs_per_layer, axis=0))
            labels.append(int(out["label"]))

        x = np.stack(all_hidden_states, axis=0)  # (n_samples, n_layers, d_model)
        x = np.transpose(x, (1, 0, 2))  # (n_layers, n_samples, d_model)
        y = np.asarray(labels, dtype=np.int32)   # (n_samples,)

        return {
            "x": x,
            "y": y,
        }
    
    def _train_linear_probe(self,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          x_val: np.ndarray,
                          y_val: np.ndarray
                          ) -> dict[str, Any]:


        scalers_by_layer = []
        models_by_layer = []
        optimal_thresholds_by_layer = []
        val_metrics_by_layer = []

        for layer in range(x_train.shape[0]):

            # get data and scale
            x_layer_train = x_train[layer]  # (n_samples, d_model)

            scaler = StandardScaler()
            x_layer_train_scaled = scaler.fit_transform(x_layer_train)

            # train probe
            clf_binary = LogisticRegression(random_state=42, 
                                            max_iter=1000)
            clf_binary.fit(x_layer_train_scaled, y_train)

            # find optimal thresholds on val set
            x_layer_val = x_val[layer]  # (n_samples, d_model)
            x_layer_val_scaled = scaler.transform(x_layer_val)
            y_val_score = clf_binary.predict_proba(x_layer_val_scaled)[:, 1]
            thresholds = optimal_thresholds(y_true=y_val, y_predict=y_val_score)
            
            # collect
            scalers_by_layer.append(scaler)
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
            "models_by_layer": models_by_layer,
            "optimal_thresholds_by_layer": optimal_thresholds_by_layer,
            "val_metrics_by_layer": val_metrics_by_layer,
        }
    
    def _evaluate(self, 
                  train_out: dict[str, Any], 
                  test: dict[str, Any]) -> dict[str, Any]:


        # Select top-k probes
        top_k = min(5, len(train_out["models_by_layer"]))
        top_k_models = sorted(
            enumerate(train_out["val_metrics_by_layer"]),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )[:top_k]
        top_k_indices = [idx for idx, _ in top_k_models]

        top_k_hard_preds = []
        test_metrics_by_layer = []
        y_true = test["y"]
        
        # run eval by layer
        for layer in range(len(train_out["models_by_layer"])):

            scaler = train_out["scalers_by_layer"][layer]
            model = train_out["models_by_layer"][layer]
            thresholds = train_out["optimal_thresholds_by_layer"][layer]

            x_layer_test = test["x"][layer]  # (n_samples, d_model)
            x_layer_test_scaled = scaler.transform(x_layer_test)
            y_score = model.predict_proba(x_layer_test_scaled)[:, 1]

            # metrics for this layer
            layer_metrics = metrics(
                y_true=y_true,
                y_predict=y_score,
                f1_threshold=thresholds["threshold_f1"],
                acc_threshold=thresholds["threshold_acc"],
            )
            test_metrics_by_layer.append(layer_metrics)

            if layer in top_k_indices:
                y_pred_hard = (y_score >= thresholds["threshold_acc"]).astype(np.float32)
                top_k_hard_preds.append(y_pred_hard)

        # ensemble score
        votes = np.stack(top_k_hard_preds, axis=0)              # (top_k, n_samples)
        ensemble_score = (votes.sum(axis=0) >= (top_k / 2)).astype(np.float32)  # (n_samples,)
        ensemble_metrics = metrics(
            y_true=y_true,
            y_predict=ensemble_score,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        return {
            "test_metrics_by_layer": test_metrics_by_layer,
            "top_k_layers_by_val_acc": top_k_indices,
            "ensemble_metrics": ensemble_metrics,
        }


    def run(self, args: Namespace) -> None:
        source_data = load_dataset(args=args)
        train = self._collect_hidden_states(source_data["train"])
        val = self._collect_hidden_states(source_data["val"])

        # train
        train_out = self._train_linear_probe(x_train=train["x"], 
                                             y_train=train["y"],
                                             x_val=val["x"], 
                                             y_val=val["y"])

        # RUN EVAL
        for target_dataset in OOD[args.dataset.split("_")[0]]:
            if not args.ood:
                if args.dataset != target_dataset:
                    continue
            
            print("="*60)
            print(f"Evaluating on target dataset: {target_dataset}")
            print("="*60)

            target_data = load_dataset(args=Namespace(dataset=target_dataset, 
                                                      smoke_test=args.smoke_test))['test']
            test = self._collect_hidden_states(target_data)
            test_metrics = self._evaluate(train_out=train_out, test=test)

            filename = f"{args.mode}_{args.dataset}_2_{target_dataset}_metrics.json"
            with open(os.path.join(OUT_DIR, filename), "w") as f:
                json.dump(test_metrics, f, indent=4)
