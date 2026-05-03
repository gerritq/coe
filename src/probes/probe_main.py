from argparse import Namespace
from typing import Any
import os
import json
import torch
import numpy as np
from src.inference import Inference
from src.utils import load_dataset, optimal_thresholds, metrics, OOD, return_args
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
os.makedirs(OUT_DIR, exist_ok=True)

class LinearProbing:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.inference = Inference(self.args.model_name)

    def _collect_hidden_states(self, 
                               items: list[dict]
                               ) -> dict[str, Any]:
        
        all_hidden_states = []
        labels = []

        for item in items:
            out = self.inference.run(item=item, args=self.args)
            hidden_states = out["hidden_states"]

            hs_per_layer = []
            for l in hidden_states:
                hs_per_layer.append(l.detach().to(torch.float32).cpu().numpy())

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
        pca_by_layer = []
        models_by_layer = []
        optimal_thresholds_by_layer = []
        val_metrics_by_layer = []
        
        # for the meta probe
        pca_meta_by_layer = []
        pca_features_concatenated = []

        for layer in range(x_train.shape[0]):

            # get data and scale
            x_layer_train = x_train[layer]  # (n_samples, d_model)

            scaler = StandardScaler()
            x_layer_train_scaled = scaler.fit_transform(x_layer_train)

            if self.args.pca:
                pca = PCA(n_components=50, random_state=42)
                x_layer_train_scaled = pca.fit_transform(x_layer_train_scaled)

            # apply pca and add features to list for concatenation
            meta_pca = PCA(n_components=20, random_state=42)
            meta_pca_features = meta_pca.fit_transform(x_layer_train_scaled)
            pca_meta_by_layer.append(meta_pca)
            pca_features_concatenated.append(meta_pca_features)

            # train probe
            clf_binary = LogisticRegression(max_iter=2000, 
                                            random_state=42)
            clf_binary.fit(x_layer_train_scaled, y_train)

            # find optimal thresholds on val set
            x_layer_val = x_val[layer]  # (n_samples, d_model)
            x_layer_val_scaled = scaler.transform(x_layer_val)

            if self.args.pca:
                x_layer_val_scaled = pca.transform(x_layer_val_scaled)
            y_val_score = clf_binary.predict_proba(x_layer_val_scaled)[:, 1]
            thresholds = optimal_thresholds(y_true=y_val, y_predict=y_val_score)
            
            # collect
            scalers_by_layer.append(scaler)
            if self.args.pca:
                pca_by_layer.append(pca)
            models_by_layer.append(clf_binary)
            optimal_thresholds_by_layer.append(thresholds)

            # metrics on val set
            val_metrics = metrics(y_true=y_val, 
                                  y_predict=y_val_score,
                                  f1_threshold=thresholds["threshold_f1"],
                                  acc_threshold=thresholds["threshold_acc"])
            val_metrics_by_layer.append(val_metrics)


        # run the meta probes on the concatenated pca features
        meta_pca_probe = LogisticRegression(max_iter=2000, random_state=42)
        pca_features_concatenated = np.concatenate(pca_features_concatenated, axis=1)
        meta_pca_probe.fit(pca_features_concatenated, y_train)

        return {
            "scalers_by_layer": scalers_by_layer,
            "pca_by_layer": pca_by_layer,
            "models_by_layer": models_by_layer,
            "optimal_thresholds_by_layer": optimal_thresholds_by_layer,
            "val_metrics_by_layer": val_metrics_by_layer,
            "meta_pca_probe": meta_pca_probe,
            "meta_pca_by_layer": pca_meta_by_layer,
        }
    
    def _evaluate(self, 
                  train_out: dict[str, Any], 
                  test: dict[str, Any]) -> dict[str, Any]:

        test_metrics_by_layer = []
        y_true = test["y"]

                # run eval by layer
        all_projections = []
        meta_pca_features_concatenated = []
        for layer in range(len(train_out["models_by_layer"])):
            
            # Get scaler, pca, model, and thresholds per layer
            scaler = train_out["scalers_by_layer"][layer]
            if self.args.pca:
                pca = train_out["pca_by_layer"][layer]
            else:
                pca=None
            model = train_out["models_by_layer"][layer]
            thresholds = train_out["optimal_thresholds_by_layer"][layer]
            meta_pca = train_out["meta_pca_by_layer"][layer]

            # Get test data for this layer
            x_layer_test = test["x"][layer]  # (n_samples, d_model)
            x_layer_test_scaled = scaler.transform(x_layer_test)
            if self.args.pca:
                x_layer_test_scaled = pca.transform(x_layer_test_scaled)

            # Get meta pca features
            meta_pca_features = meta_pca.transform(x_layer_test_scaled)
            meta_pca_features_concatenated.append(meta_pca_features)

            # A Probing vector
            probe_vector = model.coef_[0]
            probe_vector = probe_vector / np.linalg.norm(probe_vector)

            # project 
            # n_samples x d_model/d_pca  x  d_model/d_pca -> n_samples
            x_proj = np.dot(x_layer_test_scaled, probe_vector)
            all_projections.append(x_proj)

            # layer metrics
            layer_metrics = metrics(
                y_true=y_true,
                y_predict=x_proj,
                f1_threshold=0.5, # these are wrong
                acc_threshold=0.5, # these are wrong
            )
            test_metrics_by_layer.append(layer_metrics)


        # A aggregate projection score, mean projection
        all_projections = np.stack(all_projections, axis=0)  # (n_layers, n_samples)
        mean_projection = all_projections.mean(axis=0)  # (n_samples,)
        mean_projection_metrics = metrics(
            y_true=y_true,
            y_predict=mean_projection,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        # B Aggreagtion using auroc as weight
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

        # C Meta probe on concatenated pca features
        meta_pca_features_concatenated = np.concatenate(meta_pca_features_concatenated, axis=1)
        meta_pca_probe = train_out["meta_pca_probe"]
        meta_pca_scores = meta_pca_probe.predict_proba(meta_pca_features_concatenated)[:, 1]
        meta_pca_metrics = metrics(
            y_true=y_true,
            y_predict=meta_pca_scores,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        return {
            "test_metrics_by_layer": test_metrics_by_layer,
            "mean_projection_metrics": mean_projection_metrics,
            "weighted_projection_metrics": weighted_projection_metrics,
            "meta_pca_metrics": meta_pca_metrics,
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

            filename = f"{args.token_mode}_{args.dataset}_2_{target_dataset}_pca{int(bool(args.pca))}.json"
            out = {'args': return_args(args), 'test_metrics': test_metrics}
            with open(os.path.join(OUT_DIR, filename), "w") as f:
                json.dump(out, f, indent=4)
