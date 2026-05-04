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

    def _collect_model_states(self, 
                              items: list[dict]
                              ) -> dict[str, Any]:
        
        all_hidden_states = []
        all_attention_entropies = []
        labels = []

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


            # B ATTENTIONS
            if self.args.mode == "meta_attn":
                # tuple of len layer
                attentions = out["raw_attentions"]

                attentions_per_layer = []
                for a in attentions:
                    # a: (num_heads, seq_len, seq_len)
                    attentions_per_layer.append(a.detach().to(torch.float32).cpu().numpy())

                # compute attention features
                sample_attn_entropy = self.compute_attention_entropy(attentions_per_layer)
                all_attention_entropies.append(sample_attn_entropy)

            # LABELS
            labels.append(int(out["label"]))


        # STACK HS
        x = np.stack(all_hidden_states, axis=0)  # (n_samples, n_layers, d_model)
        x = np.transpose(x, (1, 0, 2))  # (n_layers, n_samples, d_model)
        y = np.asarray(labels, dtype=np.int32)   # (n_samples,)

        # STACK ATTENTIONS
        if self.args.mode == "meta_attn":
            # (n_samples, n_layers, num_heads) -> (n_layers, n_samples, num_heads)
            a = np.stack(all_attention_entropies, axis=0)
            a = np.transpose(a, (1, 0, 2))
        else:
            a = None

        return {
            "hidden_x": x,
            "attentions": a,
            "y": y,
        }
    

    def compute_attention_entropy(self, attentions_per_layer: list[np.ndarray]) -> np.ndarray:
        """
        attentions: (n_layers, n_samples, num_heads, seq_len, seq_len)
        
        return attn_entropy where (n_layers, n_samples, num_heads,) 

        """
            
        # compute entropy per head
        attention_entropy_by_layer = []
        for A in attentions_per_layer:
            # A (n_heads, seq_len, seq_len) 
        
            log_A = np.log(A + 1e-12)
            # sum over keys and mean over query
            ent = -(A * log_A).sum(axis=-1).mean(axis=-1)
        
            attention_entropy_by_layer.append(ent)

        return np.stack(attention_entropy_by_layer, axis=0)  # (n_layers, n_heads)
    

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
            
            layer_pca = PCA(n_components=self.args.components, random_state=42)
            x_hidden_layer_train_scaled = layer_pca.fit_transform(x_hidden_layer_train_scaled)

            hidden_pca_features_concatenated.append(x_hidden_layer_train_scaled)

            # add items
            scalers_by_layer.append(scaler)
            pca_by_layer.append(layer_pca)

        # Combine pca-ed hidden states and attention features
        # (n_samples, n_layers * pca_dim)
        hidden_pca_features_concatenated = np.concatenate(hidden_pca_features_concatenated, axis=1) 

        if self.args.mode == "meta_attn":
            # (n_layers, n_samples, n_heads) -> (n_samples, n_layers * n_heads)
            attn_features = np.transpose(x_attn_train, (1, 0, 2)).reshape(x_attn_train.shape[1], -1)
            features = np.concatenate([hidden_pca_features_concatenated, attn_features], axis=1)
        else:
            features = hidden_pca_features_concatenated


        # Fit the meta probe
        meta_probe = LogisticRegression(max_iter=2000, random_state=42)
        meta_probe.fit(features, y_train)

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

            if self.args.mode == "pca":
                pca = PCA(n_components=self.args.components, random_state=42)
                x_layer_train_scaled = pca.fit_transform(x_layer_train_scaled)

            # train probe
            clf_binary = LogisticRegression(max_iter=2000, 
                                            random_state=42)
            clf_binary.fit(x_layer_train_scaled, y_train)

            # find optimal thresholds on val set
            x_layer_val = x_val[layer]  # (n_samples, d_model)
            x_layer_val_scaled = scaler.transform(x_layer_val)

            if self.args.mode == "pca":
                x_layer_val_scaled = pca.transform(x_layer_val_scaled)
            y_val_score = clf_binary.predict_proba(x_layer_val_scaled)[:, 1]
            thresholds = optimal_thresholds(y_true=y_val, y_predict=y_val_score)
            
            # collect
            scalers_by_layer.append(scaler)
            if self.args.mode == "pca":
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
        x_attn_test = test["attentions"]

        hidden_pca_features_concatenated = []

        for layer in range(len(train_out["scalers_by_layer"])):
            scaler = train_out["scalers_by_layer"][layer]
            pca = train_out["pca_by_layer"][layer]
            
            # get hidden states
            x_layer_test = test["hidden_x"][layer]
            x_layer_test_scaled = scaler.transform(x_layer_test)
            x_layer_test_scaled = pca.transform(x_layer_test_scaled)
            hidden_pca_features_concatenated.append(x_layer_test_scaled)


        # Combine pca-ed hidden states and attention features
        hidden_pca_features_concatenated = np.concatenate(hidden_pca_features_concatenated, axis=1) 

        # (n_layers, n_samples, n_heads) -> (n_samples, n_layers * n_heads)
        if self.args.mode == "meta_attn":
            attn_features = np.transpose(x_attn_test, (1, 0, 2)).reshape(x_attn_test.shape[1], -1)
            features = np.concatenate([hidden_pca_features_concatenated, attn_features], axis=1)
        else:
            features = hidden_pca_features_concatenated

        # run the meta probe
        meta_probe = train_out["meta_probe"]
        meta_scores = meta_probe.predict_proba(features)[:, 1]


        meta_metrics = metrics(
            y_true=y_true,
            y_predict=meta_scores,
            f1_threshold=0.5,
            acc_threshold=0.5,
        )

        return {
            "meta_metrics": meta_metrics,
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
            if self.args.mode == "pca":
                pca = train_out["pca_by_layer"][layer]
            else:
                pca=None
            model = train_out["models_by_layer"][layer]
            thresholds = train_out["optimal_thresholds_by_layer"][layer]

            # Get test data for this layer
            x_layer_test = test["hidden_x"][layer]  # (n_samples, d_model)
            x_layer_test_scaled = scaler.transform(x_layer_test)
            if self.args.mode == "pca":
                x_layer_test_scaled = pca.transform(x_layer_test_scaled)

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

        return {
            "test_metrics_by_layer": test_metrics_by_layer,
            "mean_projection_metrics": mean_projection_metrics,
            "weighted_projection_metrics": weighted_projection_metrics,
        }


    def run(self, args: Namespace) -> None:
        source_data = load_dataset(args=args)
        train = self._collect_model_states(source_data["train"])
        val = self._collect_model_states(source_data["val"])

        if self.args.mode in ["default", "pca"]:
            # train
            train_out = self._train_linear_probe(x_train=train["hidden_x"], 
                                                y_train=train["y"],
                                                x_val=val["hidden_x"], 
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
                test = self._collect_model_states(target_data)
                test_metrics = self._evaluate(train_out=train_out, test=test)

                filename = f"{args.mode}_{args.token_mode}_{args.dataset}_2_{target_dataset}.json"
                out = {'args': return_args(args), 'test_metrics': test_metrics}
                with open(os.path.join(OUT_DIR, filename), "w") as f:
                    json.dump(out, f, indent=4)

        if self.args.mode in ["meta" ,"meta_attn"]:
            # train meta probe
            train_out = self._train_meta_probe(x_hidden_train=train["hidden_x"],
                                              x_attn_train=train["attentions"],
                                              y_train=train["y"])

            # eval meta probe
            for target_dataset in OOD[args.dataset.split("_")[0]]:
                if not args.ood:
                    if args.dataset != target_dataset:
                        continue
                
                print("="*60)
                print(f"Evaluating on target dataset: {target_dataset}")
                print("="*60)

                target_data = load_dataset(args=Namespace(dataset=target_dataset, 
                                                        smoke_test=args.smoke_test))['test']
                test = self._collect_model_states(target_data)
                test_metrics = self._evaluate_meta_probe(train_out=train_out, test=test)

                filename = f"{self.args.mode}_{args.token_mode}_{args.dataset}_2_{target_dataset}.json"
                out = {'args': return_args(args), 'test_metrics': test_metrics}
                with open(os.path.join(OUT_DIR, filename), "w") as f:
                    json.dump(out, f, indent=4)