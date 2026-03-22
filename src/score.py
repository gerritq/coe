
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


BASE_DIR = os.getenv("BASE_COE")
SCORER_DIR = os.path.join(BASE_DIR, "scores", "test")
os.makedirs(SCORER_DIR, exist_ok=True)

def _features_labels(out: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for item in out:
        features.append(
            [
                item["magnitude_change_mean"],
                item["angle_change_mean"],
                item["length_change_mean"],
            ]
        )
        labels.append(item["label"])
    return np.array(features, dtype=float), np.array(labels, dtype=int)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "prec": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
    }

def _args_payload(args: Any | None) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    return {"value": str(args)}

def make_split(
    out: list[dict[str, Any]],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _features_labels(out)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


@dataclass
class ScoreGMM:
    n_components: int = 2
    random_state: int = 42
    test_size: float = 0.2

    def run(
        self,
        out: list[dict[str, Any]],
        suffix: str,
        args: Any | None = None,
    ) -> dict[str, float]:
        x_train, x_test, y_train, y_test = make_split(out, test_size=self.test_size, random_state=self.random_state)
        gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        train_clusters = gmm.fit_predict(x_train)

        # Map clusters to labels by majority vote
        mapping: dict[int, int] = {}
        for cluster_id in range(self.n_components):
            cluster_labels = y_train[train_clusters == cluster_id]
            if cluster_labels.size == 0:
                mapping[cluster_id] = 0
            else:
                mapping[cluster_id] = int(np.round(cluster_labels.mean()))
        test_clusters = gmm.predict(x_test)
        y_pred = np.vectorize(mapping.get)(test_clusters)

        metrics = _metrics(y_test, y_pred)
        payload = {
            "metrics": metrics,
            "args": _args_payload(args),
        }
        
        with open(os.path.join(SCORER_DIR, f"score_gmm_{suffix}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return metrics


@dataclass
class ScoreLogistic:
    random_state: int = 42
    test_size: float = 0.2

    def run(
        self,
        out: list[dict[str, Any]],
        suffix: str,
        args: Any | None = None,
    ) -> dict[str, float]:
        x_train, x_test, y_train, y_test = make_split(out, test_size=self.test_size, random_state=self.random_state)
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = _metrics(y_test, y_pred)
        payload = {
            "metrics": metrics,
            "args": _args_payload(args),
        }
        
        with open(os.path.join(SCORER_DIR, f"score_logistic_{suffix}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return metrics


@dataclass
class ScoreMLP:
    hidden_size: int = 16
    random_state: int = 42
    test_size: float = 0.2

    def run(
        self,
        out: list[dict[str, Any]],
        suffix: str,
        args: Any | None = None,
    ) -> dict[str, float]:
        x_train, x_test, y_train, y_test = make_split(out, test_size=self.test_size, random_state=self.random_state)
        model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_size,),
            random_state=self.random_state,
            max_iter=1000,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = _metrics(y_test, y_pred)
        payload = {
            "metrics": metrics,
            "args": _args_payload(args),
        }
        
        with open(os.path.join(SCORER_DIR, f"score_mlp_{suffix}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return metrics
