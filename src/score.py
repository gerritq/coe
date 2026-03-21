
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture


BASE_DIR = os.getenv("BASE_COE")

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


@dataclass
class ScoreGMM:
    n_components: int = 2
    random_state: int = 42

    def run(self, out: list[dict[str, Any]], suffix: str) -> dict[str, float]:
        x, y = _features_labels(out)
        gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        clusters = gmm.fit_predict(x)

        # Map clusters to labels by majority vote
        mapping: dict[int, int] = {}
        for cluster_id in range(self.n_components):
            cluster_labels = y[clusters == cluster_id]
            if cluster_labels.size == 0:
                mapping[cluster_id] = 0
            else:
                mapping[cluster_id] = int(np.round(cluster_labels.mean()))
        y_pred = np.vectorize(mapping.get)(clusters)

        metrics = _metrics(y, y_pred)
        out_dir = os.path.join(BASE_DIR, "out", "scores")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"score_gmm_{suffix}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics


@dataclass
class ScoreLogistic:
    random_state: int = 42

    def run(self, out: list[dict[str, Any]], suffix: str) -> dict[str, float]:
        x, y = _features_labels(out)
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(x, y)
        y_pred = model.predict(x)

        metrics = _metrics(y, y_pred)
        out_dir = os.path.join(BASE_DIR, "out", "scores")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"score_logistic_{suffix}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
