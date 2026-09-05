"""
Stage II: K-means clustering of the pruned feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterSplit:
    """Per-cluster slice of the dataset."""
    cluster_id: int
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray


@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    n_clusters: int
    metric_scores: Dict[str, float]


class KANClusterer:
    """K-means clusterer applied to the feature space."""

    def __init__(
        self,
        n_clusters: int = 3,
        n_init: int = 50,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.kmeans: KMeans = None
        self.result_: ClusterResult = None

    def fit(self, X: np.ndarray, *, normalize: bool = True) -> "KANClusterer":
        X = np.asarray(X, dtype=np.float64)
        Z = self.scaler.fit_transform(X) if normalize else X
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels = self.kmeans.fit_predict(Z)

        self.result_ = ClusterResult(
            labels=labels,
            centers=self.kmeans.cluster_centers_,
            n_clusters=self.n_clusters,
            metric_scores={
                "silhouette": float(silhouette_score(Z, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
                "davies_bouldin": float(davies_bouldin_score(Z, labels)),
                "inertia": float(self.kmeans.inertia_),
            },
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("Clusterer not fitted.")
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self.kmeans.predict(Z)

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[int, ClusterSplit]:
        """Return one ``ClusterSplit`` per cluster label."""
        if self.result_ is None:
            raise RuntimeError("Clusterer not fitted.")
        out: Dict[int, ClusterSplit] = {}
        labels = self.result_.labels
        for k in range(self.n_clusters):
            mask = labels == k
            out[k] = ClusterSplit(
                cluster_id=k,
                X=np.asarray(X)[mask],
                y=np.asarray(y).reshape(-1)[mask],
                indices=np.where(mask)[0],
            )
        return out

    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(2, 8),
        metric: str = "silhouette",
    ) -> Tuple[int, Dict[int, Dict[str, float]]]:
        """Sweep over ``k_range`` and return the best ``k`` (and per-k metrics)."""
        X = np.asarray(X, dtype=np.float64)
        Z = self.scaler.fit_transform(X)
        results: Dict[int, Dict[str, float]] = {}
        for k in k_range:
            km = KMeans(n_clusters=k,
                        n_init=self.n_init,
                        max_iter=self.max_iter,
                        random_state=self.random_state)
            labels = km.fit_predict(Z)
            results[k] = {
                "silhouette": float(silhouette_score(Z, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
                "davies_bouldin": float(davies_bouldin_score(Z, labels)),
                "inertia": float(km.inertia_),
            }
        if metric == "elbow":
            inertia = np.array([results[k]["inertia"] for k in k_range])
            norm = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-10)
            best_idx = int(np.argmin(np.diff(np.diff(norm)))) + 1
        elif metric == "calinski_harabasz":
            best_idx = int(np.argmax([results[k]["calinski_harabasz"] for k in k_range]))
        elif metric == "davies_bouldin":
            best_idx = int(np.argmin([results[k]["davies_bouldin"] for k in k_range]))
        else:
            best_idx = int(np.argmax([results[k]["silhouette"] for k in k_range]))
        best_k = list(k_range)[best_idx]
        self.n_clusters = best_k
        return best_k, results