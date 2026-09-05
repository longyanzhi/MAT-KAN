"""Stage I: Feature pruning via RandomForest importance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class PruningResult:
    """Result of feature pruning."""
    selected_indices: List[int]
    selected_features: List[str]
    importance: List[float]


class AdaptivePruner:
    """Feature pruner using RandomForest importance."""

    def __init__(
        self,
        min_features: int = 2,
        threshold_ratio: float = 0.05,
        random_state: int = 42,
    ):
        self.min_features = int(min_features)
        self.threshold_ratio = float(threshold_ratio)
        self.random_state = random_state
        self.result_: PruningResult = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "AdaptivePruner":
        """Select features based on RandomForest importance."""
        rf = RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=self.random_state,
        )
        rf.fit(X, y)
        importances = rf.feature_importances_

        top = importances.max() or 1.0
        mask = importances >= self.threshold_ratio * top
        if mask.sum() < self.min_features:
            order = np.argsort(importances)[::-1]
            top_k = order[:self.min_features]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_k] = True

        sel_idx = np.where(mask)[0]
        sel_feat = [feature_names[i] for i in sel_idx]

        self.result_ = PruningResult(
            selected_indices=sel_idx.tolist(),
            selected_features=sel_feat,
            importance=importances.tolist(),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Pruner not fitted.")
        return X[:, self.result_.selected_indices]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[List[int], List[str]]:
        self.fit(X, y, feature_names)
        return self.result_.selected_indices, self.result_.selected_features
