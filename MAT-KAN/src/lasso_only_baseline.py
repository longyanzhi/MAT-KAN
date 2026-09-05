"""Lasso-only baseline: sparse polynomial regression on the raw standardized descriptors.

This module implements a minimal "no-KAN, no-clustering" ablation.  It applies
the same Stage IV polynomial feature construction (x, x^2, x^3) used by
``FormulaExtractor`` but skips both KAN edge extraction and K-means clustering.
The single design matrix is fitted once with ``LassoCV`` over the same alpha
range that MAT-KAN explores per cluster.

The intent is to isolate the contribution of (a) the RandomForest feature
selection at Stage I, (b) the K-means regime partitioning at Stage II, and
(c) the KAN edge extraction at Stage III, by removing (b) and (c) at once.

The number of non-zero terms is reported using the same |coef| > 5e-3
threshold as ``FormulaExtractor.min_coef`` so the value is comparable across
all ablation configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


@dataclass
class LassoOnlyResult:
    """Bundle of trained artifacts."""
    alpha: float
    r2_train: float
    rmse_train: float
    n_terms: int
    feature_names: List[str]
    coefficients: np.ndarray
    intercept: float
    scaler: StandardScaler


class LassoOnlyBaseline:
    """Sparse polynomial regression on raw standardized descriptors.

    Parameters
    ----------
    poly_degree : int
        Highest polynomial degree (1, 2 or 3).  ``1`` gives a pure linear fit,
        ``3`` matches the Stage IV design matrix in ``FormulaExtractor``.
    min_alpha : float
        Lower bound of the Lasso alpha grid.
    max_alpha : float
        Upper bound of the Lasso alpha grid.
    n_alphas : int
        Number of alphas sampled on a log scale between ``min_alpha`` and
        ``max_alpha``.
    cv_folds : int
        Number of folds used by ``LassoCV``.
    max_iter : int
        Maximum Lasso iterations.
    min_coef : float
        Coefficients with absolute value below this threshold are dropped
        from the symbolic count.  Default 5e-3 matches
        ``FormulaExtractor.min_coef``.
    random_state : int
        Random seed for ``LassoCV``.
    """

    def __init__(
        self,
        poly_degree: int = 3,
        min_alpha: float = 2e-2,
        max_alpha: float = 1.0,
        n_alphas: int = 50,
        cv_folds: int = 5,
        max_iter: int = 20000,
        min_coef: float = 5e-3,
        random_state: int = 42,
    ):
        if poly_degree < 1 or poly_degree > 3:
            raise ValueError("poly_degree must be 1, 2 or 3.")
        self.poly_degree = int(poly_degree)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.n_alphas = int(n_alphas)
        self.cv_folds = int(cv_folds)
        self.max_iter = int(max_iter)
        self.min_coef = float(min_coef)
        self.random_state = int(random_state)

        self.result_: Optional[LassoOnlyResult] = None

    # ------------------------------------------------------------------ build

    @staticmethod
    def _poly_feature_names(feature_names: List[str], degree: int) -> List[str]:
        """Return ``['x1', 'x1^2', 'x1^3', 'x2', ...]`` (no intercept)."""
        names: List[str] = []
        for fname in feature_names:
            for d in range(1, degree + 1):
                if d == 1:
                    names.append(fname)
                else:
                    names.append(f"{fname}^{d}")
        return names

    def _build_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Stack x, x^2, ..., x^poly_degree columns from each feature."""
        cols: List[np.ndarray] = []
        for d in range(1, self.poly_degree + 1):
            cols.append(X.astype(np.float64) ** d)
        return np.hstack(cols)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
    ) -> "LassoOnlyBaseline":
        """Fit a single Lasso over the polynomial design matrix of the training fold.

        Parameters
        ----------
        X_train : (n_samples, d) array of standardized descriptors.
        y_train : (n_samples,) target values, in their original unit system.
        feature_names : list of length ``d`` with the original descriptor names.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if X_train.shape[0] != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")

        Phi = self._build_design_matrix(X_train)
        names = self._poly_feature_names(feature_names, self.poly_degree)

        scaler = StandardScaler()
        Phi_s = scaler.fit_transform(Phi) if Phi.shape[1] > 0 else Phi

        if Phi.shape[1] == 0:
            # Degenerate case, return a constant predictor.
            mean = float(np.mean(y_train))
            self.result_ = LassoOnlyResult(
                alpha=0.0,
                r2_train=0.0,
                rmse_train=float(np.std(y_train)),
                n_terms=0,
                feature_names=[],
                coefficients=np.zeros(0),
                intercept=mean,
                scaler=scaler,
            )
            return self

        alphas = np.logspace(
            np.log10(self.min_alpha), np.log10(self.max_alpha), self.n_alphas
        )
        cv = max(2, min(self.cv_folds, len(y_train) // 5))
        model = LassoCV(
            alphas=alphas,
            cv=cv,
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=self.random_state,
        )
        model.fit(Phi_s, y_train)

        y_pred = model.predict(Phi_s)
        ss_res = float(np.sum((y_train - y_pred) ** 2))
        ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
        r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))

        n_terms = int(np.sum(np.abs(model.coef_) > self.min_coef))

        self.result_ = LassoOnlyResult(
            alpha=float(model.alpha_),
            r2_train=float(r2),
            rmse_train=float(rmse),
            n_terms=n_terms,
            feature_names=names,
            coefficients=np.asarray(model.coef_, dtype=np.float64),
            intercept=float(model.intercept_),
            scaler=scaler,
        )
        return self

    # ------------------------------------------------------------------ predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Lasso on a new (n, d) descriptor matrix."""
        if self.result_ is None:
            raise RuntimeError("LassoOnlyBaseline not fitted.")
        Phi = self._build_design_matrix(np.asarray(X, dtype=np.float64))
        try:
            Phi_s = self.result_.scaler.transform(Phi) if Phi.shape[1] > 0 else Phi
            return np.asarray(self.result_.intercept + Phi_s @ self.result_.coefficients).reshape(-1)
        except Exception:
            # Fall back to the train mean if the scaler cannot handle the input.
            return np.full(len(X), self.result_.intercept)

    # ------------------------------------------------------------------ helpers

    def expression(self) -> str:
        """Return a human-readable LaTeX-style formula string of the non-zero terms."""
        if self.result_ is None:
            return ""
        parts: List[str] = []
        if abs(self.result_.intercept) > self.min_coef:
            parts.append(f"{self.result_.intercept:.4g}")
        for j, name in enumerate(self.result_.feature_names):
            c = self.result_.coefficients[j]
            if abs(c) <= self.min_coef:
                continue
            if abs(c - 1.0) < 1e-6:
                parts.append(name)
            elif abs(c + 1.0) < 1e-6:
                parts.append(f"-{name}")
            else:
                parts.append(f"{c:.4g}*{name}")
        if not parts:
            return "0"
        return " + ".join(parts).replace("+ -", "- ")

    def terms(self) -> List[dict]:
        """Return a list of surviving (non-zero) terms as plain dicts.

        Each dict contains ``coefficient``, ``feature_idx``, ``feature_name``
        and ``degree`` (``1`` for raw x, ``2`` for x^2, ``3`` for x^3).
        Intercept is included as a separate entry with ``feature_name = "intercept"``
        when its magnitude exceeds ``min_coef``.
        """
        if self.result_ is None:
            return []
        out: List[dict] = []
        if abs(self.result_.intercept) > self.min_coef:
            out.append({
                "coefficient": float(self.result_.intercept),
                "feature_idx": -1,
                "feature_name": "intercept",
                "degree": 0,
            })
        n_feats = len(self.result_.feature_names) // self.poly_degree
        for j, name in enumerate(self.result_.feature_names):
            c = self.result_.coefficients[j]
            if abs(c) <= self.min_coef:
                continue
            deg = (j % self.poly_degree) + 1
            feat_idx = j // self.poly_degree
            # Recover the original feature name (without degree suffix).
            base_name = self.result_.feature_names[
                feat_idx * self.poly_degree
            ]
            out.append({
                "coefficient": float(c),
                "feature_idx": int(feat_idx),
                "feature_name": base_name,
                "degree": int(deg),
            })
        return out


__all__ = ["LassoOnlyBaseline", "LassoOnlyResult"]