"""Stage IV (ablation): Direct KAN-extracted formulas without LASSO sparsification.

This module mirrors :class:`mat_kan.src.formula_extractor.FormulaExtractor`
but skips the L1 shrinkage step entirely. The KAN edges are evaluated as before,
then a plain Ordinary Least Squares fit assigns a coefficient to *every* edge
expression (no sparsity, no coefficient thresholding, no cross-validation
over ``alpha``). The resulting "dense" formula keeps the full symbolic
candidate library and is therefore a clean ablation baseline that isolates
the contribution of LASSO sparsification in MAT-KAN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from .edge_extractor import EdgeCandidate
from .formula_extractor import (
    Formula,
    FormulaExtractor,
    FormulaTerm,
    _ClusterArtifacts,
    _StoredEdge,
)
from .kan_symbolic_eval import evaluate_affine

@dataclass
class NoLassoArtifacts(_ClusterArtifacts):
    """Same as the LASSO artifacts but with a plain :class:`LinearRegression`."""
    lasso: object  # actually a sklearn.linear_model.LinearRegression
    alpha: float = 0.0


@dataclass
class NoLassoResult:
    """Output of :class:`NoLassoExtractor.fit`."""
    formulas: Dict[int, Formula] = field(default_factory=dict)
    artifacts: Dict[int, NoLassoArtifacts] = field(default_factory=dict)
    # ---- NEW: intermediate data that we explicitly want to persist ----
    design_matrices: Dict[int, np.ndarray] = field(default_factory=dict)        # Phi per cluster
    design_feature_names: Dict[int, List[str]] = field(default_factory=dict)   # column labels
    scaled_design_matrices: Dict[int, np.ndarray] = field(default_factory=dict)  # Phi_s
    y_train_per_cluster: Dict[int, np.ndarray] = field(default_factory=dict)
    y_pred_train: Dict[int, np.ndarray] = field(default_factory=dict)
    edges_per_cluster: Dict[int, List[dict]] = field(default_factory=dict)     # raw edge info

class NoLassoExtractor:
    """Stage IV ablation: fit OLS on the KAN edge design matrix (no sparsity)."""

    def __init__(
        self,
        add_polynomial: bool = True,
        poly_degree: int = 3,
    ):
        self.add_polynomial = add_polynomial
        self.poly_degree = poly_degree
        self.result_: NoLassoResult = NoLassoResult()

    def fit(
        self,
        cluster_edges: Dict[int, List[EdgeCandidate]],
        cluster_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        feature_names: List[str],
    ) -> "NoLassoExtractor":
        """Fit one OLS per cluster on the symbolic-expression design matrix."""
        result = NoLassoResult()
        # Reuse the edge preparation/polynomial augmentation logic from the
        # Lasso-based extractor — the design matrix is identical.
        helper = FormulaExtractor(
            use_cv=False, add_polynomial=self.add_polynomial,
            poly_degree=self.poly_degree,
        )

        for cluster_id, candidates in cluster_edges.items():
            if cluster_id not in cluster_data:
                continue
            X_k, y_k = cluster_data[cluster_id]
            if len(y_k) < 10:
                mean = float(np.mean(y_k))
                result.formulas[cluster_id] = Formula(
                    cluster_id=cluster_id, intercept=mean, terms=[],
                    r2=0.0, rmse=float(np.std(y_k)), n_terms=0, alpha=0.0,
                )
                continue

            edges = helper._prepare_edges(candidates)
            if not edges:
                mean = float(np.mean(y_k))
                result.formulas[cluster_id] = Formula(
                    cluster_id=cluster_id, intercept=mean, terms=[],
                    r2=0.0, rmse=float(np.std(y_k)), n_terms=0, alpha=0.0,
                )
                continue

            # Augment with raw / polynomial features (identical to LASSO version).
            extra_edges: List[_StoredEdge] = []
            if self.add_polynomial:
                seen_feats = list({e.feature_idx for e in edges})
                for fi in seen_feats:
                    feat_name = next(
                        (e.feature_name for e in edges if e.feature_idx == fi),
                        f"x{fi}",
                    )
                    extra_edges.append(_StoredEdge(
                        fun_name="x_raw", a=0.0, b=0.0,
                        feature_idx=fi, feature_name=feat_name,
                        poly_coef=[0.0, 1.0],
                    ))
                    for deg in range(2, min(self.poly_degree + 1, 4)):
                        extra_edges.append(_StoredEdge(
                            fun_name=f"poly{deg}", a=0.0, b=0.0,
                            feature_idx=fi, feature_name=feat_name,
                            poly_coef=[0.0] * deg + [1.0],
                        ))

            all_edges = edges + extra_edges
            Phi = FormulaExtractor._build_design_matrix(X_k, all_edges)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Phi_s = scaler.fit_transform(Phi) if Phi.shape[1] > 0 else Phi

            # ---- OLS (no L1, no CV, no thresholding) ----
            ols = LinearRegression()
            if Phi.shape[1] == 0:
                ols.fit(np.zeros((len(y_k), 1)), y_k)
            else:
                ols.fit(Phi_s, y_k)

            y_pred = ols.predict(Phi_s)
            r2 = float(
                1.0 - np.sum((y_k - y_pred) ** 2)
                / max(np.sum((y_k - np.mean(y_k)) ** 2), 1e-12)
            )
            rmse = float(np.sqrt(np.mean((y_k - y_pred) ** 2)))

            # Keep *every* term — no sparsity.
            terms: List[FormulaTerm] = []
            order = np.argsort(-np.abs(ols.coef_)) if Phi.shape[1] > 0 else []
            for j in order:
                if Phi.shape[1] == 0:
                    break
                e = all_edges[j]
                if e.fun_name == "poly" and e.poly_coef is not None:
                    terms.append(FormulaTerm(
                        coefficient=float(ols.coef_[j]),
                        fun_name="poly",
                        a=float(e.poly_coef[0]) if len(e.poly_coef) > 0 else 0.0,
                        b=float(e.poly_coef[1]) if len(e.poly_coef) > 1 else 0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                        poly_coef=e.poly_coef,
                    ))
                elif e.fun_name.startswith("poly") and e.fun_name[4:].isdigit():
                    deg = int(e.fun_name[4:])
                    terms.append(FormulaTerm(
                        coefficient=float(ols.coef_[j]),
                        fun_name=f"x^{deg}",
                        a=1.0, b=0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))
                elif e.fun_name == "x_raw":
                    terms.append(FormulaTerm(
                        coefficient=float(ols.coef_[j]),
                        fun_name="x",
                        a=1.0, b=0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))
                else:
                    terms.append(FormulaTerm(
                        coefficient=float(ols.coef_[j]),
                        fun_name=e.fun_name,
                        a=e.a, b=e.b,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))

            result.formulas[cluster_id] = Formula(
                cluster_id=cluster_id,
                intercept=float(ols.intercept_),
                terms=terms,
                r2=r2, rmse=rmse,
                n_terms=len(terms),
                alpha=0.0,                 # alpha is meaningless for OLS
            )
            result.artifacts[cluster_id] = NoLassoArtifacts(
                scaler=scaler, lasso=ols, edges=all_edges,
            )
            # ---- persist every intermediate artifact ----
            result.design_matrices[cluster_id] = Phi
            result.scaled_design_matrices[cluster_id] = Phi_s
            result.y_train_per_cluster[cluster_id] = y_k
            result.y_pred_train[cluster_id] = y_pred
            result.design_feature_names[cluster_id] = [
                _edge_to_label(e) for e in all_edges
            ]
            result.edges_per_cluster[cluster_id] = [
                {
                    "feature_idx": e.feature_idx,
                    "feature_name": e.feature_name,
                    "fun_name": e.fun_name,
                    "a": e.a,
                    "b": e.b,
                    "poly_coef": list(e.poly_coef) if e.poly_coef else None,
                    "label": _edge_to_label(e),
                    "coefficient": float(ols.coef_[j]),
                }
                for j, e in enumerate(all_edges) if Phi.shape[1] > 0
            ]

        self.result_ = result
        return self

    def predict(self, X: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Piecewise predict using each cluster's stored OLS coefficients."""
        out = np.zeros(len(X))
        for cluster_id, artifacts in self.result_.artifacts.items():
            mask = cluster_labels == cluster_id
            if not mask.any():
                continue
            edges = artifacts.edges
            Phi = FormulaExtractor._build_design_matrix(X[mask], edges)
            try:
                Phi_s = artifacts.scaler.transform(Phi)
                out[mask] = artifacts.lasso.predict(Phi_s)
            except Exception:
                out[mask] = self.result_.formulas[cluster_id].intercept
        return out


def _edge_to_label(e: _StoredEdge) -> str:
    """Return a short human-readable label for an edge (used in CSV headers)."""
    if e.fun_name == "poly" and e.poly_coef is not None:
        coef = e.poly_coef
        terms = []
        names = ['1', 'x', 'x^2', 'x^3']
        for i, c in enumerate(coef):
            if i >= len(names):
                break
            if abs(c) < 1e-8:
                continue
            name = names[i]
            if i == 0:
                terms.append(f"{c:.3g}")
            elif i == 1:
                if abs(c - 1) < 1e-6:
                    terms.append("x")
                elif abs(c + 1) < 1e-6:
                    terms.append("-x")
                else:
                    terms.append(f"{c:.3g}*x")
            else:
                if abs(c - 1) < 1e-6:
                    terms.append(name)
                elif abs(c + 1) < 1e-6:
                    terms.append(f"-{name}")
                else:
                    terms.append(f"{c:.3g}*{name}")
        poly_str = " + ".join(terms).replace("+ -", "- ") if terms else "0"
        return f"({poly_str})[{e.feature_name}]"
    if e.fun_name.startswith("poly") and e.fun_name[4:].isdigit():
        deg = int(e.fun_name[4:])
        return f"x^{deg}[{e.feature_name}]"
    if e.fun_name == "x_raw":
        return f"x[{e.feature_name}]"
    inner = "x" if abs(e.a - 1.0) < 1e-6 else f"{e.a:.4g} * x"
    if abs(e.b) > 1e-6:
        inner += f" + {e.b:.4g}" if e.b > 0 else f" - {abs(e.b):.4g}"
    return f"{e.fun_name}({inner})[{e.feature_name}]"