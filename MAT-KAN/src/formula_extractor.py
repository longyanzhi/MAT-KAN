"""Stage IV: Sparse symbolic distillation via Lasso."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from .edge_extractor import EdgeCandidate
from .kan_symbolic_eval import PRIMITIVES, evaluate_affine, parse_expr

@dataclass
class FormulaTerm:
    """One non-zero Lasso coefficient attached to a KAN-extracted expression."""
    coefficient: float
    fun_name: str
    a: float
    b: float
    feature_name: str
    feature_idx: int
    # For polynomial edges: coefficients [c0, c1, c2, c3] for x^0, x^1, x^2, x^3
    poly_coef: Optional[List[float]] = None

    def label(self) -> str:
        if self.fun_name == "poly" and self.poly_coef is not None:
            # Display polynomial as: (c0 + c1*x + c2*x^2 + c3*x^3)[feature]
            terms = []
            coef = self.poly_coef
            names = ['1', 'x', 'x^2', 'x^3']
            for i, c in enumerate(coef):
                if i >= len(names):
                    break
                if abs(c) < 1e-8:
                    continue
                name = names[i]
                if i == 0:  # constant term
                    terms.append(f"{c:.3g}")
                elif i == 1:  # x term
                    if abs(c - 1) < 1e-6:
                        terms.append("x")
                    elif abs(c + 1) < 1e-6:
                        terms.append("-x")
                    else:
                        terms.append(f"{c:.3g}*x")
                else:  # x^2, x^3 terms
                    if abs(c - 1) < 1e-6:
                        terms.append(name)
                    elif abs(c + 1) < 1e-6:
                        terms.append(f"-{name}")
                    else:
                        terms.append(f"{c:.3g}*{name}")
            if not terms:
                return f"0[{self.feature_name}]"
            poly_str = " + ".join(terms).replace("+ -", "- ")
            return f"({poly_str})[{self.feature_name}]"
        elif self.fun_name == "x":
            # Handle raw feature (x)
            if abs(self.coefficient - 1.0) < 1e-6:
                return f"x[{self.feature_name}]"
            elif abs(self.coefficient + 1.0) < 1e-6:
                return f"-x[{self.feature_name}]"
            else:
                return f"{self.coefficient:.4g}*x[{self.feature_name}]"
        elif self.fun_name in ("x^2", "x^3"):
            # Handle raw polynomial terms display
            if abs(self.coefficient - 1.0) < 1e-6:
                return f"{self.fun_name}[{self.feature_name}]"
            elif abs(self.coefficient + 1.0) < 1e-6:
                return f"-{self.fun_name}[{self.feature_name}]"
            else:
                return f"{self.coefficient:.4g}*{self.fun_name}[{self.feature_name}]"
        else:
            inner = "x" if abs(self.a - 1.0) < 1e-6 else f"{self.a:.4g} * x"
            if abs(self.b) > 1e-6:
                inner += f" + {self.b:.4g}" if self.b > 0 else f" - {abs(self.b):.4g}"
            return f"{self.fun_name}({inner})[{self.feature_name}]"


@dataclass
class Formula:
    """Symbolic formula for a single cluster."""
    cluster_id: int
    intercept: float
    terms: List[FormulaTerm]
    r2: float
    rmse: float
    n_terms: int
    alpha: float

    @property
    def expression(self) -> str:
        parts: List[str] = []
        if abs(self.intercept) > 1e-6:
            parts.append(f"{self.intercept:.4g}")
        for term in self.terms:
            coef = term.coefficient
            if abs(coef) <= 1e-6:
                continue
            label = term.label()
            if abs(coef - 1.0) < 1e-6:
                parts.append(label)
            elif abs(coef + 1.0) < 1e-6:
                parts.append(f"-{label}")
            else:
                parts.append(f"{coef:.4g}*{label}")
        if not parts:
            return "0"
        return " + ".join(parts).replace("+ -", "- ")


@dataclass
class _StoredEdge:
    """The minimal description of one design-matrix column."""
    fun_name: str
    a: float
    b: float
    feature_idx: int
    feature_name: str
    # For polynomial edges: coefficients [c0, c1, c2, c3] for x^0, x^1, x^2, x^3
    # None for primitive functions (sin, exp, etc.)
    poly_coef: Optional[List[float]] = None


@dataclass
class _StoredPolyEdge:
    """Polynomial edge: phi(x) = c0 + c1*x + c2*x^2 + c3*x^3 stored as coefficients."""
    coefficients: List[float]  # [c0, c1, c2, c3] for x^0, x^1, x^2, x^3
    feature_idx: int
    feature_name: str

    @property
    def fun_name(self) -> str:
        """Return a readable string representation."""
        terms = []
        for i, c in enumerate(self.coefficients):
            if abs(c) < 1e-8:
                continue
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
                    terms.append(f"x^{i}")
                elif abs(c + 1) < 1e-6:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{c:.3g}*x^{i}")
        if not terms:
            return "0"
        return (" + ".join(terms)).replace("+ -", "- ")


@dataclass
class _ClusterArtifacts:
    scaler: StandardScaler
    lasso: Lasso
    edges: List[_StoredEdge] = field(default_factory=list)


@dataclass
class ExtractionResult:
    formulas: Dict[int, Formula] = field(default_factory=dict)
    artifacts: Dict[int, _ClusterArtifacts] = field(default_factory=dict)


class FormulaExtractor:
    """Distill KAN edges into compact symbolic formulas via Lasso."""

    def __init__(
        self,
        use_cv: bool = True,
        alphas: Optional[np.ndarray] = None,
        min_alpha: float = 1e-4,
        max_alpha: float = 10.0,
        n_alphas: int = 100,
        cv_folds: int = 5,
        max_iter: int = 20000,
        tol: float = 1e-4,
        min_coef: float = 1e-6,
        add_polynomial: bool = True,  # NEW: add raw polynomial features
        poly_degree: int = 3,         # NEW: polynomial degree (1-3)
        add_interactions: bool = True,   # NEW: add feature interaction terms
        max_interactions: int = 20,      # NEW: max number of interactions per cluster
    ):
        self.use_cv = use_cv
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.n_alphas = n_alphas
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.tol = tol
        self.min_coef = min_coef
        self.add_polynomial = add_polynomial  # NEW
        self.poly_degree = poly_degree         # NEW
        self.add_interactions = add_interactions  # NEW
        self.max_interactions = max_interactions   # NEW
        self.alphas = alphas if alphas is not None else np.logspace(
            np.log10(min_alpha), np.log10(max_alpha), n_alphas)
        self._feature_indices: List[int] = []  # NEW: store feature indices for poly

        self.result_: ExtractionResult = ExtractionResult()


    def fit(
        self,
        cluster_edges: Dict[int, List[EdgeCandidate]],
        cluster_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        feature_names: List[str],
    ) -> "FormulaExtractor":
        """Fit one Lasso per cluster on the symbolic-expression design matrix."""
        result = ExtractionResult()

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

            edges = self._prepare_edges(candidates)
            if not edges:
                mean = float(np.mean(y_k))
                result.formulas[cluster_id] = Formula(
                    cluster_id=cluster_id, intercept=mean, terms=[],
                    r2=0.0, rmse=float(np.std(y_k)), n_terms=0, alpha=0.0,
                )
                continue

            # NEW: Add raw features and polynomial features
            extra_edges = []
            if self.add_polynomial:
                seen_feats = list(set(e.feature_idx for e in edges))
                for fi in seen_feats:
                    feat_name = next((e.feature_name for e in edges if e.feature_idx == fi), f"x{fi}")
                    # Add raw feature (x) - this is important for Lasso
                    extra_edges.append(_StoredEdge(
                        fun_name="x_raw",
                        a=0.0, b=0.0,
                        feature_idx=fi,
                        feature_name=feat_name,
                        poly_coef=[0.0, 1.0],  # This will evaluate as x
                    ))
                    # Add polynomial terms (x², x³) for extra expressiveness
                    for deg in range(2, min(self.poly_degree + 1, 4)):
                        extra_edges.append(_StoredEdge(
                            fun_name=f"poly{deg}",
                            a=0.0, b=0.0,
                            feature_idx=fi,
                            feature_name=feat_name,
                            poly_coef=[0.0] * deg + [1.0],
                        ))

            # Combine KAN edges and extra edges
            all_edges = edges + extra_edges
            Phi = self._build_design_matrix(X_k, all_edges)            # (n, m)
            scaler = StandardScaler()
            Phi_s = scaler.fit_transform(Phi)

            lasso, alpha = self._fit_lasso(Phi_s, y_k, n=len(y_k))
            y_pred = lasso.predict(Phi_s)
            r2 = float(1.0 - np.sum((y_k - y_pred) ** 2) /
                       max(np.sum((y_k - np.mean(y_k)) ** 2), 1e-12))
            rmse = float(np.sqrt(np.mean((y_k - y_pred) ** 2)))

            terms: List[FormulaTerm] = []
            order = np.argsort(-np.abs(lasso.coef_))
            for j in order:
                if abs(lasso.coef_[j]) <= self.min_coef:
                    continue
                e = all_edges[j]
                # Handle KAN polynomial edges
                if e.fun_name == "poly" and e.poly_coef is not None:
                    terms.append(FormulaTerm(
                        coefficient=float(lasso.coef_[j]),
                        fun_name="poly",
                        a=float(e.poly_coef[0]) if len(e.poly_coef) > 0 else 0.0,
                        b=float(e.poly_coef[1]) if len(e.poly_coef) > 1 else 0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                        poly_coef=e.poly_coef,
                    ))
                # Handle raw polynomial terms (poly1=x, poly2=x^2, poly3=x^3)
                elif e.fun_name.startswith("poly") and e.fun_name[4:].isdigit():
                    deg = int(e.fun_name[4:])
                    terms.append(FormulaTerm(
                        coefficient=float(lasso.coef_[j]),
                        fun_name=f"x^{deg}",
                        a=1.0, b=0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))
                # Handle raw feature (x)
                elif e.fun_name == "x_raw":
                    terms.append(FormulaTerm(
                        coefficient=float(lasso.coef_[j]),
                        fun_name="x",
                        a=1.0, b=0.0,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))
                else:
                    terms.append(FormulaTerm(
                        coefficient=float(lasso.coef_[j]),
                        fun_name=e.fun_name,
                        a=e.a, b=e.b,
                        feature_name=e.feature_name,
                        feature_idx=e.feature_idx,
                    ))

            result.formulas[cluster_id] = Formula(
                cluster_id=cluster_id,
                intercept=float(lasso.intercept_),
                terms=terms,
                r2=r2, rmse=rmse,
                n_terms=len(terms),
                alpha=float(alpha),
            )
            result.artifacts[cluster_id] = _ClusterArtifacts(
                scaler=scaler, lasso=lasso, edges=all_edges,
            )

        self.result_ = result
        return self

    @staticmethod
    def _parse_polynomial_expr(expr: str) -> Optional[List[float]]:
        """Parse a polynomial expression like '0.5 + 0.3*x - 0.1*x^2 + 0.02*x^3'.

        Returns a list of coefficients [c0, c1, c2, c3] for x^0, x^1, x^2, x^3,
        or None if the expression is not a polynomial.
        """
        expr = expr.strip()
        if not expr:
            return None

        # Check if this looks like a polynomial (contains x terms)
        if '*' not in expr and 'x' not in expr:
            return None

        # Initialize coefficients for up to x^3
        coef = [0.0, 0.0, 0.0, 0.0]

        # Pattern for polynomial terms: [coef*]x[^exp] with optional + or -
        # Matches: "0.5", "x", "0.3*x", "-0.1*x^2", "x^3", "2*x", etc.
        term_pattern = r'([+-]?\s*\d*\.?\d*(?:[eE][+-]?\d+)?)\s*\*?\s*x(?:\s*\^\s*(\d+))?'

        # First, try to match the whole polynomial
        terms_found = re.findall(term_pattern, expr)
        if not terms_found:
            # Try simpler patterns
            return None

        for coef_str, exp_str in terms_found:
            coef_str = coef_str.replace(' ', '')
            exp = int(exp_str) if exp_str else 1

            if exp > 3:
                continue  # Skip terms beyond x^3

            # Parse coefficient
            if coef_str == '' or coef_str == '+':
                c = 1.0
            elif coef_str == '-':
                c = -1.0
            else:
                try:
                    c = float(coef_str)
                except ValueError:
                    continue

            coef[exp] += c

        # Check if we found any terms with x
        has_x_term = any(abs(coef[i]) > 1e-10 for i in range(1, 4))
        if not has_x_term:
            return None

        # Normalize: if only constant term, this is not really a polynomial edge
        if not has_x_term:
            return None

        return coef


    def _prepare_edges(self, candidates: List[EdgeCandidate]) -> List[_StoredEdge]:
        """Reduce ``EdgeCandidate`` objects to the minimal symbolic description."""
        kept: List[_StoredEdge] = []
        seen: set = set()

        for cand in candidates:
            params = None
            if cand.abcd is not None:
                params = (cand.abcd["a"], cand.abcd["b"], cand.abcd["c"], cand.abcd["d"])

            if params is None and cand.expr:
                # First try to parse as polynomial (for numerical B-spline edges)
                poly_coef = self._parse_polynomial_expr(cand.expr)
                if poly_coef is not None:
                    key = ("poly", tuple(poly_coef), cand.feature_idx)
                    if key not in seen:
                        seen.add(key)
                        kept.append(_StoredEdge(
                            fun_name="poly",
                            a=0.0, b=0.0,
                            feature_idx=cand.feature_idx,
                            feature_name=cand.feature_name,
                            poly_coef=poly_coef,
                        ))
                    continue

                # Try to parse as primitive symbolic expression
                parsed = parse_expr(cand.expr)
                if parsed is not None:
                    fn, a, b, c, d = parsed
                    # Drop the outer c, d; Lasso absorbs them. Only the inner
                    # ``f(a x + b)`` is meaningful per edge.
                    params = (a, b, 1.0, 0.0)
                    fun_name = fn
                else:
                    continue
            elif params is not None:
                fun_name = self._infer_fun_name(cand)
                if fun_name is None or fun_name not in PRIMITIVES:
                    continue
                a, b, _, _ = params
            else:
                continue

            key = (fun_name, round(a, 6), round(b, 6), cand.feature_idx)
            if key in seen:
                continue
            seen.add(key)
            kept.append(_StoredEdge(
                fun_name=fun_name, a=float(a), b=float(b),
                feature_idx=cand.feature_idx, feature_name=cand.feature_name,
            ))
        return kept

    @staticmethod
    def _infer_fun_name(cand: EdgeCandidate) -> Optional[str]:
        """Recover ``fun_name`` from the rendered expression ``cand.expr``."""
        if not cand.expr:
            return None
        expr = cand.expr.strip()
        # Take the substring before the first '('.
        head = expr.split("(", 1)[0].strip()
        # Strip a leading "coef * " if present.
        if "*" in head:
            head = head.split("*")[-1].strip()
        return head if head in PRIMITIVES else None

    @staticmethod
    def _build_design_matrix(X: np.ndarray, edges: List[_StoredEdge]) -> np.ndarray:
        """Evaluate each symbolic expression on ``X`` to form the design matrix."""
        cols: List[np.ndarray] = []
        for e in edges:
            x_col = X[:, e.feature_idx].astype(np.float64)

            if e.fun_name == "poly" and e.poly_coef is not None:
                # Evaluate polynomial: c0 + c1*x + c2*x^2 + c3*x^3
                coef = e.poly_coef
                vals = np.zeros_like(x_col)
                if len(coef) > 0:
                    vals += coef[0]  # constant term
                if len(coef) > 1:
                    vals += coef[1] * x_col  # x term
                if len(coef) > 2:
                    vals += coef[2] * x_col ** 2  # x^2 term
                if len(coef) > 3:
                    vals += coef[3] * x_col ** 3  # x^3 term
            elif e.fun_name.startswith("poly") and e.fun_name[4:].isdigit():
                # Handle raw polynomial terms (poly1=x, poly2=x^2, poly3=x^3)
                deg = int(e.fun_name[4:])
                vals = x_col ** deg
            elif e.fun_name == "x_raw" or (e.fun_name == "poly" and e.poly_coef == [0.0, 1.0]):
                # Handle raw feature (x) - just pass through
                vals = x_col
            else:
                vals = evaluate_affine(e.fun_name, e.a, e.b, 1.0, 0.0, x_col)

            if not np.all(np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(vals.reshape(-1, 1))
        if not cols:
            return np.zeros((len(X), 0))
        return np.hstack(cols)


    def _fit_lasso(self, Phi: np.ndarray, y: np.ndarray, n: int) -> Tuple[Lasso, float]:
        if Phi.shape[1] == 0:
            # No candidate expressions: return a constant predictor.
            lasso = Lasso(alpha=1.0, max_iter=self.max_iter)
            lasso.fit(np.zeros((len(y), 1)), y)
            return lasso, 1.0
        if self.use_cv:
            cv = max(2, min(self.cv_folds, n // 5))
            try:
                model = LassoCV(
                    alphas=self.alphas,
                    cv=cv,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                model.fit(Phi, y)
                return model, float(model.alpha_)
            except Exception:
                pass
        model = Lasso(alpha=self.alphas.mean(), max_iter=self.max_iter, tol=self.tol)
        model.fit(Phi, y)
        return model, float(model.alpha_)


    def predict(self, X: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Piecewise predict: rebuild the design matrix with each cluster's edges
        and apply the cluster's fitted Lasso. Pure numpy; no KAN involved."""
        out = np.zeros(len(X))
        for cluster_id, artifacts in self.result_.artifacts.items():
            mask = cluster_labels == cluster_id
            if not mask.any():
                continue
            edges = artifacts.edges
            Phi = self._build_design_matrix(X[mask], edges)
            try:
                Phi_s = artifacts.scaler.transform(Phi)
                out[mask] = artifacts.lasso.predict(Phi_s)
            except Exception:
                out[mask] = self.result_.formulas[cluster_id].intercept
        return out

