"""
Stage III: collect symbolic edge candidates from a trained local KAN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .kan_model import KANRegressor


@dataclass
class EdgeCandidate:
    """One learnable edge from a local KAN, summarised symbolically."""
    cluster_id: int
    layer: int
    in_id: int
    out_id: int
    feature_idx: int
    feature_name: str
    expr: str
    abcd: Optional[Dict[str, float]] = None
    poly_coef: Optional[np.ndarray] = None  # polynomial coefficients for numerical edges
    activations: Optional[np.ndarray] = None  # evaluated on cluster X
    feature_values: Optional[np.ndarray] = None
    design_value: Optional[np.ndarray] = None  # activations * feature_values


def extract_edges(
    local_kan: KANRegressor,
    X: np.ndarray,
    feature_names: List[str],
    cluster_id: int = 0,
    run_auto_symbolic: bool = False,  # default off; polynomial extraction is primary
) -> List[EdgeCandidate]:
    # Activate all edges FIRST so numerical masks are guaranteed on.
    local_kan.activate_all_edges()

    if run_auto_symbolic:
        try:
            local_kan.auto_symbolic(verbose=False)
            local_kan.activate_all_edges()  # restore any edges clobbered by fix_symbolic
        except Exception:
            pass

    edges = local_kan.list_symbolic_edges(feature_names)
    candidates: List[EdgeCandidate] = []

    for edge in edges:
        idx = edge["feature_idx"]
        if idx >= X.shape[1]:
            continue  # skip edges feeding into hidden nodes

        # Get activations for this edge
        activations = local_kan.evaluate_edge(edge, X)
        if activations is None or len(activations) != len(X):
            continue
        if not np.all(np.isfinite(activations)):
            continue

        # Fit a polynomial to the spline activations.
        poly_coef = None
        poly_expr = None
        x_vals = X[:, idx].astype(np.float64)

        try:
            degree = min(3, len(x_vals) - 1)
            if degree < 1:
                degree = 1

            poly_coef = np.polyfit(x_vals, activations.astype(np.float64), degree)

            if poly_coef is not None and len(poly_coef) > 0:
                # Format as readable expression.
                terms = []
                names = ['1', 'x', 'x^2', 'x^3'][:len(poly_coef)]
                for a, name in zip(poly_coef, names):
                    if abs(a) < 1e-6:
                        continue
                    if abs(a - 1) < 1e-6:
                        terms.append(name)
                    elif abs(a + 1) < 1e-6:
                        terms.append(f"-{name}")
                    else:
                        terms.append(f"{a:.3g}*{name}" if name != '1' else f"{a:.3g}")

                if terms:
                    poly_expr = (' + '.join(terms)).replace('+ -', '- ')
                    edge["expr"] = poly_expr
        except Exception:
            pass  # keep placeholder expression

        feature_values = X[:, idx].astype(np.float32)
        abcd = edge.get("abcd")
        candidates.append(EdgeCandidate(
            cluster_id=cluster_id,
            layer=edge["layer"],
            in_id=edge["in_id"],
            out_id=edge["out_id"],
            feature_idx=idx,
            feature_name=edge["feature_name"],
            expr=edge["expr"],
            abcd={"a": abcd[0], "b": abcd[1], "c": abcd[2], "d": abcd[3]} if abcd else None,
            poly_coef=poly_coef,
            activations=activations,
            feature_values=feature_values,
            design_value=(activations * feature_values).astype(np.float32),
        ))

    return candidates
