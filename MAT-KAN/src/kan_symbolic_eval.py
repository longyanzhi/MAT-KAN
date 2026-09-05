"""
Evaluate the symbolic edge expressions that PyKAN's ``auto_symbolic`` produces.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _safe_log(z: np.ndarray) -> np.ndarray:
    return np.log(np.abs(z) + 1e-12)


def _safe_inv(z: np.ndarray) -> np.ndarray:
    return 1.0 / (z + np.sign(z) * 1e-12 + (z == 0).astype(np.float64) * 1e-12)


def _safe_inv2(z: np.ndarray) -> np.ndarray:
    return 1.0 / (z ** 2 + 1e-12)


def _safe_sqrt(z: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(z))


PRIMITIVES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # monomials / algebraic
    "x":     lambda z: z,
    "x^2":   lambda z: z ** 2,
    "x^3":   lambda z: z ** 3,
    "x^4":   lambda z: z ** 4,
    "1/x":   _safe_inv,
    "1/x^2": _safe_inv2,
    "sqrt":  _safe_sqrt,
    "1/sqrt(x)": lambda z: 1.0 / (np.sqrt(np.abs(z)) + 1e-12),
    # transcendental
    "exp":   lambda z: np.exp(np.clip(z, -50.0, 50.0)),
    "log":   _safe_log,
    "abs":   np.abs,
    # trigonometric / hyperbolic
    "sin":   np.sin,
    "cos":   np.cos,
    "tan":   np.tan,
    "arctan": np.arctan,
    "sinh":  np.sinh,
    "cosh":  np.cosh,
    "tanh":  np.tanh,
    # bump-shaped
    "gaussian": lambda z: np.exp(-z ** 2),
    "sigmoid":  lambda z: 1.0 / (1.0 + np.exp(np.clip(-z, -50.0, 50.0))),
    # placeholder for an inactive edge; evaluates to zero.
    "0":     lambda z: np.zeros_like(z),
}


def fun_registry() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Read-only view of the supported primitive registry."""
    return dict(PRIMITIVES)

def evaluate_affine(fun_name: str, a: float, b: float, c: float, d: float,
                    x: np.ndarray) -> np.ndarray:
    """
    Return ``c * f(a * x + b) + d`` for a 1-D array ``x``.
    """
    z = a * x + b
    fn = PRIMITIVES.get(fun_name)
    if fn is None:
        out = z
    else:
        out = fn(z)
    return c * out + d


def parse_expr(expr: str) -> Tuple[str, float, float, float, float] | None:
    """Parse a string produced by ``kan_model._format_symbolic_edge`` back into
    ``(fun_name, a, b, c, d)``.

    The accepted format is::

        [c * ]f(a * x [+|- b]) [+|- d]

    The function name can include ``^`` (e.g. ``x^2``). Returns ``None`` when
    the expression is not parseable; the caller should then drop the edge.
    """
    import re
    if not expr:
        return None
    expr = expr.strip()

    # Plain "x" (no parens, no scaling) -- emitted for the identity symbol.
    if expr == "x":
        return "x", 1.0, 0.0, 1.0, 0.0

    m = re.match(
        r"^\s*(?:([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\*\s*)?"
        r"([A-Za-z_][A-Za-z0-9_/^]*)\s*\((.*)\)\s*(.*)$",
        expr,
    )
    if m is None:
        return None
    c_str, fun_name, inner, rest = m.groups()
    c = float(c_str) if c_str is not None else 1.0
    if fun_name not in PRIMITIVES:
        return None

    a_val, b_val = _parse_inner(inner)
    if a_val is None:
        return None

    d_val = 0.0
    rs = rest.strip()
    if rs.startswith("+"):
        try: d_val = float(rs[1:].strip())
        except ValueError: d_val = 0.0
    elif rs.startswith("-"):
        try: d_val = -float(rs[1:].strip())
        except ValueError: d_val = 0.0

    return fun_name, a_val, b_val, c, d_val


def _parse_inner(inner: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse the parenthesised ``a * x [+|- b]`` substring into (a, b)."""
    inner = inner.strip()
    if inner == "x":
        return 1.0, 0.0

    # ``a * x [+|- b]``  -- split at the first '*', then parse the tail.
    if "*" in inner:
        head, tail = inner.split("*", 1)
        try:
            a_val = float(head.strip())
        except ValueError:
            return None, None
        tail = tail.strip()
        # tail looks like "x", "x + b", "x - b", "b + x" (rare).
        if tail.startswith("x"):
            b_str = tail[1:].strip()
            b_val = _signed_const(b_str, default=0.0)
            return a_val, b_val
        # Handle rare case "b + x" (rearrange the tail).
        if tail.endswith("x"):
            b_str = tail[:-1].strip()
            b_val = _signed_const(b_str, default=0.0)
            return a_val, b_val
        return a_val, 0.0

    # No leading '*': ``x + b``, ``x - b``  (with possible whitespace).
    if inner.startswith("x"):
        tail = inner[1:].lstrip()
        if not tail:
            return 1.0, 0.0
        sign = 1.0 if tail[0] == "+" else (-1.0 if tail[0] == "-" else None)
        if sign is None:
            return 1.0, 0.0
        rest = tail[1:].strip()
        try:
            return 1.0, sign * float(rest) if rest else (1.0, 0.0)
        except ValueError:
            return 1.0, 0.0
    return None, None


def _signed_const(s: str, default: float = 0.0) -> float:
    """Parse ``'+b'`` / ``'-b'`` (with optional surrounding whitespace)."""
    s = s.strip()
    if not s:
        return default
    if s[0] == "+":
        try: return float(s[1:].strip())
        except ValueError: return default
    if s[0] == "-":
        try: return -float(s[1:].strip())
        except ValueError: return default
    try: return float(s)
    except ValueError: return default
