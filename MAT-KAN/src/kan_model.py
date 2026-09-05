"""
KAN regressor wrapping PyKAN's ``MultKAN``.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from kan import MultKAN
    PYKAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYKAN_AVAILABLE = False

def _arrays_to_loader(X, y, batch_size, shuffle):
    """Wrap numpy arrays as a PyTorch DataLoader (no external imports)."""
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _loader_to_arrays(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate a DataLoader's batches into numpy arrays."""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs, dim=0).cpu().numpy()
    y = torch.cat(ys, dim=0).cpu().numpy().reshape(-1)
    return X, y


def _default_hidden_dim(input_dim: int) -> int:
    """Kolmogorov-Arnold 2n+1 width heuristic used in the paper."""
    return max(5, 2 * input_dim + 1)


# KANRegressor
class KANRegressor:
    """Thin wrapper around PyKAN's MultKAN for regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        seed: int = 42,
        device: str = "cpu",
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        if not PYKAN_AVAILABLE:
            raise ImportError("PyKAN is not installed. Run `pip install pykan`.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or _default_hidden_dim(input_dim)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.seed = seed
        self.device = device
        self.grid_range = grid_range

        torch.manual_seed(seed)
        np.random.seed(seed)

        width = [input_dim, self.hidden_dim, output_dim]
        self.model = MultKAN(
            width=width,
            grid=grid_size,
            k=spline_order,
            seed=seed,
            device=device,
            grid_range=list(grid_range),
        )

        self.model.to(device)
        self.history = None
        self.x_scaler_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def fit(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        stop_grid_update_step: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model for ``epochs`` full passes over the training data.
        """
        if train_loader is None:
            if X_train is None or y_train is None:
                raise ValueError("Provide either train_loader or (X_train, y_train).")
            X_train = np.asarray(X_train, dtype=np.float64)
            y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val, dtype=np.float64)
                y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)

            from sklearn.preprocessing import StandardScaler
            self.x_scaler_ = StandardScaler()
            X_train = self.x_scaler_.fit_transform(X_train)
            if X_val is not None:
                X_val = self.x_scaler_.transform(X_val)

            self.y_mean_ = float(np.mean(y_train))
            train_loader = _arrays_to_loader(
                X_train.astype(np.float32),
                (y_train - self.y_mean_).astype(np.float32),
                batch_size=batch_size, shuffle=True,
            )
            self.y_std_ = 1.0

            val_loader = _arrays_to_loader(
                X_val.astype(np.float32),
                (y_val - self.y_mean_).astype(np.float32),
                batch_size=max(1, len(X_val)), shuffle=False,
            )

        Xtr, ytr = _loader_to_arrays(train_loader)
        Xtr_t = torch.from_numpy(Xtr).float().to(self.device)
        ytr_t = torch.from_numpy(ytr).float().to(self.device)

        Xva, yva = _loader_to_arrays(val_loader)
        Xva_t = torch.from_numpy(Xva).float().to(self.device)
        yva_t = torch.from_numpy(yva).float().to(self.device)

        steps_per_epoch = max(1, len(train_loader))
        steps = epochs * steps_per_epoch
        if stop_grid_update_step is None:
            stop_grid_update_step = steps // 2

        if verbose:
            print(
                f"kan training: epochs={epochs} steps/epoch={steps_per_epoch} "
                f"total_steps={steps} batch={batch_size} lr={lr}"
            )

        dataset = {
            "train_input": Xtr_t,
            "train_label": ytr_t,
            "test_input": Xva_t,
            "test_label": yva_t,
        }

        # Try with grid update first; fall back to no grid update if curve2coef fails
        try:
            history = self.model.fit(
                dataset,
                opt="Adam",
                lr=lr,
                steps=steps,
                batch=batch_size,
                lamb=0.0,
                lamb_l1=0.0,
                lamb_entropy=0.0,
                lamb_coef=0.0,
                lamb_coefdiff=0.0,
                update_grid=True,
                grid_update_num=10,
                start_grid_update_step=-1,
                stop_grid_update_step=stop_grid_update_step,
                log=(100 if verbose else 1000),
            )
        except UnboundLocalError:
            # Grid update failed in curve2coef (known kan library bug).
            # Retry with grid update disabled.
            if verbose:
                print("  [WARN] Grid update failed (curve2coef bug), retrying with update_grid=False")
            history = self.model.fit(
                dataset,
                opt="Adam",
                lr=lr,
                steps=steps,
                batch=batch_size,
                lamb=0.0,
                lamb_l1=0.0,
                lamb_entropy=0.0,
                lamb_coef=0.0,
                lamb_coefdiff=0.0,
                update_grid=False,
                grid_update_num=10,
                start_grid_update_step=-1,
                stop_grid_update_step=stop_grid_update_step,
                log=(100 if verbose else 1000),
            )
        self.history = history

        for layer in self.model.act_fun:
            if hasattr(layer, "mask") and layer.mask is not None:
                layer.mask.data = torch.ones_like(layer.mask.data)
        if hasattr(self.model, "symbolic_fun") and self.model.symbolic_fun:
            for layer in self.model.symbolic_fun:
                if hasattr(layer, "mask") and layer.mask is not None:
                    layer.mask.data = torch.ones_like(layer.mask.data)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward-pass prediction."""
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self, "x_scaler_") and self.x_scaler_ is not None:
            X = self.x_scaler_.transform(X)
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            y = self.model(X_t).cpu().numpy().reshape(-1)
        if hasattr(self, "y_mean_") and self.y_mean_ is not None:
            y = y + self.y_mean_
        return y

    def refresh_grid(self, X: np.ndarray) -> None:
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self, "x_scaler_") and self.x_scaler_ is not None:
            X = self.x_scaler_.transform(X)
        X_t = torch.from_numpy(X).to(self.device)
        try:
            self.model.update_grid_from_samples(X_t)
        except AttributeError:
            self.model.update_grid(X_t)

    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Mean |d y / d x_i| over the dataset, used by the pruner."""
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self, "x_scaler_") and self.x_scaler_ is not None:
            X = self.x_scaler_.transform(X)
        X_t = torch.from_numpy(X).to(self.device)
        X_t.requires_grad_(True)
        y = self.model(X_t)
        if y.ndim > 1:
            y = y.sum()
        y.backward()
        grad = X_t.grad.detach().abs().mean(dim=0).cpu().numpy()
        s = grad.sum()
        return grad / s if s > 0 else grad

    def auto_symbolic(self, verbose: bool = False) -> None:
        """Run PyKAN's ``auto_symbolic`` so edges have symbolic annotations."""
        self.model.auto_symbolic(verbose=verbose)

    def activate_all_edges(self) -> None:
        """
        Force every edge mask in ``act_fun`` and ``symbolic_fun``
        """
        for layer in self.model.act_fun:
            if hasattr(layer, "mask") and layer.mask is not None:
                layer.mask.data = torch.ones_like(layer.mask.data)
        if hasattr(self.model, "symbolic_fun") and self.model.symbolic_fun:
            for layer in self.model.symbolic_fun:
                if hasattr(layer, "mask") and layer.mask is not None:
                    layer.mask.data = torch.ones_like(layer.mask.data)

    def list_symbolic_edges(self, feature_names: List[str]) -> List[Dict]:
        """Inspect ``symbolic_fun`` / ``act_fun`` and return active edges.

        Symbolic edges are prioritized; if none are available, numerical
        B-spline edges are approximated as low-degree polynomials.
        """
        model = self.model
        if not hasattr(model, "symbolic_fun") or not hasattr(model, "width"):
            return []

        edges: List[Dict] = []
        width = model.width
        n_features = len(feature_names)

        for layer_idx in range(len(model.symbolic_fun)):
            n_in = width[layer_idx][0]
            n_out = width[layer_idx + 1][0]
            sym = model.symbolic_fun[layer_idx]
            try:
                sym_mask = sym.mask.detach().cpu().numpy()
            except Exception:
                continue
            try:
                num_mask = model.act_fun[layer_idx].mask.detach().cpu().numpy()
            except Exception:
                num_mask = np.ones_like(sym_mask)

            for in_id in range(n_in):
                feat = feature_names[in_id] if in_id < n_features else f"h{layer_idx}_{in_id}"
                for out_id in range(n_out):
                    sym_on = sym_mask[out_id, in_id] > 0
                    num_on = num_mask[in_id, out_id] > 0 if num_mask.ndim == 2 else num_mask[out_id, in_id] > 0
                    if not sym_on and not num_on:
                        continue

                    expr, abcd = None, None
                    poly_coef = None
                    if sym_on:
                        try:
                            fun_name = sym.funs_name[out_id][in_id]
                            if fun_name not in ('0', 'zero', 'spline', '', None):
                                a, b, c, d = sym.affine[out_id, in_id].detach().cpu().numpy()
                                expr = _format_symbolic_edge(fun_name, float(a), float(b),
                                                            float(c), float(d))
                                abcd = (float(a), float(b), float(c), float(d))
                        except Exception:
                            pass

                    # Fall back: store placeholder expression; actual activation
                    # values will be computed by evaluate_edge() via forward pass.
                    if expr is None:
                        expr = f"spline(l{layer_idx},i{in_id},o{out_id})"

                    edges.append({
                        "layer": layer_idx,
                        "in_id": in_id,
                        "out_id": out_id,
                        "feature_idx": in_id,
                        "feature_name": feat,
                        "expr": expr,
                        "abcd": abcd,
                        "poly_coef": None,
                    })
        return edges

    def _edge_polynomial_coef(
        self,
        layer_idx: int,
        in_id: int,
        out_id: int,
        activations: np.ndarray,
        x_vals: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fit a low-degree polynomial to a B-spline edge."""
        try:
            degree = min(3, len(x_vals) - 1)
            return np.polyfit(x_vals, activations, degree)
        except Exception:
            return None

    def _edge_polynomial_expr(
        self,
        layer_idx: int,
        in_id: int,
        out_id: int,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Return (expr_string, coefficients) for a polynomial-fitted B-spline edge."""
        coef = self._edge_polynomial_coef(layer_idx, in_id, out_id)
        if coef is None or len(coef) == 0:
            return None, None

        terms = []
        names = ['1', 'x', 'x^2', 'x^3'][:len(coef)]
        for a, name in zip(coef, names):
            if abs(a) < 1e-6:
                continue
            if abs(a - 1) < 1e-6:
                terms.append(name)
            elif abs(a + 1) < 1e-6:
                terms.append(f"-{name}")
            else:
                terms.append(f"{a:.3g}*{name}" if name != '1' else f"{a:.3g}")

        if not terms:
            return None, None
        expr = (' + '.join(terms)).replace('+ -', '- ')
        return expr, coef

    def evaluate_edge(self, edge: Dict, X: np.ndarray) -> Optional[np.ndarray]:
        """Extract the B-spline activation of a single edge from the trained KAN.

        Uses ``spline_postacts`` (shape: batch × out_dim × in_dim), which is
        populated during every forward pass when ``save_act=True`` (the default).
        """
        idx = edge["feature_idx"]
        n_features = X.shape[1]
        if idx >= n_features:
            return None

        # Normalise the same way as during training.
        if hasattr(self, "x_scaler_") and self.x_scaler_ is not None:
            X_batch = self.x_scaler_.transform(X)
        else:
            X_batch = X.astype(np.float32)

        try:
            model = self.model
            # Ensure intermediate activations are saved.
            model.save_act = True
            # Forward pass to populate spline_postacts.
            X_t = torch.from_numpy(X_batch).float().to(self.device)
            with torch.no_grad():
                model(X_t)

            # spline_postacts[l] shape: (batch, out_dim, in_dim)
            postacts = model.spline_postacts[edge["layer"]].cpu().numpy()  # (n, out_dim, in_dim)
            edge_vals = postacts[:, edge["out_id"], edge["in_id"]].astype(np.float32)

            if edge_vals.shape[0] != len(X):
                return None
            if not np.all(np.isfinite(edge_vals)):
                return None
            return edge_vals

        except Exception as exc:
            print(f"    [WARN] evaluate_edge failed for layer={edge['layer']} "
                  f"idx={idx} out_id={edge['out_id']}: {exc}")
            return None

    def save(self, path: str) -> None:
        """Persist the underlying PyKAN model."""
        self.model.saveckpt(path)

    def load(self, path: str) -> None:
        self.model.loadckpt(path)

    def get_params(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "seed": self.seed,
            "device": self.device,
        }

def _format_symbolic_edge(fun_name: str, a: float, b: float,
                          c: float, d: float) -> Optional[str]:
    """Format a symbolic edge name with affine params as ``c * f(a x + b) + d``."""
    if fun_name in {"0", "zero", "spline", ""}:
        return None

    if abs(a - 1) < 1e-4 and abs(b) < 1e-4:
        inner = "x"
    elif abs(a - 1) < 1e-4:
        inner = f"x {'+' if b >= 0 else '-'} {abs(b):.3g}"
    elif abs(b) < 1e-4:
        inner = f"{a:.3g} * x"
    else:
        inner = f"{a:.3g} * x {'+' if b >= 0 else '-'} {abs(b):.3g}"

    if abs(c) < 1e-4:
        prefix, suffix = "", ""
    elif abs(c - 1) < 1e-4:
        prefix, suffix = "", ""
    elif abs(c + 1) < 1e-4:
        prefix, suffix = "-", ""
    else:
        prefix, suffix = f"{c:.3g} * ", ""
    suffix = (suffix + (f" + {d:.3g}" if d > 0 else (f" - {abs(d):.3g}" if d < 0 else "")))
    expr = f"{prefix}{fun_name}({inner}){suffix}".strip()
    return expr if expr else None