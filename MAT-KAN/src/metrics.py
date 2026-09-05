"""Regression metrics used throughout MAT-KAN."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return R^2 / RMSE / MAE / MAPE with NaN-safe filtering."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "n": 0}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.where(np.abs(y_true) < 1e-10, 1e-10, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "n": int(len(y_true))}