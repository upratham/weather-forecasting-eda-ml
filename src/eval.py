"""Evaluation helpers for forecasting and regression tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, np.nan, y_true)
    percentage_errors = np.abs((y_true - y_pred) / denominator) * 100.0
    return float(np.nanmean(percentage_errors))


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute standard regression metrics."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _safe_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    """Alias for forecast evaluation using the same metrics."""

    return evaluate_regression(y_true, y_pred)


def summarize_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionaries to a comparison table."""

    return pd.DataFrame(results).T.sort_values("rmse")
