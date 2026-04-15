from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def evaluate_forecast(actual, predicted) -> Dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual_arr, pred_arr)))
    mae = float(mean_absolute_error(actual_arr, pred_arr))

    # Calculate MAPE with better handling of edge cases
    non_zero_mask = actual_arr != 0
    if non_zero_mask.any():
        # Calculate MAPE only for non-zero actual values
        mape_values = np.abs((actual_arr[non_zero_mask] - pred_arr[non_zero_mask]) / actual_arr[non_zero_mask]) * 100
        
        # Filter out extreme outliers (e.g., MAPE > 500%)
        mape_values = mape_values[mape_values <= 500]
        
        if len(mape_values) > 0:
            mape = float(np.mean(mape_values))
        else:
            mape = float("nan")
    else:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape}


def backtest_last_window(series, forecast_values, horizon: int):
    if len(series) < horizon * 2:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    actual = series[-horizon:]
    predicted = forecast_values[:horizon]
    return evaluate_forecast(actual, predicted)
