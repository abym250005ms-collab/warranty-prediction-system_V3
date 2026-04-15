from __future__ import annotations

import json
import numpy as np

import config
from claim_prediction_xgboost import train_claim_model
from cost_forecasting_prophet_arima import forecast_costs
from data_preprocessing import (
    add_derived_features,
    build_claim_target,
    ensure_output_directories,
    load_warranty_data,
    save_claim_preprocessor,
)


def _aggregate_forecasting_metrics(variant_metrics: dict) -> dict:
    """Aggregate error metrics across all model variants"""
    all_rmse = []
    all_mae = []
    all_mape = []

    for variant, metrics in variant_metrics.items():
        backtest = metrics.get("backtest", {})
        
        if "rmse" in backtest and not np.isnan(backtest["rmse"]):
            all_rmse.append(backtest["rmse"])
        
        if "mae" in backtest and not np.isnan(backtest["mae"]):
            all_mae.append(backtest["mae"])
        
        if "mape" in backtest and not np.isnan(backtest["mape"]):
            all_mape.append(backtest["mape"])

    # Calculate aggregated metrics
    aggregated = {}
    
    if all_rmse:
        aggregated["rmse"] = float(np.mean(all_rmse))
    else:
        aggregated["rmse"] = None
    
    if all_mae:
        aggregated["mae"] = float(np.mean(all_mae))
    else:
        aggregated["mae"] = None
    
    if all_mape:
        aggregated["mape"] = float(np.mean(all_mape))
    else:
        aggregated["mape"] = None

    # Keep variant-level metrics as well
    aggregated["by_variant"] = variant_metrics

    return aggregated


def run_pipeline():
    ensure_output_directories()
    data = load_warranty_data()
    enriched_data = add_derived_features(data)
    labeled_data = build_claim_target(enriched_data)

    claim_result = train_claim_model(labeled_data)
    save_claim_preprocessor(claim_result.artifacts)

    claim_result.predictions.to_csv(config.CLAIM_OUTPUT_FILE, index=False)
    claim_result.feature_importance.to_csv(config.FEATURE_IMPORTANCE_FILE, index=False)

    cost_result = forecast_costs(enriched_data, periods=config.COST_FORECAST_MONTHS)
    cost_result.forecasts.to_csv(config.COST_OUTPUT_FILE, index=False)

    # Aggregate forecasting metrics
    aggregated_cost_metrics = _aggregate_forecasting_metrics(cost_result.metrics)

    metrics = {
        "claim_prediction": claim_result.metrics,
        "cost_forecasting": aggregated_cost_metrics,
    }
    with open(config.METRICS_OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


if __name__ == "__main__":
    run_pipeline()
