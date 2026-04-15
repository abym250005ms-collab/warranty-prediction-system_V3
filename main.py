from __future__ import annotations

import json

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

    metrics = {
        "claim_prediction": claim_result.metrics,
        "cost_forecasting": cost_result.metrics,
    }
    with open(config.METRICS_OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


if __name__ == "__main__":
    run_pipeline()
