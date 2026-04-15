# Warranty Prediction System V3

Comprehensive warranty analytics pipeline for Montra Electric vehicles.

## Objectives

1. **Warranty Claim Prediction (next 3 months)**
   - Model: **XGBoost binary classifier**
   - Output: `vehicle_id`, `model_variant`, `risk_score`, `risk_rank`

2. **Monthly Warranty Cost Forecasting (next 12 months)**
   - Models: **Prophet + ARIMA ensemble**
   - Grouping: `model_variant`
   - Output includes 80% and 95% confidence intervals.

## Project Structure

- `data_preprocessing.py`
- `claim_prediction_xgboost.py`
- `cost_forecasting_prophet_arima.py`
- `model_evaluation.py`
- `main.py`
- `config.py`
- `requirements.txt`

## Data

- Input file: `warranty_dataset.xlsx`
- Sheet: `Clean_merge_data`

## Preprocessing

- Missing value imputation
- Outlier clipping using quantiles
- Numerical normalization via `StandardScaler`
- Categorical one-hot encoding (`drop='first'`)
- Derived features:
  - `cost_per_km`
  - `parts_to_labour_ratio`
  - `avg_daily_usage_ratio`
  - `service_coverage_ratio`
  - cyclic season encoding (`season_sin`, `season_cos`)

## Claim Prediction (XGBoost)

- Target: `target_claim_next_3m`
- Class imbalance handling: `scale_pos_weight`
- Hyperparameter tuning: `GridSearchCV`
- Cross-validation: 5-fold stratified
- Metrics: AUC-ROC, Precision, Recall, F1, Confusion Matrix
- Feature importance exported.

## Cost Forecasting (Prophet + ARIMA)

- Monthly aggregation of `total_claim_cost_inr` per `model_variant`
- Stationarity check: ADF test
- ARIMA order selection: `auto_arima`
- Ensemble: `0.6 * Prophet + 0.4 * ARIMA`
- Forecast horizon: 12 months
- Confidence intervals: 80%, 95%
- Includes variant-level backtesting metrics.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

## Outputs

Generated in `outputs/`:

- `predictions_claim_risk.csv`
- `predictions_cost_forecast.csv`
- `xgboost_feature_importance.csv`
- `model_metrics.json`

Saved deployment artifacts in `artifacts/`:

- `claim_preprocessor.joblib`
- `xgboost_claim_model.joblib`
