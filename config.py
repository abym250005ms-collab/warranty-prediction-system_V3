from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "warranty_dataset.xlsx"
DATA_SHEET = "Clean_merge_data"
OUTPUT_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CLAIM_OUTPUT_FILE = OUTPUT_DIR / "predictions_claim_risk.csv"
COST_OUTPUT_FILE = OUTPUT_DIR / "predictions_cost_forecast.csv"
METRICS_OUTPUT_FILE = OUTPUT_DIR / "model_metrics.json"

PREPROCESSOR_FILE = ARTIFACTS_DIR / "claim_preprocessor.joblib"
CLAIM_MODEL_FILE = ARTIFACTS_DIR / "xgboost_claim_model.joblib"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"

CLAIM_EXCLUDED_COLUMNS = {
    "claim_id",
    "vehicle_id",
    "vin_number",
    "dealer_name",
    "supplier_code",
    "claim_date",
    "failure_date",
    "manufacture_date",
    "sale_date",
    "warranty_start_date",
    "warranty_end_date",
}

CATEGORICAL_COLUMNS = [
    "component_id",
    "dealer_id",
    "failure_mode",
    "fault_code",
    "repair_type",
    "repeat_claim_flag",
    "season",
    "model_variant",
    "motor_type",
    "state",
    "city",
    "zone",
    "use_case",
    "component_name",
    "subsystem",
    "criticality",
    "supplier_code",
    "dealer_name",
]

NUMERIC_CLIP_QUANTILES = (0.01, 0.99)

XGB_PARAM_GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

XGB_BASE_PARAMS = {
    "n_estimators": 300,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

CLAIM_FORECAST_MONTHS = 3
COST_FORECAST_MONTHS = 12
ENSEMBLE_WEIGHTS = {"prophet": 0.6, "arima": 0.4}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
