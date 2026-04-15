from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


DATE_COLUMNS = [
    "claim_date",
    "failure_date",
    "manufacture_date",
    "sale_date",
    "warranty_start_date",
    "warranty_end_date",
]

NUMERIC_BASE_COLUMNS = [
    "odometer_at_failure_km",
    "months_in_service",
    "repair_duration_days",
    "parts_cost_inr",
    "labour_cost_inr",
    "total_claim_cost_inr",
    "ambient_temp_celsius",
    "battery_capacity_kwh",
    "avg_daily_km",
    "avg_replacement_cost_inr",
    "expected_life_km",
    "warranty_coverage_months",
    "service_capacity_score",
]


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.lower_bounds_ = frame.quantile(self.lower_quantile, axis=0).to_numpy()
        self.upper_bounds_ = frame.quantile(self.upper_quantile, axis=0).to_numpy()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        clipped = frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return clipped.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


@dataclass
class ClaimPreprocessorArtifacts:
    preprocessor: ColumnTransformer
    feature_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]



def load_warranty_data(path: str | None = None, sheet_name: str | None = None) -> pd.DataFrame:
    data_path = path or config.DATA_FILE
    sheet = sheet_name or config.DATA_SHEET
    df = pd.read_excel(data_path, sheet_name=sheet)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_BASE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["parts_to_labour_ratio"] = out["parts_cost_inr"] / (out["labour_cost_inr"].replace(0, np.nan))
    out["cost_per_km"] = out["total_claim_cost_inr"] / (out["odometer_at_failure_km"].replace(0, np.nan))
    out["avg_daily_usage_ratio"] = out["odometer_at_failure_km"] / (out["months_in_service"].replace(0, np.nan) * 30)
    out["service_coverage_ratio"] = out["months_in_service"] / (out["warranty_coverage_months"].replace(0, np.nan))
    out["claim_month"] = out["claim_date"].dt.to_period("M").dt.to_timestamp()

    season_map = {
        "spring": 0,
        "summer": np.pi / 2,
        "monsoon": np.pi,
        "autumn": 3 * np.pi / 2,
        "winter": 2 * np.pi,
    }
    season_angle = out["season"].astype(str).str.lower().map(season_map)
    out["season_sin"] = np.sin(season_angle)
    out["season_cos"] = np.cos(season_angle)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_claim_target(df: pd.DataFrame, horizon_months: int = config.CLAIM_FORECAST_MONTHS) -> pd.DataFrame:
    frame = df.sort_values(["vehicle_id", "claim_date"]).copy()
    frame["next_claim_date"] = frame.groupby("vehicle_id")["claim_date"].shift(-1)
    horizon_days = int(30 * horizon_months)
    frame["target_claim_next_3m"] = (
        (frame["next_claim_date"] - frame["claim_date"]).dt.days.le(horizon_days)
    ).fillna(False).astype(int)
    return frame


def _get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    ignored = set(config.CLAIM_EXCLUDED_COLUMNS) | {"next_claim_date", "target_claim_next_3m", "claim_month"}
    candidate = [c for c in df.columns if c not in ignored]

    categorical_cols = [
        c
        for c in candidate
        if c in df.columns and (c in config.CATEGORICAL_COLUMNS or df[c].dtype == "object")
    ]
    numeric_cols = [c for c in candidate if c not in categorical_cols]
    return candidate, categorical_cols, numeric_cols


def fit_claim_preprocessor(df: pd.DataFrame) -> ClaimPreprocessorArtifacts:
    feature_columns, categorical_cols, numeric_cols = _get_feature_columns(df)
    lower_q, upper_q = config.NUMERIC_CLIP_QUANTILES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower_q, upper_q)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    preprocessor.fit(df[feature_columns])
    return ClaimPreprocessorArtifacts(preprocessor, feature_columns, categorical_cols, numeric_cols)


def transform_claim_features(df: pd.DataFrame, artifacts: ClaimPreprocessorArtifacts):
    return artifacts.preprocessor.transform(df[artifacts.feature_columns])


def save_claim_preprocessor(artifacts: ClaimPreprocessorArtifacts, path: str | None = None) -> None:
    save_path = path or config.PREPROCESSOR_FILE
    joblib.dump(artifacts, save_path)


def prepare_claim_dataset() -> Tuple[pd.DataFrame, ClaimPreprocessorArtifacts]:
    raw = load_warranty_data()
    enriched = add_derived_features(raw)
    labeled = build_claim_target(enriched)
    labeled = labeled.dropna(subset=["claim_date", "vehicle_id"])
    artifacts = fit_claim_preprocessor(labeled)
    return labeled, artifacts


def ensure_output_directories() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
