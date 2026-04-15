from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_COLUMNS = [
    "claim_date",
    "failure_date",
    "manufacture_date",
    "sale_date",
    "warranty_start_date",
    "warranty_end_date",
]

NUMERIC_PRIORITY = [
    "odometer_at_failure_km",
    "months_in_service",
    "parts_cost_inr",
    "labour_cost_inr",
    "total_claim_cost_inr",
    "ambient_temp_celsius",
    "battery_capacity_kwh",
    "avg_daily_km",
    "avg_replacement_cost_inr",
    "expected_life_km",
    "warranty_coverage_months",
    "repair_duration_days",
    "service_capacity_score",
]

CATEGORICAL_PRIORITY = [
    "model_variant",
    "failure_mode",
    "subsystem",
    "use_case",
    "repair_type",
    "state",
    "zone",
    "season",
    "criticality",
    "motor_type",
]


def load_dataset(file_path: str | Path, sheet_name: str = "Clean_merge_data") -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_output_dirs(base_output: str | Path) -> dict[str, Path]:
    base = Path(base_output)
    plots = base / "eda_plots"
    paths = {
        "base": base,
        "plots": plots,
        "distributions": plots / "distributions",
        "correlations": plots / "correlations",
        "temporal": plots / "temporal",
        "risk_analysis": plots / "risk_analysis",
        "geographic": plots / "geographic",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def pick_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in NUMERIC_PRIORITY if c in numeric_cols]
    remaining = [c for c in numeric_cols if c not in preferred]
    return preferred + remaining


def pick_categorical_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if (df[c].dtype == "object" or str(df[c].dtype) == "category")]
    preferred = [c for c in CATEGORICAL_PRIORITY if c in cols]
    remaining = [c for c in cols if c not in preferred]
    return preferred + remaining


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat() if not pd.isna(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT:
        return None
    return value


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)

