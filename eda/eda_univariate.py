from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

from .common import pick_categorical_columns, pick_numeric_columns

matplotlib.use("Agg")

KEY_NUMERIC = [
    "odometer_at_failure_km",
    "total_claim_cost_inr",
    "months_in_service",
    "battery_capacity_kwh",
    "ambient_temp_celsius",
]
KEY_CATEGORICAL = ["model_variant", "failure_mode", "subsystem", "use_case"]


def _numeric_stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {}
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((clean < low) | (clean > high)).sum()
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "skewness": float(skew(clean, bias=False)) if len(clean) > 2 else 0.0,
        "kurtosis": float(kurtosis(clean, bias=False)) if len(clean) > 3 else 0.0,
        "outlier_count_iqr": int(outliers),
        "outlier_pct_iqr": float(round((outliers / len(clean)) * 100, 2)),
    }


def run_univariate_analysis(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    numeric_cols = pick_numeric_columns(df)
    categorical_cols = pick_categorical_columns(df)

    key_numeric = [c for c in KEY_NUMERIC if c in numeric_cols] or numeric_cols[:6]
    key_categorical = [c for c in KEY_CATEGORICAL if c in categorical_cols] or categorical_cols[:6]

    numeric_report = {}
    for col in key_numeric:
        numeric_report[col] = _numeric_stats(df[col])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=axes[0], color="#2E86AB")
        axes[0].set_title(f"{col} distribution")
        sns.boxplot(x=df[col], ax=axes[1], color="#F18F01")
        axes[1].set_title(f"{col} box plot")
        plt.tight_layout()
        plt.savefig(out / f"{col}_distribution_box.png", dpi=150)
        plt.close(fig)

    categorical_report = {}
    for col in key_categorical:
        vc = df[col].fillna("Missing").value_counts()
        top10 = vc.head(10)
        categorical_report[col] = {
            "unique_count": int(df[col].nunique(dropna=True)),
            "top10_counts": top10.to_dict(),
            "top10_pct": (top10 / len(df) * 100).round(2).to_dict(),
        }
        plt.figure(figsize=(10, 4))
        sns.barplot(x=top10.values, y=top10.index, color="#4C72B0")
        plt.title(f"{col} top 10 categories")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out / f"{col}_top10.png", dpi=150)
        plt.close()

    return {
        "numerical_features": numeric_report,
        "categorical_features": categorical_report,
        "all_numeric_columns_count": len(numeric_cols),
        "all_categorical_columns_count": len(categorical_cols),
    }
