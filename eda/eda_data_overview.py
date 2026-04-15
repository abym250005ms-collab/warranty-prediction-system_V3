from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import DATE_COLUMNS, pick_categorical_columns, pick_numeric_columns

matplotlib.use("Agg")


def run_data_overview(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
    missing_df.to_csv(out / "missing_values.csv", index=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_df.index[:20], y=missing_df["missing_pct"].values[:20], color="#4C72B0")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing (%)")
    plt.title("Top 20 Columns by Missing Value Percentage")
    plt.tight_layout()
    plt.savefig(out / "missing_values_top20.png", dpi=150)
    plt.close()

    categorical_cols = pick_categorical_columns(df)
    numeric_cols = pick_numeric_columns(df)
    date_ranges = {}
    for col in DATE_COLUMNS:
        if col in df.columns:
            date_ranges[col] = {
                "min": df[col].min(),
                "max": df[col].max(),
                "missing_count": int(df[col].isna().sum()),
            }

    cat_value_counts = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False).head(20)
        cat_value_counts[col] = vc.to_dict()

    report = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 3),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": missing_df.to_dict(orient="index"),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_quality": {
            "complete_rows_pct": round((1 - (df.isna().any(axis=1).sum() / len(df))) * 100, 2),
            "columns_with_missing": int((missing_counts > 0).sum()),
        },
        "numeric_summary": df[numeric_cols].describe(include="all").to_dict() if numeric_cols else {},
        "categorical_value_counts_top20": cat_value_counts,
        "date_ranges": date_ranges,
    }
    return report

