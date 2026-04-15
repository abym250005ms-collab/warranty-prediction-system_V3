from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import pick_numeric_columns

matplotlib.use("Agg")


def run_relationship_analysis(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    numeric_cols = pick_numeric_columns(df)
    corr = df[numeric_cols].corr(numeric_only=True) if numeric_cols else pd.DataFrame()

    if not corr.empty:
        top_cols = corr.abs().sum().sort_values(ascending=False).head(15).index
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr.loc[top_cols, top_cols], cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (top 15 numeric features)")
        plt.tight_layout()
        plt.savefig(out / "correlation_heatmap_top15.png", dpi=150)
        plt.close()

    target_col = "total_claim_cost_inr"
    corr_with_cost = {}
    if target_col in corr.columns:
        corr_with_cost = corr[target_col].drop(target_col, errors="ignore").sort_values(ascending=False).head(10).to_dict()

    group_features = ["model_variant", "subsystem", "failure_mode", "repair_type", "use_case", "zone", "state"]
    cost_group_stats = {}
    for col in group_features:
        if col in df.columns and target_col in df.columns:
            stats = (
                df.groupby(col, dropna=False)[target_col]
                .agg(["count", "mean", "median"])
                .sort_values("mean", ascending=False)
                .head(15)
            )
            cost_group_stats[col] = stats.to_dict(orient="index")
            plt.figure(figsize=(11, 5))
            sns.boxplot(data=df, x=col, y=target_col, showfliers=False)
            plt.xticks(rotation=70, ha="right")
            plt.title(f"Claim cost by {col}")
            plt.tight_layout()
            plt.savefig(out / f"cost_by_{col}.png", dpi=150)
            plt.close()

    if {"odometer_at_failure_km", "total_claim_cost_inr"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="odometer_at_failure_km", y="total_claim_cost_inr", alpha=0.4, s=18)
        plt.title("Odometer at failure vs claim cost")
        plt.tight_layout()
        plt.savefig(out / "odometer_vs_cost.png", dpi=150)
        plt.close()

    failure_rate_stats = {}
    if "repeat_claim_flag" in df.columns:
        repeat_series = pd.to_numeric(df["repeat_claim_flag"], errors="coerce")
        for col in ["model_variant", "subsystem", "criticality", "months_in_service"]:
            if col in df.columns:
                tmp = pd.DataFrame({col: df[col], "repeat_claim_flag": repeat_series}).dropna()
                if tmp.empty:
                    continue
                grouped = tmp.groupby(col)["repeat_claim_flag"].mean().sort_values(ascending=False).head(15)
                failure_rate_stats[col] = grouped.to_dict()

    return {
        "correlation_with_total_claim_cost_top10": corr_with_cost,
        "cost_group_stats": cost_group_stats,
        "repeat_claim_pattern_stats": failure_rate_stats,
    }

