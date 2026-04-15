from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def generate_visualizations(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = []

    if {"claim_date", "model_variant"}.issubset(df.columns):
        monthly_mv = (
            df.dropna(subset=["claim_date"])
            .assign(month=lambda x: x["claim_date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "model_variant"])
            .size()
            .unstack(fill_value=0)
        )
        top_mv = monthly_mv.sum().sort_values(ascending=False).head(6).index
        plt.figure(figsize=(12, 5))
        monthly_mv[top_mv].plot(ax=plt.gca())
        plt.title("Claim count over time by model variant (top 6)")
        plt.ylabel("Claim count")
        plt.tight_layout()
        path = out / "claim_count_by_model_variant_over_time.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if {"claim_date", "model_variant", "total_claim_cost_inr"}.issubset(df.columns):
        monthly_cost = (
            df.dropna(subset=["claim_date"])
            .assign(month=lambda x: x["claim_date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "model_variant"])["total_claim_cost_inr"]
            .sum()
            .unstack(fill_value=0)
        )
        top_mv = monthly_cost.sum().sort_values(ascending=False).head(6).index
        plt.figure(figsize=(12, 5))
        monthly_cost[top_mv].plot(ax=plt.gca())
        plt.title("Monthly cost trends by model variant (top 6)")
        plt.ylabel("Total cost (INR)")
        plt.tight_layout()
        path = out / "monthly_cost_by_model_variant.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if "failure_mode" in df.columns:
        top = df["failure_mode"].fillna("Missing").value_counts().head(10)
        plt.figure(figsize=(10, 4))
        sns.barplot(x=top.values, y=top.index, color="#4C72B0")
        plt.title("Failure mode distribution (top 10)")
        plt.tight_layout()
        path = out / "failure_mode_top10.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if "subsystem" in df.columns:
        top = df["subsystem"].fillna("Missing").value_counts().head(15)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index, color="#55A868")
        plt.title("Subsystem failure rates")
        plt.tight_layout()
        path = out / "subsystem_failure_rates.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if {"model_variant", "repeat_claim_flag"}.issubset(df.columns):
        tmp = (
            pd.DataFrame(
                {
                    "model_variant": df["model_variant"],
                    "repeat_claim_flag": pd.to_numeric(df["repeat_claim_flag"], errors="coerce"),
                }
            )
            .dropna()
            .groupby("model_variant")["repeat_claim_flag"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        plt.figure(figsize=(10, 4))
        sns.barplot(x=tmp.index, y=tmp.values, color="#C44E52")
        plt.xticks(rotation=60, ha="right")
        plt.title("Repeat claim percentage by model variant (top 10)")
        plt.tight_layout()
        path = out / "repeat_claim_pct_by_model_variant.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if {"failure_mode", "total_claim_cost_inr"}.issubset(df.columns):
        top_modes = df["failure_mode"].fillna("Missing").value_counts().head(10).index
        tmp = df[df["failure_mode"].isin(top_modes)]
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=tmp, x="failure_mode", y="total_claim_cost_inr", showfliers=False)
        plt.xticks(rotation=60, ha="right")
        plt.title("Cost distribution by failure mode (top 10)")
        plt.tight_layout()
        path = out / "cost_distribution_by_failure_mode.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    if {"state"}.issubset(df.columns):
        top_state = df["state"].fillna("Missing").value_counts().head(20)
        plt.figure(figsize=(12, 5))
        sns.barplot(x=top_state.index, y=top_state.values, color="#4C72B0")
        plt.xticks(rotation=60, ha="right")
        plt.title("Geographic heatmap proxy: claims by state (top 20)")
        plt.tight_layout()
        path = out / "claims_by_state_top20.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(str(path))

    return {"generated_plots": generated}
