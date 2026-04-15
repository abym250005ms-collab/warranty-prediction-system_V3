from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def run_temporal_analysis(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    if "claim_date" in df.columns:
        monthly_claims = (
            df.dropna(subset=["claim_date"])
            .assign(month=lambda x: x["claim_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")
            .size()
            .rename("claim_count")
        )
        report["monthly_claim_count"] = monthly_claims.to_dict()
        plt.figure(figsize=(11, 4))
        monthly_claims.plot(kind="line")
        plt.title("Monthly claim count trend")
        plt.ylabel("Claim count")
        plt.tight_layout()
        plt.savefig(out / "monthly_claim_count_trend.png", dpi=150)
        plt.close()

        if "total_claim_cost_inr" in df.columns:
            monthly_cost = (
                df.dropna(subset=["claim_date"])
                .assign(month=lambda x: x["claim_date"].dt.to_period("M").dt.to_timestamp())
                .groupby("month")["total_claim_cost_inr"]
                .sum()
            )
            report["monthly_total_cost"] = monthly_cost.to_dict()
            plt.figure(figsize=(11, 4))
            monthly_cost.plot(kind="line", color="#C44E52")
            plt.title("Monthly total claim cost trend")
            plt.ylabel("Total cost (INR)")
            plt.tight_layout()
            plt.savefig(out / "monthly_total_cost_trend.png", dpi=150)
            plt.close()

    if "season" in df.columns:
        season_count = df["season"].fillna("Missing").value_counts()
        report["season_counts"] = season_count.to_dict()
        plt.figure(figsize=(8, 4))
        sns.barplot(x=season_count.index, y=season_count.values, color="#55A868")
        plt.title("Claims by season")
        plt.tight_layout()
        plt.savefig(out / "claims_by_season.png", dpi=150)
        plt.close()

    if {"season", "total_claim_cost_inr"}.issubset(df.columns):
        season_cost = df.groupby("season", dropna=False)["total_claim_cost_inr"].mean().sort_values(ascending=False)
        report["season_avg_cost"] = season_cost.to_dict()
        plt.figure(figsize=(8, 4))
        sns.barplot(x=season_cost.index.astype(str), y=season_cost.values, color="#C44E52")
        plt.title("Average cost by season")
        plt.tight_layout()
        plt.savefig(out / "cost_by_season.png", dpi=150)
        plt.close()

    if "months_in_service" in df.columns:
        bins = [0, 3, 6, 12, np.inf]
        labels = ["0-3", "3-6", "6-12", "12+"]
        bucket = pd.cut(df["months_in_service"], bins=bins, labels=labels, right=False)
        claims_by_age = bucket.value_counts().reindex(labels, fill_value=0)
        report["claims_by_months_in_service_bins"] = claims_by_age.to_dict()

        plt.figure(figsize=(8, 4))
        sns.barplot(x=claims_by_age.index, y=claims_by_age.values, color="#55A868")
        plt.title("Claims by months in service bins")
        plt.tight_layout()
        plt.savefig(out / "claims_by_month_bins.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.histplot(df["months_in_service"].dropna(), kde=True, color="#8172B3")
        plt.title("Distribution of months_in_service at failure")
        plt.tight_layout()
        plt.savefig(out / "months_in_service_distribution.png", dpi=150)
        plt.close()

    if {"months_in_service", "total_claim_cost_inr"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=df, x="months_in_service", y="total_claim_cost_inr", scatter_kws={"alpha": 0.35, "s": 20})
        plt.title("Cost increase with vehicle age")
        plt.tight_layout()
        plt.savefig(out / "cost_vs_vehicle_age.png", dpi=150)
        plt.close()

    return report
