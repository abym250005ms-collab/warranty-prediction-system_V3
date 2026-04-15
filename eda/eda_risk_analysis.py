from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def run_risk_analysis(df: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    repeat = pd.to_numeric(df["repeat_claim_flag"], errors="coerce") if "repeat_claim_flag" in df.columns else None
    if repeat is not None:
        report["overall_repeat_claim_rate"] = float(repeat.mean())

    if "vehicle_id" in df.columns:
        claims_per_vehicle = df.groupby("vehicle_id").size()
        report["average_claims_per_vehicle"] = float(claims_per_vehicle.mean())
        report["vehicles_with_multiple_claims_top20"] = claims_per_vehicle[claims_per_vehicle > 1].sort_values(ascending=False).head(20).to_dict()

    def group_rate(col: str) -> dict:
        if repeat is None or col not in df.columns:
            return {}
        tmp = pd.DataFrame({col: df[col], "repeat_claim_flag": repeat}).dropna()
        if tmp.empty:
            return {}
        return (
            tmp.groupby(col)["repeat_claim_flag"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .to_dict()
        )

    report["repeat_claim_rate_by_model_variant"] = group_rate("model_variant")
    report["repeat_claim_rate_by_subsystem"] = group_rate("subsystem")

    for col, metric_name in [
        ("model_variant", "claim_frequency_by_model_variant"),
        ("subsystem", "claim_frequency_by_subsystem"),
        ("failure_mode", "common_failure_modes"),
        ("state", "claim_frequency_by_state"),
        ("zone", "claim_frequency_by_zone"),
        ("dealer_name", "claims_by_dealer"),
    ]:
        if col in df.columns:
            report[metric_name] = df[col].fillna("Missing").value_counts().head(20).to_dict()

    if {"state", "total_claim_cost_inr"}.issubset(df.columns):
        state_cost = df.groupby("state", dropna=False)["total_claim_cost_inr"].mean().sort_values(ascending=False).head(20)
        report["average_cost_by_state_top20"] = state_cost.to_dict()

    if {"dealer_name", "service_capacity_score"}.issubset(df.columns):
        dealer_stats = df.groupby("dealer_name", dropna=False)["service_capacity_score"].mean()
        dealer_claims = df.groupby("dealer_name", dropna=False).size()
        joined = pd.concat([dealer_stats.rename("service_capacity_score"), dealer_claims.rename("claim_count")], axis=1)
        report["dealer_service_capacity_vs_claims_corr"] = float(joined["service_capacity_score"].corr(joined["claim_count"]))

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=joined, x="service_capacity_score", y="claim_count", alpha=0.7)
        plt.title("Dealer service capacity score vs claim count")
        plt.tight_layout()
        plt.savefig(out / "dealer_capacity_vs_claims.png", dpi=150)
        plt.close()

    return report
