from __future__ import annotations

from pathlib import Path


def _top_items(mapping: dict, n: int = 5) -> list[tuple]:
    if not mapping:
        return []
    return sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:n]


def create_summary_report(eda_results: dict, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rel = eda_results.get("relationships", {})
    risk = eda_results.get("risk_analysis", {})
    temporal = eda_results.get("temporal", {})
    overview = eda_results.get("data_overview", {})

    insights = []
    insights.append(f"Dataset size: {overview.get('shape', {}).get('rows', 0)} rows x {overview.get('shape', {}).get('columns', 0)} columns.")
    insights.append(f"Duplicate rows: {overview.get('duplicate_rows', 0)}.")

    top_cost_corr = _top_items(rel.get("correlation_with_total_claim_cost_top10", {}), 5)
    if top_cost_corr:
        insights.append("Top claim cost correlated numeric features: " + ", ".join([f"{k} ({v:.3f})" for k, v in top_cost_corr]))

    top_failure_modes = _top_items(risk.get("common_failure_modes", {}), 5)
    if top_failure_modes:
        insights.append("Most common failure modes: " + ", ".join([f"{k}: {v}" for k, v in top_failure_modes]))

    top_subsystems = _top_items(risk.get("claim_frequency_by_subsystem", {}), 5)
    if top_subsystems:
        insights.append("Most affected subsystems: " + ", ".join([f"{k}: {v}" for k, v in top_subsystems]))

    top_states = _top_items(risk.get("claim_frequency_by_state", {}), 5)
    if top_states:
        insights.append("Highest claim states: " + ", ".join([f"{k}: {v}" for k, v in top_states]))

    season_cost = _top_items(temporal.get("season_avg_cost", {}), 4)
    if season_cost:
        insights.append("Average claim cost by season (descending): " + ", ".join([f"{k}: {v:.2f}" for k, v in season_cost]))

    recommendations = [
        "Model features to prioritize: months_in_service, odometer_at_failure_km, model_variant, subsystem, failure_mode, state/zone, ambient_temp_celsius, use_case, repeat_claim_flag.",
        "Data quality checks: missing values in date/categorical fields and duplicate handling before modeling.",
        "Feature engineering opportunities: warranty coverage gap, cost-per-km, vehicle age buckets, season-temperature interaction, component criticality signals.",
        "Investigate anomalies: extremely high total_claim_cost_inr, rare failure modes with high severity, dealers with high claim concentrations.",
    ]

    txt_path = out / "eda_insights.txt"
    txt_path.write_text("\n".join(insights + ["", "Recommendations:"] + recommendations), encoding="utf-8")

    html_content = "<html><head><title>EDA Report</title></head><body>"
    html_content += "<h1>Warranty Dataset EDA Summary</h1><h2>Key Insights</h2><ul>"
    for line in insights:
        html_content += f"<li>{line}</li>"
    html_content += "</ul><h2>Recommendations for Modeling</h2><ul>"
    for rec in recommendations:
        html_content += f"<li>{rec}</li>"
    html_content += "</ul></body></html>"
    (out / "eda_report.html").write_text(html_content, encoding="utf-8")

    return {"insights": insights, "recommendations": recommendations}

