from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from flask import Flask, jsonify, render_template

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
CLAIM_FILE = OUTPUT_DIR / "predictions_claim_risk.csv"
COST_FILE = OUTPUT_DIR / "predictions_cost_forecast.csv"
FEATURE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            return list(csv.DictReader(fp))
    except OSError:
        return []


def _fmt_metric(value: Any, digits: int = 4, suffix: str = "") -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}{suffix}"


def _fmt_currency(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"₹{number:,.0f}"


def _risk_bucket(score: float) -> str:
    if score > 0.70:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def _build_risk_chart(risk_counts: dict[str, int]) -> str:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["High", "Medium", "Low"],
                values=[risk_counts["High"], risk_counts["Medium"], risk_counts["Low"]],
                marker=dict(colors=["#ef4444", "#f59e0b", "#22c55e"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title="Risk Distribution", paper_bgcolor="rgba(0,0,0,0)", height=350)
    return fig.to_html(full_html=False, include_plotlyjs=True)


def _build_cost_chart(cost_rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    monthly_totals: dict[str, float] = defaultdict(float)
    for row in cost_rows:
        month = row.get("date", "")[:7]
        forecasted = _safe_float(row.get("forecasted_cost"))
        if month and forecasted is not None:
            monthly_totals[month] += forecasted

    ordered = sorted(monthly_totals.items())[:12]
    dates = [x[0] for x in ordered]
    values = [x[1] for x in ordered]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            name="Forecasted Cost",
        )
    )
    fig.update_layout(
        title="12-Month Cost Forecast",
        xaxis_title="Month",
        yaxis_title="Cost (₹)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    table_data = [{"month": month, "cost": f"₹{value:,.0f}"} for month, value in ordered]
    return chart_html, table_data


def _build_feature_chart(top_features: list[dict[str, Any]]) -> str:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[f["importance"] for f in top_features],
                y=[f["feature"] for f in top_features],
                orientation="h",
                marker=dict(color="#a78bfa"),
            )
        ]
    )
    fig.update_layout(
        title="Top 15 Risk Factors",
        xaxis_title="Importance",
        yaxis_title="Feature",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=200, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_top_vehicles_chart(top_vehicles: list[dict[str, Any]]) -> str:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[v["label"] for v in top_vehicles],
                y=[v["risk_score"] * 100 for v in top_vehicles],
                marker=dict(color="#f97316"),
            )
        ]
    )
    fig.update_layout(
        title="Top 10 High-Risk Vehicles",
        xaxis_title="Vehicle",
        yaxis_title="Risk Score (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def load_dashboard_data() -> dict[str, Any]:
    metrics = _read_json(METRICS_FILE)
    claim_rows = _read_csv(CLAIM_FILE)
    cost_rows = _read_csv(COST_FILE)
    feature_rows = _read_csv(FEATURE_FILE)

    if not metrics:
        return {}

    claim_metrics = metrics.get("claim_prediction", {})
    cost_metrics = metrics.get("cost_forecasting", {})

    parsed_claim_rows = []
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    for row in claim_rows:
        score = _safe_float(row.get("risk_score"))
        if score is None:
            continue
        bucket = _risk_bucket(score)
        risk_counts[bucket] += 1
        parsed_claim_rows.append(
            {
                "vehicle_id": row.get("vehicle_id", "-"),
                "model_variant": row.get("model_variant", "-"),
                "risk_score": score,
                "risk_rank": row.get("risk_rank", "-"),
            }
        )

    parsed_claim_rows.sort(key=lambda x: x["risk_score"], reverse=True)
    top_vehicles = [
        {
            "vehicle_id": row["vehicle_id"],
            "model_variant": row["model_variant"],
            "risk_score": row["risk_score"],
            "risk_rank": row["risk_rank"],
            "label": f"{row['vehicle_id']} ({row['model_variant']})",
        }
        for row in parsed_claim_rows[:10]
    ]

    parsed_features = []
    for row in feature_rows:
        importance = _safe_float(row.get("importance"))
        if importance is None:
            continue
        parsed_features.append({"feature": row.get("feature", "-"), "importance": importance})
    parsed_features.sort(key=lambda x: x["importance"], reverse=True)
    top_features = parsed_features[:15]

    risk_chart = _build_risk_chart(risk_counts)
    cost_chart, monthly_forecast = _build_cost_chart(cost_rows)
    feature_chart = _build_feature_chart(top_features)
    top_vehicles_chart = _build_top_vehicles_chart(top_vehicles)

    total_cost = sum(_safe_float(row.get("forecasted_cost")) or 0.0 for row in cost_rows)
    avg_cost = total_cost / len(monthly_forecast) if monthly_forecast else 0.0

    return {
        "claim_metrics": {
            "auc": _fmt_metric(claim_metrics.get("auc_roc")),
            "precision": _fmt_metric(claim_metrics.get("precision")),
            "recall": _fmt_metric(claim_metrics.get("recall")),
            "f1": _fmt_metric(claim_metrics.get("f1")),
        },
        "cost_metrics": {
            "rmse": _fmt_currency(cost_metrics.get("rmse")),
            "mae": _fmt_currency(cost_metrics.get("mae")),
            "mape": _fmt_metric(cost_metrics.get("mape"), digits=2, suffix="%"),
        },
        "risk_counts": risk_counts,
        "top_vehicles": top_vehicles,
        "top_features": top_features,
        "monthly_forecast": monthly_forecast,
        "risk_chart": risk_chart,
        "cost_chart": cost_chart,
        "feature_chart": feature_chart,
        "top_vehicles_chart": top_vehicles_chart,
        "total_vehicles": len(parsed_claim_rows),
        "total_cost": _fmt_currency(total_cost),
        "avg_cost": _fmt_currency(avg_cost),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.route("/")
def index():
    dashboard_data = load_dashboard_data()
    if not dashboard_data:
        return render_template(
            "error.html",
            message="Required output files are missing or empty. Run 'python main.py' first.",
        )
    return render_template("index.html", data=dashboard_data)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(_read_json(METRICS_FILE))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
