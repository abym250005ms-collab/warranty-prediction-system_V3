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
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
CLAIM_FILE = OUTPUT_DIR / "predictions_claim_risk.csv"
COST_FILE = OUTPUT_DIR / "predictions_cost_forecast.csv"
FEATURE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"

REQUIRED_METRICS_SECTIONS = {"claim_prediction", "cost_forecasting"}
REQUIRED_CLAIM_COLUMNS = {"vehicle_id", "model_variant", "risk_score", "risk_rank"}
REQUIRED_COST_COLUMNS = {"date", "model_variant", "forecasted_cost"}
REQUIRED_FEATURE_COLUMNS = {"feature", "importance"}
YEAR_MONTH_FORMAT_LENGTH = 7

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _extract_year_month(date_text: str) -> str:
    if len(date_text) < YEAR_MONTH_FORMAT_LENGTH:
        return ""
    candidate = date_text[:YEAR_MONTH_FORMAT_LENGTH]
    if candidate[4] != "-":
        return ""
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            columns = [column for column in (reader.fieldnames or []) if column]
        return rows, columns
    except OSError:
        return [], []


def _validate_inputs(
    metrics: dict[str, Any],
    claim_columns: set[str],
    cost_columns: set[str],
    feature_columns: set[str],
) -> list[str]:
    errors: list[str] = []
    if not metrics:
        errors.append("Missing or unreadable outputs/model_metrics.json")
    elif not REQUIRED_METRICS_SECTIONS.issubset(metrics.keys()):
        errors.append("model_metrics.json is missing required sections")

    if not REQUIRED_CLAIM_COLUMNS.issubset(claim_columns):
        errors.append("predictions_claim_risk.csv is missing required columns")
    if not REQUIRED_COST_COLUMNS.issubset(cost_columns):
        errors.append("predictions_cost_forecast.csv is missing required columns")
    if not REQUIRED_FEATURE_COLUMNS.issubset(feature_columns):
        errors.append("xgboost_feature_importance.csv is missing required columns")

    return errors


def _build_risk_chart(risk_counts: dict[str, int]) -> str:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["High", "Medium", "Low"],
                values=[risk_counts["High"], risk_counts["Medium"], risk_counts["Low"]],
                marker=dict(colors=["#ef4444", "#f59e0b", "#22c55e"]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(title="Risk Distribution", paper_bgcolor="rgba(0,0,0,0)", height=360)
    return fig.to_html(full_html=False, include_plotlyjs=True)


def _build_cost_chart(cost_rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    monthly_totals: dict[str, float] = defaultdict(float)
    for row in cost_rows:
        date_text = _extract_year_month(row.get("date") or "")
        forecasted = _safe_float(row.get("forecasted_cost"))
        if date_text and forecasted is not None:
            monthly_totals[date_text] += forecasted

    ordered = sorted(monthly_totals.items())[:12]
    dates = [month for month, _ in ordered]
    values = [cost for _, cost in ordered]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.2)",
            name="Forecasted Cost",
            hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="12-Month Cost Forecast",
        xaxis_title="Month",
        yaxis_title="Cost (₹)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
        hovermode="x unified",
    )
    table_data = [{"month": month, "cost": f"₹{value:,.0f}"} for month, value in ordered]
    return fig.to_html(full_html=False, include_plotlyjs=False), table_data


def _build_feature_chart(top_features: list[dict[str, Any]]) -> str:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[feature["importance"] for feature in top_features],
                y=[feature["feature"] for feature in top_features],
                orientation="h",
                marker=dict(color="#8b5cf6"),
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Top 15 Risk Factors",
        xaxis_title="Importance",
        yaxis_title="Feature",
        paper_bgcolor="rgba(0,0,0,0)",
        height=460,
        margin=dict(l=210, r=20, t=65, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_top_vehicles_chart(top_vehicles: list[dict[str, Any]]) -> str:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[vehicle["label"] for vehicle in top_vehicles],
                y=[vehicle["risk_score"] * 100 for vehicle in top_vehicles],
                marker=dict(color="#f97316"),
                hovertemplate="%{x}<br>Risk: %{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Top 10 High-Risk Vehicles",
        xaxis_title="Vehicle",
        yaxis_title="Risk Score (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_dashboard_data() -> tuple[dict[str, Any], list[str], dict[str, bool]]:
    metrics = _read_json(METRICS_FILE)
    claim_rows, claim_columns = _read_csv(CLAIM_FILE)
    cost_rows, cost_columns = _read_csv(COST_FILE)
    feature_rows, feature_columns = _read_csv(FEATURE_FILE)

    file_health = {
        "metrics": METRICS_FILE.exists(),
        "claim": CLAIM_FILE.exists(),
        "cost": COST_FILE.exists(),
        "features": FEATURE_FILE.exists(),
    }

    validation_errors = _validate_inputs(
        metrics,
        set(claim_columns),
        set(cost_columns),
        set(feature_columns),
    )

    claim_metrics = metrics.get("claim_prediction", {}) if isinstance(metrics, dict) else {}
    cost_metrics = metrics.get("cost_forecasting", {}) if isinstance(metrics, dict) else {}

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

    parsed_claim_rows.sort(key=lambda row: row["risk_score"], reverse=True)
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
        parsed_features.append(
            {
                "feature": row.get("feature", "-"),
                "importance": importance,
            }
        )
    parsed_features.sort(key=lambda row: row["importance"], reverse=True)
    top_features = parsed_features[:15]

    risk_chart = _build_risk_chart(risk_counts)
    cost_chart, monthly_forecast = _build_cost_chart(cost_rows)
    feature_chart = _build_feature_chart(top_features)
    top_vehicles_chart = _build_top_vehicles_chart(top_vehicles)

    total_cost = sum(_safe_float(row.get("forecasted_cost")) or 0.0 for row in cost_rows)
    avg_cost = total_cost / len(monthly_forecast) if monthly_forecast else 0.0

    dashboard_data = {
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
    return dashboard_data, validation_errors, file_health


def _json_error(message: str, status: int = 500):
    return jsonify({"status": "error", "message": message}), status


@app.route("/")
def index():
    app.logger.info("Loading warranty analytics dashboard")
    dashboard_data, validation_errors, _ = _build_dashboard_data()

    if validation_errors:
        app.logger.warning("Dashboard validation errors: %s", validation_errors)
        return (
            render_template(
                "error.html",
                message="Data validation failed. Run 'python main.py' to regenerate outputs.",
                details=validation_errors,
            ),
            503,
        )

    return render_template("index.html", data=dashboard_data)


@app.route("/api/health")
def api_health():
    _, validation_errors, file_health = _build_dashboard_data()
    status = "ok" if not validation_errors else "degraded"
    code = 200 if status == "ok" else 503
    return jsonify({"status": status, "file_health": file_health, "errors": validation_errors}), code


@app.route("/api/metrics")
def api_metrics():
    metrics = _read_json(METRICS_FILE)
    if not metrics:
        return _json_error("Metrics file is missing or invalid", 404)
    return jsonify(metrics), 200


@app.route("/api/predictions")
def api_predictions():
    rows, columns = _read_csv(CLAIM_FILE)
    if not rows:
        return _json_error("Predictions file is missing or empty", 404)
    if not REQUIRED_CLAIM_COLUMNS.issubset(set(columns)):
        return _json_error("Predictions file schema is invalid", 422)
    return jsonify(rows), 200


@app.route("/api/forecast")
def api_forecast():
    rows, columns = _read_csv(COST_FILE)
    if not rows:
        return _json_error("Forecast file is missing or empty", 404)
    if not REQUIRED_COST_COLUMNS.issubset(set(columns)):
        return _json_error("Forecast file schema is invalid", 422)
    return jsonify(rows), 200


@app.route("/api/features")
def api_features():
    rows, columns = _read_csv(FEATURE_FILE)
    if not rows:
        return _json_error("Feature importance file is missing or empty", 404)
    if not REQUIRED_FEATURE_COLUMNS.issubset(set(columns)):
        return _json_error("Feature importance schema is invalid", 422)
    return jsonify(rows), 200


@app.route("/api/dashboard")
def api_dashboard():
    dashboard_data, validation_errors, file_health = _build_dashboard_data()
    if validation_errors:
        return (
            jsonify({"status": "error", "errors": validation_errors, "file_health": file_health}),
            503,
        )
    return jsonify({"status": "ok", "data": dashboard_data, "file_health": file_health}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
