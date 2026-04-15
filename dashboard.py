from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

# Create Flask app
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


def load_data():
    """Load all output files"""
    try:
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    try:
        predictions_path = OUTPUT_DIR / "claim_predictions.csv"
        predictions_df = pd.read_csv(predictions_path)
    except Exception:
        predictions_df = pd.DataFrame()

    try:
        cost_path = OUTPUT_DIR / "cost_forecast.csv"
        cost_df = pd.read_csv(cost_path)
        cost_df["date"] = pd.to_datetime(cost_df["date"])
    except Exception:
        cost_df = pd.DataFrame()

    try:
        feature_path = OUTPUT_DIR / "feature_importance.csv"
        feature_df = pd.read_csv(feature_path)
    except Exception:
        feature_df = pd.DataFrame()

    return metrics, predictions_df, cost_df, feature_df


def create_risk_distribution_chart(predictions_df):
    """Create risk distribution pie chart"""
    if predictions_df.empty or "risk_score" not in predictions_df.columns:
        return None
    
    predictions_df = predictions_df.copy()
    predictions_df["risk_score"] = pd.to_numeric(predictions_df["risk_score"], errors="coerce")
    predictions_df = predictions_df.dropna(subset=["risk_score"])
    
    high_risk = len(predictions_df[predictions_df["risk_score"] > 0.70])
    medium_risk = len(predictions_df[(predictions_df["risk_score"] >= 0.30) & (predictions_df["risk_score"] <= 0.70)])
    low_risk = len(predictions_df[predictions_df["risk_score"] < 0.30])
    
    fig = go.Figure(data=[go.Pie(
        labels=["High Risk (>70%)", "Medium Risk (30-70%)", "Low Risk (<30%)"],
        values=[high_risk, medium_risk, low_risk],
        marker=dict(colors=["#e74c3c", "#f39c12", "#2ecc71"]),
        textposition="inside",
        textinfo="label+percent"
    )])
    fig.update_layout(
        title="Risk Distribution",
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_cost_forecast_chart(cost_df):
    """Create cost forecast line chart"""
    if cost_df.empty:
        return None
    
    monthly_forecast = cost_df.groupby("date")["forecasted_cost"].sum().reset_index()
    monthly_forecast = monthly_forecast.sort_values("date").head(12)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_forecast["date"],
        y=monthly_forecast["forecasted_cost"],
        mode="lines+markers",
        name="Forecasted Cost",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.2)"
    ))
    fig.update_layout(
        title="12-Month Warranty Cost Forecast",
        xaxis_title="Month",
        yaxis_title="Cost (₹)",
        hovermode="x unified",
        height=400,
        template="plotly_white"
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_feature_importance_chart(feature_df):
    """Create feature importance bar chart"""
    if feature_df.empty:
        return None
    
    top_features = feature_df.sort_values("importance", ascending=False).head(15)
    
    fig = go.Figure(data=[go.Bar(
        y=top_features["feature"],
        x=top_features["importance"],
        orientation="h",
        marker=dict(color=top_features["importance"], colorscale="Viridis")
    )])
    fig.update_layout(
        title="Top 15 Risk Factors",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        template="plotly_white",
        showlegend=False
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_top_risk_vehicles_chart(predictions_df):
    """Create top risk vehicles bar chart"""
    if predictions_df.empty or "risk_score" not in predictions_df.columns:
        return None
    
    predictions_df = predictions_df.copy()
    predictions_df["risk_score"] = pd.to_numeric(predictions_df["risk_score"], errors="coerce")
    top_risk = predictions_df.sort_values("risk_score", ascending=False).head(10)
    
    top_risk["vehicle_label"] = top_risk["vehicle_id"] + " (" + top_risk["model_variant"] + ")"
    
    fig = go.Figure(data=[go.Bar(
        x=top_risk["vehicle_label"],
        y=top_risk["risk_score"] * 100,
        marker=dict(color=top_risk["risk_score"], colorscale="Reds")
    )])
    fig.update_layout(
        title="Top 10 High-Risk Vehicles",
        xaxis_title="Vehicle",
        yaxis_title="Risk Score (%)",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/")
def index():
    """Home page"""
    metrics, predictions_df, cost_df, feature_df = load_data()
    
    if not metrics:
        return render_template("error.html", message="No data found. Please run 'python main.py' first.")
    
    # Extract claim metrics
    claim_metrics = metrics.get("claim_prediction", {})
    auc = claim_metrics.get("auc_roc", claim_metrics.get("auc", "N/A"))
    precision = claim_metrics.get("precision", "N/A")
    recall = claim_metrics.get("recall", "N/A")
    f1 = claim_metrics.get("f1", "N/A")
    
    # Extract cost metrics
    cost_metrics = metrics.get("cost_forecasting", {})
    rmse = cost_metrics.get("rmse", "N/A")
    mae = cost_metrics.get("mae", "N/A")
    mape = cost_metrics.get("mape", "N/A")
    
    # Calculate risk counts
    if not predictions_df.empty and "risk_score" in predictions_df.columns:
        predictions_df = predictions_df.copy()
        predictions_df["risk_score"] = pd.to_numeric(predictions_df["risk_score"], errors="coerce")
        high_risk = len(predictions_df[predictions_df["risk_score"] > 0.70])
        medium_risk = len(predictions_df[(predictions_df["risk_score"] >= 0.30) & (predictions_df["risk_score"] <= 0.70)])
        low_risk = len(predictions_df[predictions_df["risk_score"] < 0.30])
        total_vehicles = predictions_df["vehicle_id"].nunique()
    else:
        high_risk = medium_risk = low_risk = total_vehicles = 0
    
    # Calculate cost metrics
    if not cost_df.empty:
        monthly_forecast = cost_df.groupby("date")["forecasted_cost"].sum().reset_index()
        total_cost = monthly_forecast["forecasted_cost"].sum()
        avg_cost = monthly_forecast["forecasted_cost"].mean()
    else:
        total_cost = avg_cost = 0
    
    # Create charts
    risk_chart = create_risk_distribution_chart(predictions_df)
    cost_chart = create_cost_forecast_chart(cost_df)
    feature_chart = create_feature_importance_chart(feature_df)
    top_vehicles_chart = create_top_risk_vehicles_chart(predictions_df)
    
    return render_template(
        "index.html",
        auc=f"{float(auc):.4f}" if isinstance(auc, (int, float)) else auc,
        precision=f"{float(precision):.4f}" if isinstance(precision, (int, float)) else precision,
        recall=f"{float(recall):.4f}" if isinstance(recall, (int, float)) else recall,
        f1=f"{float(f1):.4f}" if isinstance(f1, (int, float)) else f1,
        rmse=f"₹{float(rmse):,.0f}" if isinstance(rmse, (int, float)) else rmse,
        mae=f"₹{float(mae):,.0f}" if isinstance(mae, (int, float)) else mae,
        mape=f"{float(mape):.2f}%" if isinstance(mape, (int, float)) else mape,
        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,
        total_vehicles=total_vehicles,
        total_cost=f"₹{total_cost:,.0f}",
        avg_cost=f"₹{avg_cost:,.0f}",
        risk_chart=risk_chart,
        cost_chart=cost_chart,
        feature_chart=feature_chart,
        top_vehicles_chart=top_vehicles_chart,
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.route("/api/metrics")
def api_metrics():
    """API endpoint for metrics"""
    metrics, _, _, _ = load_data()
    return jsonify(metrics)


@app.route("/api/predictions")
def api_predictions():
    """API endpoint for predictions"""
    _, predictions_df, _, _ = load_data()
    return jsonify(predictions_df.to_dict(orient="records"))


@app.route("/api/forecast")
def api_forecast():
    """API endpoint for forecast"""
    _, _, cost_df, _ = load_data()
    cost_df_copy = cost_df.copy()
    cost_df_copy["date"] = cost_df_copy["date"].astype(str)
    return jsonify(cost_df_copy.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
