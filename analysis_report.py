from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

SEPARATOR = "═" * 56


FILE_CANDIDATES = {
    "metrics": ["metrics.json", "model_metrics.json"],
    "claim_predictions": ["claim_predictions.csv", "predictions_claim_risk.csv"],
    "cost_forecast": ["cost_forecast.csv", "predictions_cost_forecast.csv"],
    "feature_importance": ["feature_importance.csv", "xgboost_feature_importance.csv"],
}


def _find_output_file(key: str) -> Path | None:
    for filename in FILE_CANDIDATES[key]:
        path = OUTPUT_DIR / filename
        if path.exists():
            return path
    return None


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _fmt_currency(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"₹{float(value):,.0f}"


def _resolve_accuracy(metrics: dict[str, Any]) -> float | None:
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        return float(accuracy)

    confusion_matrix = metrics.get("confusion_matrix")
    if confusion_matrix and len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2 and len(confusion_matrix[1]) == 2:
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]
        total = tn + fp + fn + tp
        if total > 0:
            return float((tn + tp) / total)
    return None


def _print_header(title: str) -> None:
    print(SEPARATOR)
    print(f" {title}")
    print(SEPARATOR)


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _print_claim_results(metrics_data: dict[str, Any], feature_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict[str, int]:
    print("\n📊 CLAIM PREDICTION RESULTS")
    print(SEPARATOR)

    claim_metrics = metrics_data.get("claim_prediction", {}) if isinstance(metrics_data, dict) else {}
    accuracy = _resolve_accuracy(claim_metrics)

    print("Model Performance:")
    print(f"  • Accuracy : {_fmt_pct(accuracy)}")
    print(f"  • AUC      : {_fmt_num(claim_metrics.get('auc', claim_metrics.get('auc_roc')))}")
    print(f"  • F1 Score : {_fmt_num(claim_metrics.get('f1'))}")
    print(f"  • Precision: {_fmt_num(claim_metrics.get('precision'))}")
    print(f"  • Recall   : {_fmt_num(claim_metrics.get('recall'))}")

    print("\nTop 10 Risk Factors:")
    if not feature_df.empty and {"feature", "importance"}.issubset(feature_df.columns):
        total_importance = feature_df["importance"].sum()
        top_features = feature_df.sort_values("importance", ascending=False).head(10).copy()
        if total_importance > 0:
            top_features["importance_pct"] = (top_features["importance"] / total_importance) * 100
        else:
            top_features["importance_pct"] = 0.0

        for idx, row in enumerate(top_features.itertuples(index=False), start=1):
            print(f"  {idx:>2}. {row.feature:<45} {row.importance_pct:>6.2f}%")
    else:
        print("  No feature importance data available.")

    print("\nTop 10 High-Risk Vehicles:")
    risk_counts = {"high": 0, "medium": 0, "low": 0}

    if not predictions_df.empty and "risk_score" in predictions_df.columns:
        predictions_df = predictions_df.copy()
        predictions_df["risk_score"] = pd.to_numeric(predictions_df["risk_score"], errors="coerce")

        top_risk = predictions_df.sort_values("risk_score", ascending=False).head(10)
        for idx, row in enumerate(top_risk.itertuples(index=False), start=1):
            vehicle = getattr(row, "vehicle_id", "N/A")
            model = getattr(row, "model_variant", "N/A")
            score = getattr(row, "risk_score", None)
            print(f"  {idx:>2}. {vehicle} ({model}) - Risk: {_fmt_pct(score)}")

        high_mask = predictions_df["risk_score"] > 0.70
        medium_mask = (predictions_df["risk_score"] >= 0.30) & (predictions_df["risk_score"] <= 0.70)
        low_mask = predictions_df["risk_score"] < 0.30

        risk_counts["high"] = int(high_mask.sum())
        risk_counts["medium"] = int(medium_mask.sum())
        risk_counts["low"] = int(low_mask.sum())
    else:
        print("  No claim predictions data available.")

    print("\nRisk Distribution:")
    print(f"  • High Risk   (>70%): {risk_counts['high']}")
    print(f"  • Medium Risk (30-70%): {risk_counts['medium']}")
    print(f"  • Low Risk    (<30%): {risk_counts['low']}")

    return risk_counts


def _print_cost_results(cost_df: pd.DataFrame) -> None:
    print("\n💰 COST FORECASTING RESULTS")
    print(SEPARATOR)

    if cost_df.empty:
        print("No cost forecast data available.")
        return

    required_cols = {"date", "forecasted_cost"}
    if not required_cols.issubset(cost_df.columns):
        print("Cost forecast file is missing required columns.")
        return

    monthly = cost_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly["forecasted_cost"] = pd.to_numeric(monthly["forecasted_cost"], errors="coerce")
    monthly = monthly.dropna(subset=["date", "forecasted_cost"])

    if monthly.empty:
        print("No valid forecast rows found.")
        return

    monthly_totals = (
        monthly.groupby("date", as_index=False)["forecasted_cost"]
        .sum()
        .sort_values("date")
        .head(12)
    )

    print("12-Month Warranty Cost Forecast:")
    print(f"  {'Month':<18} {'Predicted Cost':>18}")
    print(f"  {'-' * 18} {'-' * 18}")
    for row in monthly_totals.itertuples(index=False):
        print(f"  {row.date.strftime('%b %Y'):<18} {_fmt_currency(row.forecasted_cost):>18}")

    total_cost = float(monthly_totals["forecasted_cost"].sum())
    avg_cost = float(monthly_totals["forecasted_cost"].mean()) if not monthly_totals.empty else 0.0
    max_row = monthly_totals.loc[monthly_totals["forecasted_cost"].idxmax()]
    min_row = monthly_totals.loc[monthly_totals["forecasted_cost"].idxmin()]

    print("\nSummary:")
    print(f"  • Total Predicted Cost: {_fmt_currency(total_cost)}")
    print(
        f"  • Highest Cost Month : {max_row['date'].strftime('%B %Y')} ({_fmt_currency(max_row['forecasted_cost'])})"
    )
    print(
        f"  • Lowest Cost Month  : {min_row['date'].strftime('%B %Y')} ({_fmt_currency(min_row['forecasted_cost'])})"
    )
    print(f"  • Average Monthly Cost: {_fmt_currency(avg_cost)}")

    print("\nMonth-over-Month Trends:")
    deltas = monthly_totals.copy()
    deltas["delta"] = deltas["forecasted_cost"].diff()
    for row in deltas.itertuples(index=False):
        if pd.isna(row.delta):
            trend = "-"
            change = "N/A"
        elif row.delta > 0:
            trend = "📈"
            change = f"+{_fmt_currency(row.delta)}"
        elif row.delta < 0:
            trend = "📉"
            change = f"-{_fmt_currency(abs(row.delta))}"
        else:
            trend = "➡️"
            change = _fmt_currency(0)
        print(f"  • {row.date.strftime('%b %Y'):<12} {trend} {change}")


def _print_data_quality(predictions_df: pd.DataFrame, risk_counts: dict[str, int]) -> None:
    print("\n📈 DATA QUALITY & SUMMARY")
    print(SEPARATOR)

    total_vehicles = 0
    if not predictions_df.empty and "vehicle_id" in predictions_df.columns:
        total_vehicles = int(predictions_df["vehicle_id"].nunique())
    elif not predictions_df.empty:
        total_vehicles = int(len(predictions_df))

    total_records = int(len(predictions_df)) if not predictions_df.empty else 0

    print(f"  • Total Vehicles Analyzed : {total_vehicles}")
    print(f"  • Total Records Processed : {total_records}")
    print(f"  • Vehicles at High Risk   : {risk_counts['high']}")
    print(f"  • Vehicles at Medium Risk : {risk_counts['medium']}")
    print(f"  • Vehicles at Low Risk    : {risk_counts['low']}")


def main() -> None:
    _print_header("WARRANTY PREDICTION & FORECASTING - ANALYSIS REPORT")

    paths = {key: _find_output_file(key) for key in FILE_CANDIDATES}
    missing = [key for key, path in paths.items() if path is None]

    if missing:
        print("\n❌ Required output files were not found.")
        print("Please run 'python main.py' first")
        print("\nExpected files in outputs/:")
        for key, candidates in FILE_CANDIDATES.items():
            print(f"  - {candidates[0]}")
        return

    metrics_data = _safe_read_json(paths["metrics"]) if paths["metrics"] else {}
    predictions_df = _safe_read_csv(paths["claim_predictions"]) if paths["claim_predictions"] else pd.DataFrame()
    cost_df = _safe_read_csv(paths["cost_forecast"]) if paths["cost_forecast"] else pd.DataFrame()
    feature_df = _safe_read_csv(paths["feature_importance"]) if paths["feature_importance"] else pd.DataFrame()

    risk_counts = _print_claim_results(metrics_data, feature_df, predictions_df)
    _print_cost_results(cost_df)
    _print_data_quality(predictions_df, risk_counts)

    print(SEPARATOR)


if __name__ == "__main__":
    main()
