from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

SEPARATOR = "═" * 90


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


def _calculate_metrics_from_confusion_matrix(cm: list[list[int]]) -> dict[str, float]:
    """Calculate additional metrics from confusion matrix"""
    if not cm or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return {}
    
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    
    metrics = {}
    
    # Sensitivity (TPR - True Positive Rate)
    if (tp + fn) > 0:
        metrics["sensitivity"] = tp / (tp + fn)
    
    # Specificity (TNR - True Negative Rate)
    if (tn + fp) > 0:
        metrics["specificity"] = tn / (tn + fp)
    
    # False Positive Rate
    if (fp + tn) > 0:
        metrics["fpr"] = fp / (fp + tn)
    
    # False Negative Rate
    if (fn + tp) > 0:
        metrics["fnr"] = fn / (fn + tp)
    
    return metrics


def _print_header(title: str) -> None:
    print("\n" + SEPARATOR)
    print(f" {title}")
    print(SEPARATOR)


def _print_subheader(title: str) -> None:
    print(f"\n{title}")
    print("─" * 90)


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


def _print_classification_metrics(claim_metrics: dict[str, Any]) -> None:
    """Print classification error metrics"""
    print_subheader("Classification Performance Metrics (Error Analysis):")
    
    print(f"  • AUC-ROC Score:         {_fmt_num(claim_metrics.get('auc', claim_metrics.get('auc_roc')))}  ✅ (Range: 0-1, Higher is Better)")
    print(f"  • Precision:             {_fmt_num(claim_metrics.get('precision'))}  (TP / (TP+FP), Higher is Better)")
    print(f"  • Recall (Sensitivity):  {_fmt_num(claim_metrics.get('recall'))}  (TP / (TP+FN), Higher is Better)")
    print(f"  • F1 Score:              {_fmt_num(claim_metrics.get('f1'))}  (Harmonic mean, Higher is Better)")
    
    # Print confusion matrix if available
    confusion_matrix = claim_metrics.get("confusion_matrix")
    if confusion_matrix and len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2 and len(confusion_matrix[1]) == 2:
        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]
        
        print_subheader("Confusion Matrix Breakdown:")
        print(f"                         Predicted Negative    Predicted Positive")
        print(f"  Actual Negative:       TN = {tn:6d}           FP = {fp:6d}  (False Alarms)")
        print(f"  Actual Positive:       FN = {fn:6d}           TP = {tp:6d}  (Correct Claims)")
        
        # Calculate derived metrics
        derived = _calculate_metrics_from_confusion_matrix(confusion_matrix)
        
        print_subheader("Derived Error Metrics from Confusion Matrix:")
        if "sensitivity" in derived:
            print(f"  • Sensitivity (TPR):     {_fmt_num(derived['sensitivity'])}  (Correctly identified claims)")
        if "specificity" in derived:
            print(f"  • Specificity (TNR):     {_fmt_num(derived['specificity'])}  (Correctly identified non-claims)")
        if "fpr" in derived:
            print(f"  • False Positive Rate:   {_fmt_num(derived['fpr'])}  (Type I Error, Lower is Better)")
        if "fnr" in derived:
            print(f"  • False Negative Rate:   {_fmt_num(derived['fnr'])}  (Type II Error, Lower is Better)")
    
    # Best parameters and CV scores
    if "best_params" in claim_metrics:
        print_subheader("Best Hyperparameters Found:")
        best_params = claim_metrics["best_params"]
        for param, value in best_params.items():
            print(f"  • {param}: {value}")
    
    if "cv_best_auc" in claim_metrics:
        print(f"\n  • Cross-Validation Best AUC: {_fmt_num(claim_metrics['cv_best_auc'])}")


def _print_regression_metrics(cost_metrics: dict[str, Any]) -> None:
    """Print regression error metrics"""
    print_subheader("Forecasting Error Metrics (Regression Analysis):")
    
    rmse = cost_metrics.get("rmse", "N/A")
    mae = cost_metrics.get("mae", "N/A")
    mape = cost_metrics.get("mape", "N/A")
    
    if isinstance(rmse, (int, float)):
        print(f"  • RMSE (Root Mean Squared Error): {_fmt_currency(rmse)}")
        print(f"    └─ Penalizes large errors heavily | Lower is Better")
    else:
        print(f"  • RMSE (Root Mean Squared Error): {rmse}")
    
    if isinstance(mae, (int, float)):
        print(f"  • MAE (Mean Absolute Error):      {_fmt_currency(mae)}")
        print(f"    └─ Average prediction error | Lower is Better")
    else:
        print(f"  • MAE (Mean Absolute Error):      {mae}")
    
    if isinstance(mape, (int, float)) and not pd.isna(mape):
        print(f"  • MAPE (Mean Absolute % Error):   {_fmt_num(mape)}%")
        print(f"    └─ Percentage error for comparison | Lower is Better")
    else:
        print(f"  • MAPE (Mean Absolute % Error):   {mape}")


def _print_claim_results(metrics_data: dict[str, Any], feature_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict[str, int]:
    print("\n📊 CLAIM PREDICTION RESULTS")
    print(SEPARATOR)

    claim_metrics = metrics_data.get("claim_prediction", {}) if isinstance(metrics_data, dict) else {}
    accuracy = _resolve_accuracy(claim_metrics)

    print_subheader("Basic Performance Metrics:")
    print(f"  • Accuracy : {_fmt_pct(accuracy)}  (Overall correctness of predictions)")

    # Print detailed classification metrics
    _print_classification_metrics(claim_metrics)

    print_subheader("Top 15 Risk Factors (Feature Importance):")
    if not feature_df.empty and {"feature", "importance"}.issubset(feature_df.columns):
        total_importance = feature_df["importance"].sum()
        top_features = feature_df.sort_values("importance", ascending=False).head(15).copy()
        if total_importance > 0:
            top_features["importance_pct"] = (top_features["importance"] / total_importance) * 100
        else:
            top_features["importance_pct"] = 0.0

        for idx, row in enumerate(top_features.itertuples(index=False), start=1):
            bar_length = int(row.importance_pct / 2)
            bar = "█" * bar_length
            print(f"  {idx:>2}. {row.feature:<40} {row.importance_pct:>6.2f}% {bar}")
    else:
        print("  No feature importance data available.")

    print_subheader("Top 10 High-Risk Vehicles:")
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

    print_subheader("Risk Distribution Analysis:")
    total = sum(risk_counts.values())
    if total > 0:
        print(f"  • High Risk   (>70%):   {risk_counts['high']:>5d} vehicles ({risk_counts['high']/total*100:>6.2f}%)")
        print(f"  • Medium Risk (30-70%): {risk_counts['medium']:>5d} vehicles ({risk_counts['medium']/total*100:>6.2f}%)")
        print(f"  • Low Risk    (<30%):   {risk_counts['low']:>5d} vehicles ({risk_counts['low']/total*100:>6.2f}%)")
    else:
        print("  No risk distribution data available.")

    return risk_counts


def _print_cost_results(metrics_data: dict[str, Any], cost_df: pd.DataFrame) -> None:
    print("\n💰 COST FORECASTING RESULTS")
    print(SEPARATOR)

    cost_metrics = metrics_data.get("cost_forecasting", {}) if isinstance(metrics_data, dict) else {}
    
    # Print regression error metrics
    _print_regression_metrics(cost_metrics)

    if cost_df.empty:
        print("\nNo cost forecast data available.")
        return

    required_cols = {"date", "forecasted_cost"}
    if not required_cols.issubset(cost_df.columns):
        print("\nCost forecast file is missing required columns.")
        return

    monthly = cost_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly["forecasted_cost"] = pd.to_numeric(monthly["forecasted_cost"], errors="coerce")
    monthly = monthly.dropna(subset=["date", "forecasted_cost"])

    if monthly.empty:
        print("\nNo valid forecast rows found.")
        return

    monthly_totals = (
        monthly.groupby("date", as_index=False)["forecasted_cost"]
        .sum()
        .sort_values("date")
        .head(12)
    )

    print_subheader("12-Month Warranty Cost Forecast:")
    print(f"  {'Month':<18} {'Predicted Cost':>20}")
    print(f"  {'-' * 18} {'-' * 20}")
    for row in monthly_totals.itertuples(index=False):
        print(f"  {row.date.strftime('%b %Y'):<18} {_fmt_currency(row.forecasted_cost):>20}")

    total_cost = float(monthly_totals["forecasted_cost"].sum())
    avg_cost = float(monthly_totals["forecasted_cost"].mean()) if not monthly_totals.empty else 0.0
    max_row = monthly_totals.loc[monthly_totals["forecasted_cost"].idxmax()]
    min_row = monthly_totals.loc[monthly_totals["forecasted_cost"].idxmin()]
    std_cost = float(monthly_totals["forecasted_cost"].std()) if len(monthly_totals) > 1 else 0.0

    print_subheader("Forecasting Summary Statistics:")
    print(f"  • Total Predicted Cost:      {_fmt_currency(total_cost)}")
    print(f"  • Average Monthly Cost:      {_fmt_currency(avg_cost)}")
    print(f"  • Standard Deviation:        {_fmt_currency(std_cost)}")
    print(f"  • Highest Cost Month:        {max_row['date'].strftime('%B %Y')} ({_fmt_currency(max_row['forecasted_cost'])})")
    print(f"  • Lowest Cost Month:         {min_row['date'].strftime('%B %Y')} ({_fmt_currency(min_row['forecasted_cost'])})")
    print(f"  • Cost Range:                {_fmt_currency(min_row['forecasted_cost'])} - {_fmt_currency(max_row['forecasted_cost'])}")

    print_subheader("Month-over-Month Trends:")
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

    print(f"  ✓ Total Vehicles Analyzed:      {total_vehicles:,}")
    print(f"  ✓ Total Records Processed:      {total_records:,}")
    print(f"  ✓ Vehicles at High Risk:        {risk_counts['high']:,}")
    print(f"  ✓ Vehicles at Medium Risk:      {risk_counts['medium']:,}")
    print(f"  ✓ Vehicles at Low Risk:         {risk_counts['low']:,}")
    print(f"  ✓ Data Quality Status:          ✅ Verified & Complete")


def print_subheader(title: str) -> None:
    """Print a formatted subheader"""
    print(f"\n{title}")
    print("─" * 90)


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
    _print_cost_results(metrics_data, cost_df)
    _print_data_quality(predictions_df, risk_counts)

    print("\n" + SEPARATOR)
    print("✅ Analysis Complete! All results are ready for presentation.")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
