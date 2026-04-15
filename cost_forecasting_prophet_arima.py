from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

import config
from model_evaluation import backtest_last_window, evaluate_forecast


@dataclass
class CostForecastResult:
    forecasts: pd.DataFrame
    metrics: Dict



def _monthly_cost_series(df: pd.DataFrame, model_variant: str) -> pd.Series:
    subset = df[df["model_variant"] == model_variant].copy()
    monthly = (
        subset.groupby(pd.Grouper(key="claim_date", freq="MS"))["total_claim_cost_inr"]
        .sum()
        .sort_index()
    )
    if monthly.empty:
        return monthly
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    return monthly.reindex(full_index, fill_value=0.0)


def _check_stationarity(series: pd.Series) -> Dict[str, float]:
    clean = series.dropna()
    if len(clean) < 12:
        return {"adf_stat": float("nan"), "adf_pvalue": float("nan")}
    stat, pvalue, *_ = adfuller(clean)
    return {"adf_stat": float(stat), "adf_pvalue": float(pvalue)}


def _fit_arima_forecast(series: pd.Series, periods: int) -> Dict[str, np.ndarray]:
    if len(series) < 4:
        base = np.repeat(float(series.iloc[-1]) if len(series) else 0.0, periods)
        return {
            "forecast": base,
            "lower_80": base,
            "upper_80": base,
            "lower_95": base,
            "upper_95": base,
        }

    arima_model = auto_arima(
        series,
        seasonal=len(series) >= 24,
        m=12,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    forecast = arima_model.predict(n_periods=periods)
    ci_80 = arima_model.predict(n_periods=periods, return_conf_int=True, alpha=0.2)[1]
    ci_95 = arima_model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)[1]

    return {
        "forecast": np.asarray(forecast),
        "lower_80": ci_80[:, 0],
        "upper_80": ci_80[:, 1],
        "lower_95": ci_95[:, 0],
        "upper_95": ci_95[:, 1],
    }


def _fit_prophet_forecast(series: pd.Series, periods: int) -> Dict[str, np.ndarray]:
    if len(series) < 4:
        base = np.repeat(float(series.iloc[-1]) if len(series) else 0.0, periods)
        return {
            "forecast": base,
            "lower_80": base,
            "upper_80": base,
            "lower_95": base,
            "upper_95": base,
        }

    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    model.fit(frame)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future).tail(periods)

    mean = forecast["yhat"].to_numpy()
    lower_95 = forecast["yhat_lower"].to_numpy()
    upper_95 = forecast["yhat_upper"].to_numpy()

    std = (upper_95 - lower_95) / (2 * 1.96)
    z80 = 1.2816
    lower_80 = mean - z80 * std
    upper_80 = mean + z80 * std

    return {
        "forecast": mean,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


def _ensemble_forecasts(prophet_fc: Dict[str, np.ndarray], arima_fc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    wp = config.ENSEMBLE_WEIGHTS["prophet"]
    wa = config.ENSEMBLE_WEIGHTS["arima"]
    out = {}
    for key in ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]:
        out[key] = wp * prophet_fc[key] + wa * arima_fc[key]
    return out


def _backtest_variant(series: pd.Series, horizon: int = 3) -> Dict[str, float]:
    if len(series) < horizon + 6:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    prophet_fc = _fit_prophet_forecast(train, horizon)
    arima_fc = _fit_arima_forecast(train, horizon)
    ensemble = _ensemble_forecasts(prophet_fc, arima_fc)

    return evaluate_forecast(test.values, ensemble["forecast"])


def forecast_costs(df: pd.DataFrame, periods: int = config.COST_FORECAST_MONTHS) -> CostForecastResult:
    work_df = df.copy()
    work_df = work_df.dropna(subset=["model_variant", "claim_date", "total_claim_cost_inr"])
    work_df["claim_date"] = pd.to_datetime(work_df["claim_date"], errors="coerce")
    work_df = work_df.dropna(subset=["claim_date"])

    outputs: List[pd.DataFrame] = []
    metrics: Dict[str, Dict] = {}

    # Get today's date dynamically
    today = datetime.now()

    for variant in sorted(work_df["model_variant"].dropna().unique()):
        series = _monthly_cost_series(work_df, variant)
        if series.empty:
            continue

        stationarity = _check_stationarity(series)
        prophet_fc = _fit_prophet_forecast(series, periods)
        arima_fc = _fit_arima_forecast(series, periods)
        ensemble = _ensemble_forecasts(prophet_fc, arima_fc)

        # Generate future dates starting from today
        future_dates = pd.date_range(start=today, periods=periods, freq="MS")
        
        variant_forecast = pd.DataFrame(
            {
                "date": future_dates,
                "model_variant": variant,
                "forecasted_cost": ensemble["forecast"],
                "lower_ci_80": ensemble["lower_80"],
                "upper_ci_80": ensemble["upper_80"],
                "lower_ci_95": ensemble["lower_95"],
                "upper_ci_95": ensemble["upper_95"],
            }
        )
        outputs.append(variant_forecast)

        metrics[variant] = {
            "stationarity": stationarity,
            "backtest": _backtest_variant(series, horizon=min(3, max(1, len(series) // 4))),
        }

    if outputs:
        forecast_df = pd.concat(outputs, ignore_index=True)
    else:
        forecast_df = pd.DataFrame(
            columns=[
                "date",
                "model_variant",
                "forecasted_cost",
                "lower_ci_80",
                "upper_ci_80",
                "lower_ci_95",
                "upper_ci_95",
            ]
        )

    return CostForecastResult(forecasts=forecast_df, metrics=metrics)
