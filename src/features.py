"""Feature engineering for time-series energy prediction."""

from __future__ import annotations
import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    lags = lags or [1, 24, 168]
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df["consumption_kwh"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [24, 168]
    df = df.copy()
    for w in windows:
        df[f"rolling_mean_{w}h"] = df["consumption_kwh"].rolling(w).mean()
        df[f"rolling_std_{w}h"] = df["consumption_kwh"].rolling(w).std()
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "temperature_c", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h", "rolling_std_168h",
]

TARGET_COLUMN = "consumption_kwh"
