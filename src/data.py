"""Data generation and loading for energy consumption time-series."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")

    hour = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month

    base_load = 500
    hourly_pattern = 200 * np.sin(2 * np.pi * (hour - 6) / 24)
    seasonal_pattern = 150 * np.cos(2 * np.pi * (month - 1) / 12)
    weekend_effect = np.where(day_of_week >= 5, -100, 0)
    temperature = 15 + 12 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 3, len(dates))
    temp_effect = 5 * (temperature - 20) ** 2 / 100
    noise = rng.normal(0, 30, len(dates))

    consumption = base_load + hourly_pattern + seasonal_pattern + weekend_effect + temp_effect + noise
    consumption = np.maximum(consumption, 50)

    df = pd.DataFrame({
        "timestamp": dates,
        "consumption_kwh": np.round(consumption, 2),
        "temperature_c": np.round(temperature, 1),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": (day_of_week >= 5).astype(int),
    })
    return df


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def save_data(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
