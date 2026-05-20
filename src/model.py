"""Model training, evaluation, and persistence."""

from __future__ import annotations
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger("predictor")


def train(df: pd.DataFrame, seed: int = 42) -> GradientBoostingRegressor:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def evaluate(model, df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y_true = df[TARGET_COLUMN]
    y_pred = model.predict(X)

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "samples": len(y_true),
    }


def cross_validate(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[val_idx])

        results.append({
            "fold": fold,
            "mae": round(float(mean_absolute_error(y.iloc[val_idx], y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))), 4),
            "r2": round(float(r2_score(y.iloc[val_idx], y_pred)), 4),
        })
    return results


def feature_importance(model, top_n: int = 10) -> list[dict]:
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs[:top_n]]


def save_model(model, metrics: dict, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return model_path


def load_model(model_dir: str | Path) -> GradientBoostingRegressor:
    return joblib.load(Path(model_dir) / "model.joblib")
