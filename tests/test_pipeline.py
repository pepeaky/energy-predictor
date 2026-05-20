import pytest
import numpy as np
import pandas as pd
from src.data import generate_synthetic_data
from src.features import add_lag_features, add_rolling_features, add_cyclical_features, prepare_features, FEATURE_COLUMNS
from src.model import train, evaluate, feature_importance


@pytest.fixture
def sample_data():
    return generate_synthetic_data(n_days=60, seed=42)


@pytest.fixture
def prepared_data(sample_data):
    return prepare_features(sample_data)


class TestDataGeneration:
    def test_shape(self, sample_data):
        assert len(sample_data) == 60 * 24
        assert "consumption_kwh" in sample_data.columns

    def test_no_negative_consumption(self, sample_data):
        assert (sample_data["consumption_kwh"] >= 0).all()

    def test_has_required_columns(self, sample_data):
        for col in ["timestamp", "consumption_kwh", "temperature_c", "hour", "day_of_week", "is_weekend"]:
            assert col in sample_data.columns


class TestFeatures:
    def test_lag_features(self, sample_data):
        df = add_lag_features(sample_data, lags=[1, 24])
        assert "lag_1h" in df.columns
        assert "lag_24h" in df.columns
        assert pd.isna(df["lag_24h"].iloc[0])

    def test_rolling_features(self, sample_data):
        df = add_rolling_features(sample_data, windows=[24])
        assert "rolling_mean_24h" in df.columns
        assert "rolling_std_24h" in df.columns

    def test_cyclical_features(self, sample_data):
        df = add_cyclical_features(sample_data)
        assert "hour_sin" in df.columns
        assert df["hour_sin"].between(-1, 1).all()

    def test_prepare_drops_na(self, sample_data):
        df = prepare_features(sample_data)
        assert df.isna().sum().sum() == 0

    def test_all_feature_columns_present(self, prepared_data):
        for col in FEATURE_COLUMNS:
            assert col in prepared_data.columns


class TestModel:
    def test_train_returns_model(self, prepared_data):
        model = train(prepared_data)
        assert hasattr(model, "predict")

    def test_evaluate_metrics(self, prepared_data):
        split = int(len(prepared_data) * 0.8)
        model = train(prepared_data.iloc[:split])
        metrics = evaluate(model, prepared_data.iloc[split:])
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.5

    def test_feature_importance_sorted(self, prepared_data):
        model = train(prepared_data)
        fi = feature_importance(model)
        importances = [f["importance"] for f in fi]
        assert importances == sorted(importances, reverse=True)

    def test_predictions_reasonable(self, prepared_data):
        model = train(prepared_data)
        preds = model.predict(prepared_data[FEATURE_COLUMNS].tail(24))
        assert all(p > 0 for p in preds)
        assert all(p < 2000 for p in preds)
