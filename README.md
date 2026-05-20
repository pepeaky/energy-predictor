# Energy Consumption Predictor

A time-series regression model that predicts hourly energy consumption using lag features, rolling statistics, cyclical encoding, and Gradient Boosting with proper time-series cross-validation.

---

## Architecture

```
  Synthetic Data Generation
  (2 years hourly: seasonal + daily patterns + temperature + noise)
         │
         ▼
┌─────────────────────┐
│  Feature Engineering  │
│  • Lag features (1h, 24h, 168h)
│  • Rolling mean/std (24h, 168h)
│  • Cyclical encoding (hour, month, dow)
│  • Temperature
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌────────┐  ┌──────────┐
│ Train   │  │   Test    │  Time-based split (no shuffle)
│  80%    │  │   20%     │
└────┬───┘  └─────┬────┘
     │            │
     ▼            ▼
┌────────────────────────┐
│ GradientBoostingRegressor │
│ • 200 estimators          │
│ • TimeSeriesSplit CV      │
│ • Feature importance      │
└───────────┬──────────────┘
            │
       ┌────┴────┐
       ▼         ▼
   model.joblib  metrics.json
```

## Features

- **Synthetic data generator** — realistic energy patterns with seasonality, day/night cycles, weekend effects, temperature correlation
- **Time-series aware** — no data leakage; train/test split is chronological, cross-validation uses `TimeSeriesSplit`
- **Rich feature set** — 15 engineered features from 7 raw columns
- **Model persistence** — joblib serialization + JSON metrics for versioning

## Quick Start

```bash
git clone <repo-url> && cd 10-energy-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py generate --days 730    # 2 years of hourly data
python main.py train                  # train + evaluate
python main.py cv --folds 5           # time-series cross-validation
python main.py predict --hours 24     # show last 24h predictions
```

## Testing

```bash
pytest -v
```

**13 tests** — data generation, feature engineering, model training, evaluation metrics, and prediction bounds.

## Project Structure

```
├── main.py              # CLI: generate, train, cv, predict
├── src/
│   ├── config.py        # .env loader
│   ├── data.py          # Synthetic data generator + I/O
│   ├── features.py      # Lag, rolling, cyclical feature engineering
│   └── model.py         # GBR train, evaluate, CV, feature importance
└── tests/
    └── test_pipeline.py # 13 tests — full ML pipeline
```
