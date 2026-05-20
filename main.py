"""CLI for the Energy Consumption Predictor."""

import argparse
import json
import logging

from src.config import get_config
from src.data import generate_synthetic_data, load_data, save_data
from src.features import prepare_features
from src.model import train, evaluate, cross_validate, feature_importance, save_model, load_model


def cmd_generate(args):
    cfg = get_config()
    df = generate_synthetic_data(n_days=args.days, seed=cfg["random_seed"])
    save_data(df, cfg["data_path"])
    print(f"Generated {len(df)} records → {cfg['data_path']}")


def cmd_train(args):
    cfg = get_config()
    df = load_data(cfg["data_path"])
    df = prepare_features(df)

    split_idx = int(len(df) * (1 - cfg["test_split"]))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = train(train_df, seed=cfg["random_seed"])
    metrics = evaluate(model, test_df)
    fi = feature_importance(model)

    save_model(model, {**metrics, "feature_importance": fi}, cfg["model_dir"])
    print(json.dumps(metrics, indent=2))
    print("\nTop features:")
    for f in fi[:5]:
        print(f"  {f['feature']:.<30} {f['importance']:.4f}")


def cmd_cv(args):
    cfg = get_config()
    df = load_data(cfg["data_path"])
    df = prepare_features(df)
    results = cross_validate(df, n_splits=args.folds, seed=cfg["random_seed"])
    for r in results:
        print(f"  Fold {r['fold']}: MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  R²={r['r2']:.4f}")


def cmd_predict(args):
    cfg = get_config()
    model = load_model(cfg["model_dir"])
    df = load_data(cfg["data_path"])
    df = prepare_features(df)
    last_n = df.tail(args.hours)
    preds = model.predict(last_n[["temperature_c", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos", "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h", "rolling_std_168h"]])
    for ts, actual, pred in zip(last_n["timestamp"], last_n["consumption_kwh"], preds):
        print(f"  {ts}  actual={actual:.1f}  predicted={pred:.1f}  error={abs(actual-pred):.1f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Energy Consumption Predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic energy data")
    p_gen.add_argument("--days", type=int, default=730)

    sub.add_parser("train", help="Train and evaluate model")

    p_cv = sub.add_parser("cv", help="Time-series cross-validation")
    p_cv.add_argument("--folds", type=int, default=5)

    p_pred = sub.add_parser("predict", help="Show predictions for last N hours")
    p_pred.add_argument("--hours", type=int, default=24)

    args = parser.parse_args()
    {"generate": cmd_generate, "train": cmd_train, "cv": cmd_cv, "predict": cmd_predict}[args.command](args)


if __name__ == "__main__":
    main()
