import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_config() -> dict:
    return {
        "data_path": os.getenv("DATA_PATH", "data/energy_consumption.csv"),
        "model_dir": os.getenv("MODEL_DIR", "models"),
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "test_split": float(os.getenv("TEST_SPLIT", "0.2")),
        "random_seed": int(os.getenv("RANDOM_SEED", "42")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }
