from pathlib import Path
import os

# Base project path (root)
BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# Data paths
DATA_DIR = BASE_DIR / "data" / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
STORE_CSV = DATA_DIR / "store.csv"

# Model path
if os.getenv("RENDER") == "1":
    # Deploy on Render
    MODEL_DIR = Path("/app/models")
else:
    # Local
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_DIR = BASE_DIR / "models"

# split date for train/validation (example; adjust if needed)
TRAIN_TEST_SPLIT_DATE = "2015-06-01"
