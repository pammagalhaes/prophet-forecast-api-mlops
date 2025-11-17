from prophet import Prophet
from pathlib import Path
import joblib

def save_model(model: Prophet, store_id: int, model_dir: Path):
    p = Path(model_dir) / f"prophet_store_{store_id}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p

def load_model(store_id: int, model_dir: Path):
    p = Path(model_dir) / f"prophet_store_{store_id}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Model for store {store_id} not found at {p}")
    return joblib.load(p)
