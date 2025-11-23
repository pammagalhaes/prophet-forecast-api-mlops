import pandas as pd
from tqdm import tqdm
from ..data.load_data import load_raw
from ..data.preprocess import basic_clean, to_prophet_df
from .model_utils import save_model
from ..config import TRAIN_TEST_SPLIT_DATE, MODEL_DIR
from src.modeling.validation import validate_prophet
from prophet import Prophet
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error

def train_store(df, store_id, split_date=TRAIN_TEST_SPLIT_DATE, regressors=None):
    """
    Train a Prophet model per store and log metrics and artifacts to MLflow.
    """
    df_store = df[df['Store'] == store_id].copy()
    df_store = basic_clean(df_store)
    df_prophet = to_prophet_df(df_store)

    train_df = df_prophet[df_prophet['ds'] < pd.to_datetime(split_date)]

    with mlflow.start_run(run_name=f"store_{store_id}"):

        # ----- Model -----
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        regressors = regressors or []
        for r in regressors:
            m.add_regressor(r)

        m.fit(train_df)

        # ----- Validation (TimeSeriesSplit) -----
        cv_metrics = validate_prophet(df_prophet, n_splits=3, regressors=regressors)

        mlflow.log_metric("mae_cv", cv_metrics["mae_mean"])
        mlflow.log_metric("rmse_cv", cv_metrics["rmse_mean"])

        # ----- In-sample forecast -----
        future = m.make_future_dataframe(periods=0, freq='D')
        forecast = m.predict(future)

        # Simple in-sample MAE
        mae_insample = mean_absolute_error(df_prophet['y'], forecast['yhat'])
        mlflow.log_metric("mae_insample", mae_insample)

        # ----- Parameters -----
        mlflow.log_param("store_id", store_id)
        mlflow.log_param("regressors", regressors)

        # ----- Save the model as an artifact -----
        model_path = f"model_store_{store_id}"
        m.save(f"{model_path}.joblib")
        mlflow.log_artifact(f"{model_path}.joblib")

    return m, forecast

def train_all_stores(limit=None, regressors=None):
    """
    Iterate over stores and train models. limit: number of stores (for quick testing).
    Saves models in models/.
    """
    df = load_raw()
    store_ids = sorted(df['Store'].unique())
    if limit:
        store_ids = store_ids[:limit]

    results = {}
    for s in tqdm(store_ids, desc="Training stores"):
        try:
            m, fc = train_store(df, s, regressors=regressors)
            results[s] = {"status": "ok"}
        except Exception as e:
            results[s] = {"status": "error", "error": str(e)}
    return results

if __name__ == "__main__":
    train_all_stores(limit=10, regressors=["Promo", "StateHoliday", "SchoolHoliday"])
