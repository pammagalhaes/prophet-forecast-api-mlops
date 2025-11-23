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
    # Filter only this store
    df_store = df[df["Store"] == store_id].copy()
    if df_store.empty:
        raise ValueError(f"No data for store {store_id}")

    # Basic cleaning -> fill NA, convert types, etc.
    df_store = basic_clean(df_store)

    # Convert to Prophet-ready dataframe
    df_prophet = to_prophet_df(df_store)

    # Split into train and test using the provided or default split date
    df_train = df_prophet[df_prophet["ds"] < pd.to_datetime(split_date)]
    df_test = df_prophet[df_prophet["ds"] >= pd.to_datetime(split_date)]

    # Determine regressors if not passed
    if regressors is None:
        regressors = [c for c in df_prophet.columns if c not in ["ds", "y"]]

    with mlflow.start_run(run_name=f"store_{store_id}"):

        # Initialize Prophet and attach regressors
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        for r in regressors:
            model.add_regressor(r)

        # Fit only on the training data
        model.fit(df_train)

        # Forecast on the entire available df_prophet (train + test)
        future = df_prophet[["ds"] + regressors].copy()
        forecast = model.predict(future)

        # Compute test metrics only if test exists
        if len(df_test) > 0:
            forecast_test = forecast[forecast["ds"] >= pd.to_datetime(split_date)]
            mae_test = mean_absolute_error(df_test["y"], forecast_test["yhat"])
            mlflow.log_metric("mae_test", mae_test)

        # Log useful params
        mlflow.log_param("store_id", store_id)
        mlflow.log_param("split_date", str(split_date))
        mlflow.log_param("n_regressors", len(regressors))

        # Log model artifact
        mlflow.prophet.log_model(model, artifact_path="model")

    # Save model locally
    save_model(model, store_id)

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
