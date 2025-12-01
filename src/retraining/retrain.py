import pandas as pd
from prophet import Prophet
import mlflow
from sklearn.metrics import mean_absolute_error

from src.data.load_data import load_raw
from src.data.preprocess import basic_clean, to_prophet_df
from src.modeling.model_utils import save_model
from src.config import TRAIN_TEST_SPLIT_DATE, MODEL_DIR


def retrain_model(store_id: int, regressors=None, forecast_horizon=30):

    df = load_raw()
    df_store = df[df["Store"] == store_id].copy()

    if df_store.empty:
        raise ValueError(f"Sem dados para o store {store_id}")

    df_store = basic_clean(df_store)
    df_prophet = to_prophet_df(df_store)

    if regressors is None:
        regressors = [c for c in df_prophet.columns if c not in ["ds", "y"]]

    df_train = df_prophet[df_prophet["ds"] < pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
    df_test = df_prophet[df_prophet["ds"] >= pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]

    with mlflow.start_run(run_name=f"retrain_store_{store_id}") as active_run:

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        for r in regressors:
            m.add_regressor(r)

        m.fit(df_train)

        future_all = df_prophet[["ds"] + regressors].copy()
        forecast_all = m.predict(future_all)

        if len(df_test) > 0:
            forecast_test = forecast_all[forecast_all["ds"] >= pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
            mae_test = mean_absolute_error(df_test["y"], forecast_test["yhat"])
            mlflow.log_metric("mae_test", mae_test)

        # previsao futura
        last_date = df_prophet["ds"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

        future_df = pd.DataFrame({"ds": future_dates})
        for r in regressors:
            future_df[r] = 0

        forecast_future = m.predict(future_df)

        mlflow.log_param("store_id", store_id)
        mlflow.log_param("regressors", regressors)
        mlflow.log_param("forecast_horizon", forecast_horizon)
        mlflow.log_param("type", "retrain")

        mlflow.prophet.log_model(m, artifact_path="model")

        run_id = active_run.info.run_id

    save_model(m, store_id, MODEL_DIR)

    return {
        "model": m,
        "future_forecast": forecast_future[["ds", "yhat"]],
        "run_id": run_id
    }
