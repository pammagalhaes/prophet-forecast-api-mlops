import pandas as pd
from tqdm import tqdm
from ..data.load_data import load_raw
from ..data.preprocess import basic_clean, to_prophet_df
from .model_utils import save_model
from ..config import TRAIN_TEST_SPLIT_DATE, MODEL_DIR
from prophet import Prophet

def train_store(df, store_id, split_date=TRAIN_TEST_SPLIT_DATE, regressors=None):
    """
    Treina um modelo Prophet para uma loja específica.
    df: DataFrame completo (merged train+store)
    store_id: int
    regressors: list of regressors to add to Prophet
    Retorna: fitted model, forecast (full in-sample + future)
    """
    df_store = df[df['Store'] == store_id].copy()
    if df_store.empty:
        raise ValueError(f"No data for store {store_id}")
    df_store = basic_clean(df_store)
    df_prophet = to_prophet_df(df_store)

    train_df = df_prophet[df_prophet['ds'] < pd.to_datetime(split_date)]
    if train_df.empty:
        raise ValueError("Train split resulted in empty dataset; choose earlier split_date.")

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    regressors = regressors or []
    for r in regressors:
        m.add_regressor(r)

    m.fit(train_df)

    # prepare future for whole period (train+test) + optionally horizon
    # to build final future use test.csv or make_future_dataframe as needed in forecasting step
    # Here we produce in-sample forecast to inspect performance.
    future = m.make_future_dataframe(periods=0, freq='D')  # in-sample only
    # merge regressors if present (align by ds)
    if regressors:
        regs = df_prophet.set_index('ds')[regressors]
        future = future.set_index('ds').join(regs, how='left').reset_index().rename(columns={'index': 'ds'})

    # forecast
    forecast = m.predict(future)

    # save model
    save_model(m, store_id, MODEL_DIR)

    return m, forecast

def train_all_stores(limit=None, regressors=None):
    """
    Itera por lojas e treina modelos. limit: number of stores (for quick test).
    Salva modelos em models/.
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
    # exemplo rápido: treinar primeiras 10 lojas com regressors
    train_all_stores(limit=10, regressors=["Promo", "StateHoliday", "SchoolHoliday"])
