import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

def validate_prophet(df_prophet, n_splits=3, regressors=None):
    """
    Validação temporal com TimeSeriesSplit.
    df_prophet: DataFrame ordenado por ds com colunas ds,y e regressors (se usados).
    regressors: list of regressors names to add to Prophet.
    Retorna dict com métricas médias e listas.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    rmses = []

    if regressors is None:
        regressors = []

    for train_idx, test_idx in tscv.split(df_prophet):
        train_df = df_prophet.iloc[train_idx]
        test_df = df_prophet.iloc[test_idx]

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

        for r in regressors:
            m.add_regressor(r)

        m.fit(train_df)

        # prepare future = test_df with regressors
        future = test_df[['ds'] + regressors].reset_index(drop=True)
        forecast = m.predict(future)
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        maes.append(mae)
        rmses.append(rmse)

    return {
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "maes": [float(x) for x in maes],
        "rmses": [float(x) for x in rmses]
    }
