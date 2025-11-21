from fastapi import FastAPI, HTTPException
from ..modeling.model_utils import load_model
from ..config import MODEL_DIR
from .schemas import PredictRequest
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Rossmann Prophet Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# cache de modelos carregados
models = {}

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    store_id = req.store_id

    # se ainda não carregou, carrega e salva no cache
    if store_id not in models:
        try:
            models[store_id] = load_model(store_id, MODEL_DIR)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Não foi possível carregar o modelo do store_id {store_id}: {str(e)}"
            )

    model = models[store_id]

    # criar n dias no futuro
    future = model.make_future_dataframe(periods=req.periods, freq="D")

    # adiciona regressors constantes
    future["Promo"] = req.promo
    future["StateHoliday"] = req.stateholiday
    future["SchoolHoliday"] = req.schoolholiday

    # gerar previsão
    forecast = model.predict(future)

    # garantir que forecast é DataFrame
    if not isinstance(forecast, pd.DataFrame):
        forecast = pd.DataFrame(forecast)

    # apenas os períodos futuros
    future_forecast = forecast.tail(req.periods)

    return {
        "store_id": store_id,
        "periods": req.periods,
        "predictions": future_forecast[["ds", "yhat"]].to_dict(orient="records")
    }
