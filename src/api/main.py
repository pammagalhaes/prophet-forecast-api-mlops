from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.modeling.model_utils import load_model
from src.config import MODEL_DIR
from src.api.schemas import PredictRequest
from src.retraining.retrain import retrain_model

from src.monitoring.drift_detector import check_drift
from src.monitoring.prometheus_exporter import (
    update_drift_metrics,
    prometheus_response
)

app = FastAPI(title="Rossmann Prophet Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum periods per prediction request (prevents large forecast spike)
MAX_PERIODS = 7


@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.periods > MAX_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Max periods is {MAX_PERIODS}, got {req.periods}"
        )

    store_id = req.store_id

    # Load model from disk (no cache - minimizes RAM)
    try:
        model = load_model(store_id, MODEL_DIR)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot load model for store_id {store_id}: {e}"
        )

    # Create future dates for prediction
    future = model.make_future_dataframe(periods=req.periods, freq="D")

    # Constant regressors
    future["Promo"] = req.promo
    future["StateHoliday"] = req.stateholiday
    future["SchoolHoliday"] = req.schoolholiday

    forecast = model.predict(future)

    if not isinstance(forecast, pd.DataFrame):
        forecast = pd.DataFrame(forecast)

    future_forecast = forecast.tail(req.periods)[["ds", "yhat"]].copy()
    future_forecast["yhat"] = future_forecast["yhat"].round(2)

    # ───────────────────────────────────────────────
    # 1. Check DRIFT (lightweight cache prevents overload)
    # ───────────────────────────────────────────────
    share_drifted, drift_detected, report = check_drift()

    # Save metrics for Prometheus
    update_drift_metrics(drift_detected)

    # ───────────────────────────────────────────────
    # 2. Do not retrain inside the /predict endpoint
    # ───────────────────────────────────────────────

    return {
        "store_id": store_id,
        "periods": req.periods,
        "predictions": future_forecast.to_dict(orient="records"),
        "drift": {
            "share_drifted": share_drifted,
            "drift_detected": drift_detected,
        },
        "retrained": False
    }


@app.get("/metrics")
def metrics():
    data, content_type = prometheus_response()
    return Response(content=data, media_type=content_type)

@app.get("/monitor/drift")
def monitor_drift():
    share_drifted, drift_detected, report = check_drift()
    return {
        "share_drifted": share_drifted,
        "drift_detected": drift_detected,
         "drift_details": report
    }

@app.post("/retrain/{store_id}")
def retrain_endpoint(store_id: int):
    try:
        result = retrain_model(store_id=store_id)
        if not isinstance(result, dict) or "model" not in result:
            raise ValueError("retrain_model did not return a dict containing the key 'model'")
        return {
            "status": "success",
            "store_id": store_id,
            "message": "Model retrained.",
            "metrics": result.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
