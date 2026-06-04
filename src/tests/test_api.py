from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

def test_root_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("src.api.main.load_model")
def test_predict_ok(mock_load_model):
    """Test the /predict route using a mocked model (no .joblib needed)."""

    # create fake model
    fake_model = MagicMock()

    # when the endpoint calls model.make_future_dataframe(...)
    fake_model.make_future_dataframe.return_value = {"ds": [1, 2, 3]}

    # when it calls model.predict(...)
    fake_model.predict.return_value = {
        "ds": [1, 2, 3],
        "yhat": [100, 110, 120]
    }

    mock_load_model.return_value = fake_model

    body = {
        "store_id": 1,
        "periods": 3,
        "promo": 0,
        "stateholiday": "0",
        "schoolholiday": 0
    }

    response = client.post("/predict", json=body)

    assert response.status_code == 200

    json_data = response.json()

    assert json_data["store_id"] == 1
    assert json_data["periods"] == 3
    assert len(json_data["predictions"]) == 3


@patch("src.api.main.check_drift")
@patch("src.retraining.retrain.retrain_model")
@patch("src.api.main.load_model")
def test_predict_does_not_retrain(mock_load_model, mock_retrain_model, mock_check_drift):
    fake_model = MagicMock()
    fake_model.make_future_dataframe.return_value = {"ds": [1, 2, 3]}
    fake_model.predict.return_value = {"ds": [1, 2, 3], "yhat": [100, 110, 120]}
    mock_load_model.return_value = fake_model
    mock_check_drift.return_value = (0.2, True, {})

    body = {
        "store_id": 1,
        "periods": 3,
        "promo": 0,
        "stateholiday": "0",
        "schoolholiday": 0
    }

    response = client.post("/predict", json=body)

    assert response.status_code == 200
    assert mock_retrain_model.call_count == 0

    json_data = response.json()
    assert json_data["drift"]["drift_detected"] is True
    assert json_data["retrained"] is False

