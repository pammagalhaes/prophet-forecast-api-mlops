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
    """Testa a rota /predict usando MODEL MOCKADO (sem precisar de .joblib)."""

    # ----- cria modelo falso
    fake_model = MagicMock()

    # quando o endpoint chamar model.make_future_dataframe(...)
    fake_model.make_future_dataframe.return_value = {"ds": [1, 2, 3]}

    # quando chamar model.predict(...)
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

