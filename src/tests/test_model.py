import pandas as pd
from src.modeling.train import train_store

def test_train_store_basic():
    df = pd.DataFrame({
        "Store": [1]*5,
        "Date": pd.to_datetime(["2022-01-01","2022-01-02",
                                "2022-01-03","2022-01-04","2022-01-05"]),
        "Sales": [100, 110, 105, 120, 130],
        "Promo": [1,0,1,0,1],
        "StateHoliday": ["0","0","0","0","0"],
        "SchoolHoliday": [0,0,1,0,0]
    })

    model, forecast = train_store(df, store_id=1, split_date="2022-01-04")

    assert forecast is not None
    assert len(forecast) > 0
