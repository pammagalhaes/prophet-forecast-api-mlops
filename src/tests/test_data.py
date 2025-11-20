import pandas as pd
from src.data.preprocess import basic_clean, to_prophet_df

def test_basic_clean():
    df = pd.DataFrame({"Date": ["2022-01-01"], "Sales": [100]})
    cleaned = basic_clean(df)
    assert "Date" in cleaned.columns

def test_to_prophet_df():
    df = pd.DataFrame({
        "Date": ["2022-01-01"],
        "Sales": [200],
        "Promo": [1],
        "StateHoliday": ["0"],
        "SchoolHoliday": [0],
    })

    prophet_df = to_prophet_df(df)

    assert "ds" in prophet_df.columns
    assert "y" in prophet_df.columns
    assert len(prophet_df) == 1
