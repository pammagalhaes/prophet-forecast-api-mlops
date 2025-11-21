import pandas as pd

def basic_clean(df):
    """
    Basic cleaning step for Rossmann dataset.
    This version is robust: it does not break if some columns are missing.

    Steps:
    - Converts 'Date' column to datetime (if present)
    - Filters only rows where store was open (Open == 1), if column exists
    - Normalizes 'StateHoliday' to binary 0/1
    - Ensures regressors (Promo, StateHoliday, SchoolHoliday) exist and are integers
    """

    df = df.copy()

    # Convert Date to datetime if available
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure 'Open' column exists; if not, assume store is open
    if "Open" not in df.columns:
        df["Open"] = 1

    # Filter only open days
    df = df[df["Open"] != 0]

    # Ensure regressors exist — fill missing ones with 0
    for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
        if col not in df.columns:
            df[col] = 0

    # Normalize StateHoliday: '0' → 0, any other value → 1
    df["StateHoliday"] = df["StateHoliday"].replace("0", 0)
    df["StateHoliday"] = df["StateHoliday"].fillna(0)
    df["StateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if x in [0, "0"] else 1).astype(int)

    # SchoolHoliday and Promo as integers
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)
    df["Promo"] = df["Promo"].fillna(0).astype(int)

    return df


def to_prophet_df(df_store):
    """
    Converts store dataframe into Prophet-compatible format.

    Required:
    - 'Date' → renamed to 'ds'
    - 'Sales' → renamed to 'y'
    - Regressors: Promo, StateHoliday, SchoolHoliday

    This function is also robust:
    - Missing regressors are created as 0
    """

    df = df_store.copy()

    # Ensure regressors exist
    for col in ["Promo", "StateHoliday", "SchoolHoliday"]:
        if col not in df.columns:
            df[col] = 0

    # Select required columns (if Sales or Date missing, raise error)
    if "Date" not in df.columns:
        raise KeyError("Column 'Date' is required for Prophet.")
    if "Sales" not in df.columns:
        raise KeyError("Column 'Sales' is required for Prophet.")

    df = df[["Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday"]]

    # Rename to Prophet format
    df = df.rename(columns={"Date": "ds", "Sales": "y"})

    # Sort by date
    df = df.sort_values("ds").reset_index(drop=True)

    return df
