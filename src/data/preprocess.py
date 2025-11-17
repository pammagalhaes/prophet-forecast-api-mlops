import pandas as pd

def basic_clean(df):
    """
    Limpeza básica do Rossmann:
     - datas como datetime
     - filtra Open == 1 (evitar muitos zeros), mas você pode manter se quiser
     - converte StateHoliday para inteiro (0/1) para ser usado como regressor
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    # Keep rows where store was open (you can change policy if prefer)
    df = df[df['Open'] != 0]
    # normalize StateHoliday: '0' -> 0, 'a','b','c' -> 1
    df['StateHoliday'] = df['StateHoliday'].replace('0', 0)
    df['StateHoliday'] = df['StateHoliday'].fillna(0)
    df['StateHoliday'] = df['StateHoliday'].apply(lambda x: 0 if x == 0 else 1).astype(int)
    df['SchoolHoliday'] = df['SchoolHoliday'].fillna(0).astype(int)
    df['Promo'] = df['Promo'].fillna(0).astype(int)
    return df

def to_prophet_df(df_store):
    """
    Recebe df filtrado por loja e retorna DataFrame com colunas:
    ds (datetime), y (sales), Promo, StateHoliday, SchoolHoliday
    """
    df = df_store[['Date', 'Sales', 'Promo', 'StateHoliday', 'SchoolHoliday']].copy()
    df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    df = df.sort_values('ds').reset_index(drop=True)
    return df
