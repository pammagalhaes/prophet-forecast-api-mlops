import pandas as pd
from ..config import TRAIN_CSV, STORE_CSV

def load_raw():
    """
    Carrega train.csv e store.csv e faz merge.
    Retorna DataFrame com colunas originais + store features.
    """
    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    store = pd.read_csv(STORE_CSV, low_memory=False)
    df = train.merge(store, on="Store", how="left")
    return df
