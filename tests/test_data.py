import pandas as pd
from src.data.load_data import load_csv


def test_load_csv():
    df = load_csv("data/raw/customers.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
