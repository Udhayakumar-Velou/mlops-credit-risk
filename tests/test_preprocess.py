from src.features.preprocess import preprocess
from src.data.load_data import load_csv


def test_preprocess_split():
    df = load_csv("data/raw/merged_data.csv")
    X_train, X_test, y_train, y_test = preprocess(df, target="default")

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
