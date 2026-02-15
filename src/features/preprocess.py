from sklearn.model_selection import train_test_split


def preprocess(df, target: str):
    # Separate target FIRST
    y = df[target]

    # Drop target and ID columns from features
    X = df.drop(columns=[target, "loan_id", "cust_id"])

    # Keep only numeric features
    X = X.select_dtypes(include=["number"])

    return train_test_split(X, y, test_size=0.2, random_state=42)
