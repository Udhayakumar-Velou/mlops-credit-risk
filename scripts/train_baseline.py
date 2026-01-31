from src.data.load_data import load_csv
from src.features.preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # Load data
    df = load_csv("data/raw/merged_data.csv")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess(df, target="default")

    # Train baseline model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Baseline Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()