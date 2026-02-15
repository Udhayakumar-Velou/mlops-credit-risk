import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data.load_data import load_csv
from src.features.preprocess import preprocess


def main():
    mlflow.set_experiment("credit-risk-baseline")

    with mlflow.start_run(run_name="logreg_v1"):
        df = load_csv("data/raw/merged_data.csv")
        X_train, X_test, y_train, y_test = preprocess(df, target="default")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
