import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.data.load_data import load_csv
from src.features.preprocess import preprocess


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk-baseline")

    with mlflow.start_run(run_name="logreg_v1"):

        # Load data
        df = load_csv("data/raw/merged_data.csv")

        # Preprocess
        X_train, X_test, y_train, y_test = preprocess(df, target="default")

        print("\n===== FEATURE COLUMNS =====")
        print(list(X_train.columns))
        print("===========================\n")

        # Train model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")

        # Log parameters
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", 2000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
