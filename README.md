# Credit Risk Modeling System for NBFC Loan Approvals

## Project Description
This project builds a machine learning–based credit risk modeling system for loan applicants at a Non-Banking Financial Company (NBFC). The system analyzes historical loan and applicant data to predict the probability of loan default and classify applicants into credit risk categories.

The project follows a structured MLOps workflow emphasizing reproducibility, experiment tracking, automated testing, and code quality. The solution is modular and scalable, allowing future integration of advanced modeling, deployment, and monitoring components.

---

## Task Definition
The objective is to develop a classification model that predicts loan default risk based on historical borrower and loan data. The model supports decision-making by categorizing applicants into risk segments.

The pipeline includes:

- Data ingestion and dataset merging
- Feature preprocessing
- Baseline model training
- Evaluation using classification metrics
- Experiment tracking with MLflow
- Automated testing with coverage validation
- Code quality enforcement with pre-commit hooks

---

## Dataset Source
The project uses structured tabular datasets provided for academic purposes:

- `customers.csv` – customer demographic and employment information
- `loans.csv` – loan attributes and default labels
- `bureau_data.csv` – credit bureau indicators

Datasets are merged using a shared customer identifier (`cust_id`) into a unified dataset:

```
merged_data.csv
```

This dataset is used for training and evaluation.

---

## Project Structure

```text
mlops-credit-risk/
├── data/
│   └── raw/
│       ├── customers.csv
│       ├── loans.csv
│       ├── bureau_data.csv
│       └── merged_data.csv
├── src/
│   ├── data/
│   │   └── load_data.py
│   └── features/
│       └── preprocess.py
├── scripts/
│   ├── merge_data.py
│   └── train_baseline.py
├── tests/
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Code Quality & Testing (Checkpoint 2)

This project enforces software engineering best practices:

- **Pre-commit hooks** for automated formatting and linting
- **Black** for code formatting
- **Flake8** for style checking
- **Pytest** for unit testing
- **Pytest-cov** for coverage reporting

Current unit test coverage: **100%**

Run tests:

```bash
uv run pytest
```

Run coverage report:

```bash
uv run pytest --cov=src --cov-report=term
```

---

## Experiment Tracking with MLflow

Model experiments are tracked using MLflow.

Each training run logs:

- model parameters
- accuracy metrics
- trained model artifacts

Run training:

```bash
uv run python -m scripts.train_baseline
```

Launch MLflow UI:

```bash
uv run mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

Experiments appear under:

```
credit-risk-baseline
```

---

## How to Run the Pipeline

Install dependencies:

```bash
uv sync
```

Merge datasets:

```bash
uv run python -m scripts.merge_data
```

Train baseline model:

```bash
uv run python -m scripts.train_baseline
```

---

## Team Members & Roles

This project was developed collaboratively by all team members. Roles indicate areas of focus rather than exclusive ownership.

## Udhayakumar Velou** – Data preprocessing and dataset handling
## Kishor Saravanan** – Model training and evaluation
## Bhavan Vasu** – Project structure and environment setup
## Siddiqui Kamran** – Testing support, experiment tracking, and documentation

---

## Contribution Statement

All team members contributed equally to:

- task definition and design
- dataset ingestion and preprocessing
- baseline model implementation
- automated testing
- MLflow experiment tracking
- version control collaboration
- documentation
