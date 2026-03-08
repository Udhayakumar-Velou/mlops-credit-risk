# Credit Risk Modeling System for NBFC Loan Approvals

## Project Overview

This project implements an end-to-end credit risk prediction system for a Non-Banking Financial Company (NBFC). The objective is to predict whether a loan applicant is likely to default using structured borrower and loan data.

The system follows modern MLOps best practices, including:

- Modular project structure
- Reproducible environment management using UV
- Automated testing with Pytest
- Experiment tracking using MLflow
- Model serving using FastAPI
- Containerization using Docker
- Continuous Integration using GitHub Actions

---

## Problem Definition

Loan approval decisions require evaluating whether a borrower is likely to repay a loan.

This project models the problem as a binary classification task:

- default = 1 → borrower likely to default
- default = 0 → borrower unlikely to default

The trained model supports data-driven loan approval decisions.

---

## Dataset Description

The project uses structured tabular datasets:

- customers.csv – customer demographic and employment information
- loans.csv – loan attributes and default labels
- bureau_data.csv – credit bureau indicators

These datasets are merged using the common identifier:

cust_id

The final training dataset becomes:

merged_data.csv

This dataset is used for model training and evaluation.

---

## Project Structure
```
mlops-credit-risk/
│
├── api/
│   └── main.py
│
├── artifacts/
│   └── model.pkl
│
├── data/
│   └── raw/
│
├── src/
│   ├── data/
│   └── features/
│
├── scripts/
│   ├── merge_data.py
│   └── train_baseline.py
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Checkpoint 1 — Project Foundations

- UV environment setup
- Modular project structure
- Dataset merging pipeline
- Baseline model training
- Initial documentation

---

## Checkpoint 2 — Code Quality & Experiment Tracking

The project integrates:

- Pre-commit hooks
- Black for formatting
- Flake8 for linting
- Pytest for testing
- Pytest-cov for coverage
- MLflow for experiment tracking

Run tests:

uv run pytest

Run coverage:

uv run pytest --cov=src --cov=api --cov-report=term

Train model:

uv run python -m scripts.train_baseline

Launch MLflow UI:

uv run mlflow ui

Open:

http://127.0.0.1:5000

Experiment name:

credit-risk-baseline

---

## Checkpoint 3 — Model Serving & Containerization

### FastAPI Inference Service

Health endpoint:

GET /health

Response:

{
  "status": "API is running"
}

Prediction endpoint:

POST /predict

Example request:

{
  "sanction_amount": 50000,
  "loan_amount": 45000,
  "processing_fee": 1000,
  "gst": 180,
  "net_disbursement": 43820,
  "loan_tenure_months": 24,
  "principal_outstanding": 30000,
  "bank_balance_at_application": 150000,
  "age": 30,
  "income": 60000,
  "number_of_dependants": 2,
  "years_at_current_address": 3,
  "zipcode": 560001,
  "number_of_open_accounts": 2,
  "number_of_closed_accounts": 1,
  "total_loan_months": 36,
  "delinquent_months": 0,
  "total_dpd": 0,
  "enquiry_count": 1,
  "credit_utilization_ratio": 0.35
}

Example response:

{
  "prediction": 0
}

---

## Docker Deployment

Build Docker image:

docker build -t credit-risk-api .

Run Docker container:

docker run -p 8000:8000 credit-risk-api

API documentation:

http://localhost:8000/docs

---

## Continuous Integration (CI)

The project includes a GitHub Actions CI pipeline that automatically runs whenever code is pushed or a pull request is created.

The pipeline performs:

- Dependency installation
- Automated testing using Pytest
- Code linting using Flake8
- Code formatting validation using Black

This ensures consistent code quality and prevents broken code from being merged into the main branch.

---

## Monitoring Strategy

Basic monitoring is implemented through logging and health checks.

The API logs important events such as:

- model loading
- prediction requests
- prediction outputs
- runtime errors

The `/health` endpoint allows external systems to verify that the API service is running.

In a production environment, monitoring could be extended using tools such as Prometheus and Grafana to track API performance, error rates, and model prediction patterns.

---

## Future Work

Several improvements could extend this system:

- Training more advanced models such as Random Forest or XGBoost
- Implementing automated model retraining pipelines
- Deploying the system to cloud platforms such as AWS or Azure
- Implementing model drift detection
- Adding a dashboard interface for loan officers
- Implementing a full CD (Continuous Deployment) pipeline

---

## MLOps Practices Applied

This project demonstrates several important MLOps practices:

- Reproducible dependency management using UV
- Modular project architecture
- Experiment tracking using MLflow
- Automated testing using Pytest
- Code quality enforcement using Flake8 and Black
- Containerized deployment using Docker
- Continuous Integration using GitHub Actions
- Logging for monitoring and debugging

---

## Team Members & Roles

Udhayakumar Velou – Data preprocessing and API development  
Bhavan Vasu – Environment setup and project structure  
Kishor Saravanan – Model training and evaluation  
Siddiqui Kamran – Testing, experiment tracking, and documentation

---

## Contribution Statement

All team members contributed collaboratively to:

- project design and task planning
- dataset ingestion and preprocessing
- baseline model implementation
- automated testing
- experiment tracking using MLflow
- version control collaboration
- documentation and final reporting