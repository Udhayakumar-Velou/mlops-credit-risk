# Credit Risk Modeling System for NBFC Loan Approvals

## Project Overview

This project implements an end-to-end **credit risk prediction system** for a Non-Banking Financial Company (NBFC). The objective is to predict whether a loan applicant is likely to default using structured borrower and loan data.

The project follows modern **MLOps best practices**, including:

- Modular project structure  
- Reproducible environment management using UV  
- Automated testing with Pytest  
- Experiment tracking with MLflow  
- Model serving using FastAPI  
- Containerization using Docker  

---

## Problem Definition

This is a **binary classification problem**:

- `default = 1` → Loan likely to default  
- `default = 0` → Loan unlikely to default  

The trained model supports data-driven loan approval decisions.

---

## Dataset Description

The project uses structured tabular datasets:

- **customers.csv** – customer demographic and employment information  
- **loans.csv** – loan attributes and default labels  
- **bureau_data.csv** – credit bureau indicators  

These datasets are merged using `cust_id` into:

```
merged_data.csv
```

This dataset is used for training and evaluation.

---

## Project Structure

```
mlops-credit-risk/
├── api/
│   └── main.py
├── artifacts/
│   └── model.pkl
├── data/
│   └── raw/
├── src/
│   ├── data/
│   └── features/
├── scripts/
│   ├── merge_data.py
│   └── train_baseline.py
├── tests/
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Checkpoint 1 – Project Foundations

- UV environment setup  
- Modular project structure  
- Dataset merging pipeline  
- Baseline model training  
- Initial documentation  

---

# Checkpoint 2 – Code Quality & Experiment Tracking

The project integrates:

- Pre-commit hooks  
- Black for formatting  
- Flake8 for linting  
- Pytest for testing  
- Pytest-cov for coverage  
- MLflow for experiment tracking  

### Run Tests

```bash
uv run pytest
```

### Run Coverage

```bash
uv run pytest --cov=src --cov=api --cov-report=term
```

### Train Model

```bash
uv run python -m scripts.train_baseline
```

### Launch MLflow UI

```bash
uv run mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

Experiment name:

```
credit-risk-baseline
```

---

# Checkpoint 3 – Model Serving & Containerization

## FastAPI Inference Service

### Health Endpoint

```
GET /
```

Response:

```json
{
  "status": "API is running"
}
```

---

### Prediction Endpoint

```
POST /predict
```

### Example Request

```json
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
```

### Example Response

```json
{
  "prediction": 0
}
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t credit-risk-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 credit-risk-api
```

Access API documentation:

```
http://localhost:8000/docs
```

---

## Monitoring & Logging

The system includes:

- Model load logging  
- Request logging  
- Prediction logging  
- Error handling and logging  

---

# MLOps Practices Applied

- Reproducible dependency management (UV)  
- Modular architecture  
- Experiment tracking (MLflow)  
- Automated testing  
- API-based deployment  
- Docker containerization  
- Logging for monitoring  

---

## Team Members & Roles

Udhayakumar Velou – Data preprocessing and API development  
Bhavan Vasu – Environment setup and project structure  
Kishor Saravanan – Model training and evaluation  
Siddiqui Kamran – Testing, experiment tracking, documentation  

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
