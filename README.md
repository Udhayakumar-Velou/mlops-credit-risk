# Credit Risk Modeling System for NBFC Loan Approvals

## Project Overview

This project implements an end-to-end credit risk prediction system for a Non-Banking Financial Company (NBFC). The objective is to predict whether a loan applicant is likely to default using structured borrower and loan data.

The project follows modern **MLOps best practices**, including:

- Modular project structure  
- Reproducible environment management using UV  
- Automated testing with Pytest  
- Experiment tracking with MLflow  
- Model serving using FastAPI  
- Containerization using Docker  
- Continuous Integration using GitHub Actions  

---

## 🎥 Project Demonstration Video

A full demonstration of the system including the CI pipeline, model training, MLflow tracking, FastAPI inference, and Docker deployment can be viewed here:

Demo Video Link:  
[Watch the Demo Video](https://drive.google.com/file/d/1dn9jMCYCtilEjLjYG8mIhi79VjJBPvzP/view?usp=sharing)

The video demonstrates:
- Automatic CI pipeline execution
- Model training and experiment tracking with MLflow
- FastAPI model serving
- Prediction endpoint usage
- Docker container deployment

---

## Problem Definition

Loan approval decisions require evaluating whether a borrower is likely to repay a loan.

This project models the problem as a **binary classification task**:

- `default = 1` → borrower likely to default  
- `default = 0` → borrower unlikely to default  

The trained model supports **data-driven loan approval decisions**.

---

## Dataset Description

The project uses structured tabular datasets:

- **customers.csv** – customer demographic and employment information  
- **loans.csv** – loan attributes and default labels  
- **bureau_data.csv** – credit bureau indicators  

These datasets are merged using the identifier:

```
cust_id
```

The final dataset used for training is:

```
merged_data.csv
```

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
│   │   └── load_data.py
│   └── features/
│       └── preprocess.py
│
├── scripts/
│   ├── merge_data.py
│   └── train_baseline.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_preprocess.py
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

## System Architecture

The system follows a modular MLOps architecture separating data processing, model training, experiment tracking, and model serving.

First, raw datasets are merged using the pipeline implemented in `merge_data.py`. The merged dataset is then processed through preprocessing functions in the `src` module to prepare the features for model training.

The baseline machine learning model is trained using `train_baseline.py`. During training, **MLflow** is used to track experiment parameters, metrics, and model artifacts.

The trained model is saved as `model.pkl` in the `artifacts` directory. This model is then loaded by a **FastAPI inference service**, which exposes REST endpoints for predictions.

The API is containerized using **Docker**, allowing the application to run consistently across different environments.

A **GitHub Actions CI pipeline** ensures code quality by automatically running tests and linting checks whenever code changes are pushed to the repository.

---

## MLOps Practices Applied

This project demonstrates several key MLOps practices:

- Reproducible dependency management using UV  
- Modular project architecture  
- Experiment tracking with MLflow  
- Automated testing with Pytest  
- Code quality checks using Flake8 and Black  
- API-based model serving with FastAPI  
- Containerized deployment with Docker  
- Continuous Integration using GitHub Actions  

---

## Running the Project

### 1. Clone the repository

```
git clone https://github.com/Udhayakumar-Velou/mlops-credit-risk.git
cd mlops-credit-risk
```

### 2. Create environment and install dependencies

```
uv sync
```

### 3. Train the model

```
uv run python -m scripts.train_baseline
```

### 4. Run the API

```
uv run uvicorn api.main:app --reload
```

Open the API documentation:

```
http://localhost:8000/docs
```

---

## FastAPI Endpoints

### Health Check

```
GET /health
```

Response:

```
{
  "status": "API is running"
}
```

---

### Prediction Endpoint

```
POST /predict
```

Example Request:

```
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

Example Response:

```
{
  "prediction": 0
}
```

---

## Docker Deployment

Build the Docker image:

```
docker build -t credit-risk-api .
```

Run the container:

```
docker run -p 8000:8000 credit-risk-api
```

Access API documentation:

```
http://localhost:8000/docs
```

---

## Continuous Integration

The project uses **GitHub Actions** for Continuous Integration. The CI pipeline automatically runs whenever code is pushed or a pull request is created.

The pipeline performs the following checks:

- Dependency installation  
- Automated testing using Pytest  
- Code linting using Flake8  
- Code formatting validation using Black  

This ensures consistent code quality and prevents broken code from being merged into the main branch.

---

## Monitoring & Reliability

Basic monitoring is implemented through application logging and health checks.

The FastAPI service logs important events such as:

- model loading  
- prediction requests  
- prediction outputs  
- runtime errors  

The `/health` endpoint allows external systems to verify that the API service is running.

In production environments, monitoring could be extended using tools such as **Prometheus** and **Grafana** to track API performance, request latency, error rates, and model prediction distributions.

---

## Limitations & Future Work

Although the system demonstrates a complete MLOps workflow, several improvements could enhance the project:

- evaluating more advanced models such as Random Forest or XGBoost  
- implementing automated model retraining pipelines  
- deploying the system to cloud platforms such as AWS or Azure  
- implementing model drift detection  
- adding a user interface dashboard for loan officers  
- implementing a full Continuous Deployment pipeline  

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
