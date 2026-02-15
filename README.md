# Credit Risk Modeling System for NBFC Loan Approvals

## Project Description
This project focuses on building a machine learning–based credit risk modeling system for loan applicants at a Non-Banking Financial Company (NBFC). The system analyzes historical loan and applicant data to predict the probability of loan default and classify applicants into different credit risk categories.

The project emphasizes a structured and reproducible approach to machine learning development, including data loading, preprocessing, baseline model training, and clear project organization. The solution is designed to be modular and scalable, allowing future integration of advanced modeling, deployment, and monitoring components.

---

## Task Definition
The primary task of this project is to develop a machine learning model that predicts the likelihood of loan default based on historical borrower and loan data. Using this prediction, applicants are classified into distinct credit risk categories to support informed lending decisions.

The task includes:

Loading and preprocessing structured loan application data

Training a baseline classification model for credit risk prediction

Evaluating model performance using standard classification metrics

Establishing a modular and reproducible project structure that can be extended for production use

---

## Dataset Source
The project uses structured tabular datasets provided as CSV files for academic
purposes:

- `customers.csv` – customer demographic and employment information
- `loans.csv` – loan-level attributes and default labels
- `bureau_data.csv` – credit bureau indicators

The datasets are merged using a common customer identifier (`cust_id`) to create
a unified dataset (`merged_data.csv`) used for model training.

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
├── pyproject.toml
├── uv.lock
└── README.md
```
---

## How to Run (Checkpoint 1)

### 1. Install dependencies

uv sync

uv run python -m scripts.merge_data

uv run python -m scripts.train_baseline

## Team Members & Roles


This project was developed collaboratively by all team members, with shared
responsibility across all stages of the MLOps pipeline. Roles are defined to
reflect areas of focus rather than exclusive ownership.

**Udhayakumar Velou** – Data preprocessing, feature preparation, and dataset handling

**Bhavan Vasu** – Model development, baseline training, and evaluation

**Kishor Saravanan** – Project structure setup, environment management, and documentation

### Contribution Statement
All team members contributed equally to:
- defining the machine learning task
- designing the project structure
- implementing data ingestion and preprocessing
- developing and validating the baseline model
- version control and GitHub collaboration
- project documentation
