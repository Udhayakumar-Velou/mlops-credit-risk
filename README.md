# Credit Risk Modeling System for NBFC Loan Approvals

## Project Description
This project focuses on building a machine learning–based credit risk modeling system for loan applicants at a Non-Banking Financial Company (NBFC). The system analyzes historical loan and applicant data to predict the probability of loan default and classify applicants into different credit risk categories.

The project emphasizes a structured and reproducible approach to machine learning development, including data loading, preprocessing, baseline model training, and clear project organization. The solution is designed to be modular and scalable, allowing future integration of advanced modeling, deployment, and monitoring components.

---

## Task Definition
The machine learning task is a **binary classification problem**.

**Objective:**  
Predict whether a loan applicant will default (`default = True / False`) based on:
- customer demographic information
- loan-related attributes
- credit bureau indicators

A baseline Logistic Regression model is trained to validate the complete ML
pipeline and ensure reproducibility.

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

- **Udhayakumar Velou**  
  Project coordination, MLOps pipeline design, data integration, and baseline
  model implementation

- **Kishor Saravanan**  
  Data understanding, feature analysis, preprocessing support, and model
  validation

- **Bhavan Vasu**  
  Documentation, experimentation support, code review, and reproducibility
  validation

### Contribution Statement
All team members contributed equally to:
- defining the machine learning task
- designing the project structure
- implementing data ingestion and preprocessing
- developing and validating the baseline model
- version control and GitHub collaboration
- project documentation
