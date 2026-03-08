from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI App Metadata
# -----------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    description="""
This API provides a machine learning service to predict whether a loan applicant
is a **credit risk**.

Features:
- Health check endpoint
- Credit risk prediction endpoint
- Model served using FastAPI
""",
    version="1.0.0",
)

model = None


# -----------------------------
# Load Model at Startup
# -----------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("artifacts/model.pkl")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API service is running.",
    tags=["System"]
)
def health():
    return {"status": "API is running"}


# -----------------------------
# Request Schema
# -----------------------------
class PredictionRequest(BaseModel):
    sanction_amount: float
    loan_amount: float
    processing_fee: float
    gst: float
    net_disbursement: float
    loan_tenure_months: int
    principal_outstanding: float
    bank_balance_at_application: float
    age: int
    income: float
    number_of_dependants: int
    years_at_current_address: int
    zipcode: int
    number_of_open_accounts: int
    number_of_closed_accounts: int
    total_loan_months: int
    delinquent_months: int
    total_dpd: int
    enquiry_count: int
    credit_utilization_ratio: float

    class Config:
        json_schema_extra = {
            "example": {
                "sanction_amount": 20000,
                "loan_amount": 18000,
                "processing_fee": 200,
                "gst": 36,
                "net_disbursement": 17764,
                "loan_tenure_months": 24,
                "principal_outstanding": 15000,
                "bank_balance_at_application": 5000,
                "age": 35,
                "income": 45000,
                "number_of_dependants": 2,
                "years_at_current_address": 5,
                "zipcode": 75001,
                "number_of_open_accounts": 3,
                "number_of_closed_accounts": 2,
                "total_loan_months": 48,
                "delinquent_months": 0,
                "total_dpd": 0,
                "enquiry_count": 1,
                "credit_utilization_ratio": 0.3
            }
        }


# -----------------------------
# Response Schema
# -----------------------------
class PredictionResponse(BaseModel):
    prediction: int


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Credit Risk",
    description="Predict whether the applicant represents a credit risk.",
    tags=["Prediction"]
)
def predict(data: PredictionRequest):
    try:
        logger.info("Prediction request received")

        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)[0]

        prediction = int(bool(prediction))

        logger.info(f"Prediction result: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e