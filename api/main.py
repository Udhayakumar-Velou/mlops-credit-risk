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
# FastAPI App
# -----------------------------
app = FastAPI(title="Credit Risk Model API")

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
@app.get("/")
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


# -----------------------------
# Response Schema
# -----------------------------
class PredictionResponse(BaseModel):
    prediction: int


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    try:
        logger.info("Prediction request received")

        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)[0]

        # Convert bool to int safely
        prediction = int(bool(prediction))

        logger.info(f"Prediction result: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e