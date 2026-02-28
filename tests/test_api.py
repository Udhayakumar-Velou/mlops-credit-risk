from fastapi.testclient import TestClient
from api.main import app
import pytest


class DummyModel:
    def predict(self, X):
        return [1]


@pytest.fixture(autouse=True)
def mock_model():
    from api import main
    main.model = DummyModel()


client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


def test_prediction_endpoint():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()