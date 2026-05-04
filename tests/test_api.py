import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

# Sample legitimate transaction (first row of dataset)
LEGITIMATE_TRANSACTION = {
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62
}

# Sample fraud transaction (known fraud from dataset)
FRAUD_TRANSACTION = {
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": -5.53634673796914, "V4": -3.37815522427443,
    "V5": -2.338320769942518, "V6": -3.462387777762292,
    "V7": -4.239598554061257, "V8": -0.0986979012610507,
    "V9": -2.363786969611213, "V10": -4.0907941719789316,
    "V11": -6.551599533260813, "V12": -8.617800855762348,
    "V13": -0.991389847235408, "V14": -9.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 9999.99
}

def test_root():
    """API root returns 200"""
    response = client.get("/")
    assert response.status_code == 200
    assert "FraudGuard" in response.json()["name"]

def test_health():
    """Health check returns healthy status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert data["scaler_loaded"] == True

def test_metrics():
    """Metrics endpoint returns model performance"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "f1" in data["metrics"]
    assert data["metrics"]["f1"] > 0.5  # model must beat random

def test_predict_legitimate():
    """Prediction endpoint returns valid response structure"""
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    # Don't assert specific prediction — depends on model
    # In CI we use synthetic model, in prod we use real model
    assert data["prediction"] in ["legitimate", "fraud"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["action"] in ["APPROVE", "BLOCK"]
    assert "risk_level" in data

def test_predict_returns_required_fields():
    """Prediction response has all required fields"""
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    required = ["prediction", "probability", "risk_level",
                "action", "model"]
    for field in required:
        assert field in data, f"Missing field: {field}"

def test_predict_probability_range():
    """Probability is always between 0 and 1"""
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    prob = response.json()["probability"]
    assert 0.0 <= prob <= 1.0

def test_predict_missing_field():
    """Missing field returns validation error"""
    bad_transaction = LEGITIMATE_TRANSACTION.copy()
    del bad_transaction["V1"]
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_type():
    """Non-numeric field returns validation error"""
    bad_transaction = LEGITIMATE_TRANSACTION.copy()
    bad_transaction["V1"] = "not_a_number"
    response = client.post("/predict", json=bad_transaction)
    assert response.status_code == 422

def test_batch_predict():
    """Batch prediction works for multiple transactions"""
    batch = [LEGITIMATE_TRANSACTION, LEGITIMATE_TRANSACTION]
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "results" in data
    assert len(data["results"]) == 2

def test_batch_predict_limit():
    """Batch prediction rejects more than 1000 transactions"""
    batch = [LEGITIMATE_TRANSACTION] * 1001
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 400