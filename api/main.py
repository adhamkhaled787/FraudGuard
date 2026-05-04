from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load model and scaler at startup ──────────────────────
# Loaded once when the API starts — not on every request
print("Loading model...")
model  = joblib.load(os.path.join(MODELS_DIR, "fraudguard_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "amount_scaler.pkl"))
print("Model loaded ✓")

# Load metrics
metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path) as f:
    model_metrics = json.load(f)

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="FraudGuard API",
    description="Real-time credit card fraud detection powered by XGBoost",
    version="1.0.0"
)

# ── Input schema ──────────────────────────────────────────
# Pydantic validates every incoming request automatically
class Transaction(BaseModel):
    V1: float;  V2: float;  V3: float;  V4: float
    V5: float;  V6: float;  V7: float;  V8: float
    V9: float;  V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float

# ── Helper ────────────────────────────────────────────────
def preprocess(transaction: Transaction) -> np.ndarray:
    """Apply same preprocessing as training pipeline"""
    data = transaction.model_dump()
    amount = data.pop("Amount")

    # Scale amount using saved scaler
    amount_scaled = scaler.transform([[amount]])[0][0]

    # Build feature array in correct order
    features = [data[f"V{i}"] for i in range(1, 29)]
    features.append(amount_scaled)

    return np.array(features).reshape(1, -1)

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:  return "HIGH"
    if probability >= 0.5:  return "MEDIUM"
    if probability >= 0.3:  return "LOW"
    return "MINIMAL"

# ── Endpoints ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "FraudGuard",
        "version":     "1.0.0",
        "description": "Real-time credit card fraud detection",
        "endpoints":   ["/predict", "/health", "/metrics", "/docs"]
    }

@app.get("/health")
def health():
    """Health check — used by Docker and deployment platforms"""
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/metrics")
def get_metrics():
    """Return model performance metrics"""
    return {
        "model":   "XGBoost (deep_xgboost)",
        "metrics": model_metrics,
        "dataset": "Kaggle Credit Card Fraud Detection",
        "notes":   "Metrics computed on held-out test set (20% of data)"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.
    Returns prediction, probability, and risk level.
    """
    try:
        # Preprocess
        features = preprocess(transaction)

        # Predict
        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        risk_level  = get_risk_level(probability)

        return {
            "prediction":   "fraud" if prediction == 1 else "legitimate",
            "probability":  round(probability, 4),
            "risk_level":   risk_level,
            "action":       "BLOCK" if prediction == 1 else "APPROVE",
            "model":        "fraudguard_v1"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    """Predict fraud for multiple transactions at once"""
    if len(transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 1000 transactions"
        )

    results = []
    for t in transactions:
        features    = preprocess(t)
        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        results.append({
            "prediction":  "fraud" if prediction == 1 else "legitimate",
            "probability": round(probability, 4),
            "risk_level":  get_risk_level(probability),
            "action":      "BLOCK" if prediction == 1 else "APPROVE"
        })

    fraud_count = sum(1 for r in results if r["prediction"] == "fraud")
    return {
        "total":       len(results),
        "fraud_count": fraud_count,
        "fraud_rate":  round(fraud_count / len(results), 4),
        "results":     results
    }