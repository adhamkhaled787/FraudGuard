import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import joblib
import os
import json

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

def load_processed():
    """Load processed train/test splits"""
    X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).squeeze()

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test

def evaluate(model, X_test, y_test):
    """Compute all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1":               f1_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred),
        "recall":           recall_score(y_test, y_pred),
        "roc_auc":          roc_auc_score(y_test, y_prob),
        "avg_precision":    average_precision_score(y_test, y_prob),
    }

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]:,}  (legit correctly identified)")
    print(f"  False Positives: {cm[0][1]:,}  (legit wrongly flagged)")
    print(f"  False Negatives: {cm[1][0]:,}  (fraud missed ← dangerous)")
    print(f"  True Positives:  {cm[1][1]:,}  (fraud correctly caught)")

    return metrics

def train(params=None, run_name="fraudguard_run"):
    """
    Train XGBoost model with MLflow tracking.
    Every metric, parameter, and model artifact is logged.
    """

    # Default parameters
    default_params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,   # SMOTE already balanced — no need to adjust
        "random_state":     42,
        "n_jobs":           -1,
        "eval_metric":      "aucpr",  # area under precision-recall curve
    }

    if params:
        default_params.update(params)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("fraudguard")

    with mlflow.start_run(run_name=run_name):

        print(f"\nStarting MLflow run: {run_name}")

        # Load data
        X_train, X_test, y_train, y_test = load_processed()

        # Log parameters
        mlflow.log_params(default_params)

        # Train model
        print("\nTraining XGBoost...")
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Evaluate
        print("\nEvaluating...")
        metrics = evaluate(model, X_test, y_test)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log model to MLflow
        mlflow.xgboost.log_model(model, "model")

        # Save model locally too
        model_path = os.path.join(MODELS_DIR, "fraudguard_model.pkl")
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path} ✓")

        # Save metrics as JSON for the API to read
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMLflow run complete ✓")
        print(f"Key metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, metrics

if __name__ == "__main__":
    print("=" * 50)
    print("FraudGuard — Model Training + MLflow Tracking")
    print("=" * 50)

    # Run 1 — baseline
    model, metrics = train(run_name="baseline_xgboost")

    # Run 2 — deeper trees
    print("\n" + "=" * 50)
    print("Running experiment 2 — deeper trees")
    print("=" * 50)
    model, metrics = train(
        params={"max_depth": 8, "n_estimators": 500},
        run_name="deep_xgboost"
    )