# FraudGuard 🛡️

A production-grade MLOps pipeline for real-time credit card fraud detection. Built with XGBoost, FastAPI, MLflow, Docker, and GitHub Actions CI/CD.

![CI/CD](https://github.com/adhamkhaled787/FraudGuard/actions/workflows/retrain.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-MLOps-orange)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

FraudGuard detects credit card fraud in real time via a REST API. Send a transaction — get back a fraud/legitimate prediction with probability score and risk level in milliseconds.

```json
POST /predict
{
  "V1": -1.36, "V2": -0.07, ..., "Amount": 149.62
}

Response:
{
  "prediction": "legitimate",
  "probability": 0.0009,
  "risk_level": "MINIMAL",
  "action": "APPROVE",
  "model": "fraudguard_v1"
}
```

---

## Model performance

Trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate).

| Metric | Baseline | Deep XGBoost |
|---|---|---|
| F1 Score | 0.60 | **0.78** |
| Precision | 0.50 | **0.79** |
| Recall | 0.76 | **0.76** |
| ROC-AUC | 0.976 | **0.982** |
| False Alarms | 58 | **15** |

Best model: XGBoost with `max_depth=8`, `n_estimators=500`, SMOTE balancing.

---

## MLOps pipeline

```
Raw Data (creditcard.csv)
        │
        │  src/features.py
        │  StandardScaler + SMOTE balancing
        ▼
Processed Data (454,856 balanced rows)
        │
        │  src/train.py
        │  XGBoost + MLflow experiment tracking
        ▼
2 Experiments logged to MLflow UI
        │
        │  api/main.py
        │  FastAPI REST API
        ▼
5 Endpoints serving predictions
        │
        │  src/monitor.py
        │  KS test drift detection
        ▼
HTML drift report + JSON summary
        │
        │  Dockerfile + docker-compose.yml
        ▼
Containerized and portable
        │
        │  .github/workflows/retrain.yml
        ▼
Automated CI/CD on every push
```

---

## Project structure

```
fraudguard/
├── api/
│   └── main.py              # FastAPI REST API (5 endpoints)
├── src/
│   ├── features.py          # Feature engineering + SMOTE
│   ├── train.py             # XGBoost training + MLflow logging
│   ├── monitor.py           # KS drift detection + HTML report
│   └── predict.py           # Prediction utilities
├── tests/
│   └── test_api.py          # 10 API tests (pytest)
├── dags/
│   └── retrain_dag.py       # Airflow DAG for scheduled retraining
├── .github/
│   └── workflows/
│       └── retrain.yml      # GitHub Actions CI/CD
├── data/
│   ├── raw/                 # creditcard.csv (not committed)
│   └── processed/           # train/test splits (not committed)
├── models/                  # trained models (not committed)
├── mlruns/                  # MLflow experiment tracking
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech stack

| Tool | Purpose |
|---|---|
| `XGBoost` | Gradient boosted tree classifier |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `MLflow` | Experiment tracking and model registry |
| `FastAPI` | REST API with auto-generated Swagger docs |
| `Evidently / scipy` | Data drift detection (KS test) |
| `Docker` | Containerization |
| `GitHub Actions` | CI/CD pipeline |
| `pytest` | API testing (10 tests) |
| `scikit-learn` | Preprocessing and metrics |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/adhamkhaled787/FraudGuard.git
cd FraudGuard
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/creditcard.csv`.

---

## Usage

**Run the full pipeline:**
```bash
python src/features.py    # feature engineering
python src/train.py       # train + log to MLflow
python src/monitor.py     # drift detection report
```

**View MLflow experiments:**
```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

**Start the API:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

**Run with Docker:**
```bash
docker build -t fraudguard:v1 .
docker run -p 8000:8000 fraudguard:v1
```

**Run with Docker Compose:**
```bash
docker-compose up
```

**Run tests:**
```bash
pytest tests/test_api.py -v
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check (used by Docker) |
| GET | `/metrics` | Model performance metrics |
| POST | `/predict` | Predict single transaction |
| POST | `/predict/batch` | Predict up to 1000 transactions |

Full interactive docs at `http://localhost:8000/docs`

---

## CI/CD pipeline

Every push to `main` triggers automatically:

```
Push to GitHub
      │
      ▼
Job 1: Run API Tests (34s)
  install deps → generate synthetic model → pytest 10 tests
      │ pass
      ▼
Job 2: Build Docker Image (1m 26s)
  docker build → start container → health check /health
      │ pass
      ▼
Job 3: Pipeline Summary (4s)
  report results
```

---

## Key MLOps concepts demonstrated

**SMOTE balancing** — synthetic minority oversampling to handle 0.17% fraud rate without biasing the model toward "legitimate"

**MLflow experiment tracking** — every hyperparameter, metric, and model artifact logged and comparable across runs

**Training-serving consistency** — same StandardScaler saved at training time and loaded at serving time, preventing data leakage

**Drift detection** — KS test compares incoming transaction distributions to training data, alerting when the model may need retraining

**Health checks** — Docker and GitHub Actions both verify the API responds correctly before marking deployment as healthy

**Automated CI/CD** — model never ships without passing all 10 API tests and a Docker health check

---

## Honest limitations

- **No real-time data source** — uses static Kaggle dataset; production would connect to a live transaction stream
- **No cloud deployment** — runs locally; production would deploy to AWS/GCP/Railway
- **No Airflow scheduler** — DAG file included but requires Airflow server to run
- **Drift = real** — the dataset shows genuine temporal drift between training and test periods, which is why monitoring matters

---

## License

MIT — free to use, modify, and distribute.

---

Built by [Adham Khaled](https://github.com/adhamkhaled787) — Computer Engineering student at Alexandria University.

