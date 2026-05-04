# ── Base image ────────────────────────────────────────────
# Start from official Python 3.11 slim image
# slim = smaller size, no unnecessary packages
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────
LABEL maintainer="Adham Khaled"
LABEL description="FraudGuard — Credit Card Fraud Detection API"
LABEL version="1.0.0"

# ── Working directory ─────────────────────────────────────
# All commands run from /app inside the container
WORKDIR /app

# ── System dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────
# Copy requirements first — Docker caches this layer
# Only reinstalls if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn xgboost \
    imbalanced-learn fastapi uvicorn \
    joblib scipy plotly

# ── Copy project files ────────────────────────────────────
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/
COPY data/processed/ ./data/processed/

# ── Environment variables ─────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────
# Tells Docker this container listens on port 8000
EXPOSE 8000

# ── Health check ──────────────────────────────────────────
# Docker checks this every 30 seconds
# If it fails 3 times → container marked unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start command ─────────────────────────────────────────
# This runs when the container starts
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]