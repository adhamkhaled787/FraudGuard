import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Paths ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH    = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_raw():
    """Load raw creditcard.csv"""
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['Class'].mean():.3%}")
    print(f"Columns: {df.shape[1]}")
    return df

def engineer_features(df):
    """
    Feature engineering:
    1. Scale Amount (varies wildly — $0 to $25,000)
    2. Drop Time (raw seconds since first transaction — not useful)
    3. Keep V1-V28 as-is (already PCA-transformed by the dataset)
    """
    df = df.copy()

    # Scale Amount
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

    # Save scaler for use at prediction time
    joblib.dump(scaler, os.path.join(MODELS_DIR, "amount_scaler.pkl"))

    # Drop original Amount and Time
    df = df.drop(columns=['Time', 'Amount'])

    print(f"Features after engineering: {df.shape[1] - 1} features + 1 label")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Time-aware split — fraud detection needs to generalize to future transactions.
    We use the last 20% of rows as test set (preserves temporal order).
    """
    # Sort by index (data is already time-ordered)
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train.drop(columns=['Class'])
    y_train = train['Class']
    X_test  = test.drop(columns=['Class'])
    y_test  = test['Class']

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Train fraud rate: {y_train.mean():.3%}")
    print(f"Test fraud rate:  {y_test.mean():.3%}")

    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE: Synthetic Minority Oversampling Technique.
    Creates synthetic fraud examples so the model sees
    enough fraud cases to learn meaningful patterns.
    Only applied to training data — never to test data.
    """
    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE:  {pd.Series(y_resampled).value_counts().to_dict()}")
    print(f"Training set size: {len(X_resampled):,}")

    return X_resampled, y_resampled

def save_processed(X_train, X_test, y_train, y_test):
    """Save all splits to data/processed/"""
    X_train.to_csv(os.path.join(PROC_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROC_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(PROC_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROC_DIR,  "y_test.csv"),  index=False)
    print(f"\nSaved processed data to data/processed/ ✓")

def run_pipeline():
    """Run the full feature engineering pipeline"""
    print("=" * 50)
    print("FraudGuard — Feature Engineering Pipeline")
    print("=" * 50)

    df          = load_raw()
    df          = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, y_train = apply_smote(X_train, y_train)
    save_processed(X_train, X_test, y_train, y_test)

    print("\nPipeline complete ✓")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_pipeline()