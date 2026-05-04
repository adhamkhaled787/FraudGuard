# Run this as a quick debug script
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

BASE_DIR = "/Users/adham/Desktop/adham/fraudguard"

# Load raw data
df = pd.read_csv(f"{BASE_DIR}/data/raw/creditcard.csv")
scaler = joblib.load(f"{BASE_DIR}/models/amount_scaler.pkl")
df['Amount_scaled'] = scaler.transform(df[['Amount']])
df = df.drop(columns=['Time', 'Amount', 'Class'])

# Split same way as features.py
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# Compare distributions
print("Feature comparison (train vs test mean):")
for col in ['V1', 'V2', 'V3', 'V14', 'Amount_scaled']:
    train_mean = train[col].mean()
    test_mean  = test[col].mean()
    diff       = abs(train_mean - test_mean)
    print(f"  {col:20} train={train_mean:.4f}  test={test_mean:.4f}  diff={diff:.4f}")

print(f"\nTrain size: {len(train):,}")
print(f"Test size:  {len(test):,}")