import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats

# ── Paths ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_reference_data(sample_size=5000):
    """
    Load ORIGINAL training data before SMOTE as reference.
    Must match real-world distribution — not synthetic balanced data.
    """
    # Read raw data and take the train portion (first 80%)
    raw_path = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
    df       = pd.read_csv(raw_path)

    # Apply same feature engineering as features.py
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler             = joblib.load(
        os.path.join(BASE_DIR, "models", "amount_scaler.pkl"))
    df['Amount_scaled'] = scaler.transform(df[['Amount']])
    df = df.drop(columns=['Time', 'Amount', 'Class'])

    # Take train portion (first 80%)
    train_end = int(len(df) * 0.8)
    reference = df.iloc[:train_end].sample(
        n=min(sample_size, train_end), random_state=42)

    print(f"Reference data: {len(reference):,} rows (pre-SMOTE)")
    return reference

def load_current_data(sample_size=5000):
    """Load test data as proxy for current production data"""
    X_test  = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
    current = X_test.sample(
        n=min(sample_size, len(X_test)), random_state=42)
    print(f"Current data: {len(current):,} rows")
    return current

def run_ks_drift(reference, current, threshold=0.05):
    """
    Run Kolmogorov-Smirnov test on each feature.
    KS test checks if two samples come from the same distribution.
    p-value < threshold → distributions differ → drift detected
    """
    results = {}
    drifted = []

    for col in reference.columns:
        stat, p_value = stats.ks_2samp(
            reference[col].dropna(),
            current[col].dropna()
        )
        drifted_col = p_value < threshold
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value":      round(float(p_value), 4),
            "drifted":      drifted_col
        }
        if drifted_col:
            drifted.append(col)

    return results, drifted

def generate_html_report(reference, current, drift_results,
                          drifted_cols, report_path):
    """Generate a standalone HTML drift report"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    n_cols    = len(reference.columns)
    n_drifted = len(drifted_cols)
    drift_pct = n_drifted / n_cols * 100

    # Build feature drift table rows
    table_rows = ""
    for col, res in sorted(drift_results.items(),
                           key=lambda x: x[1]['p_value']):
        status = "🔴 DRIFT" if res['drifted'] else "🟢 OK"
        color  = "#fff0f0" if res['drifted'] else "#f0fff0"
        table_rows += f"""
        <tr style="background:{color}">
            <td>{col}</td>
            <td>{res['ks_statistic']}</td>
            <td>{res['p_value']}</td>
            <td>{status}</td>
        </tr>"""

    # Build distribution plots for top drifted features
    plot_cols  = drifted_cols[:3] if drifted_cols \
                 else list(reference.columns)[:3]
    plots_html = ""

    for col in plot_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=reference[col], name="Reference (Training)",
            opacity=0.7, marker_color="#3498db",
            nbinsx=50))
        fig.add_trace(go.Histogram(
            x=current[col], name="Current (Production)",
            opacity=0.7, marker_color="#e74c3c",
            nbinsx=50))
        fig.update_layout(
            title=f"Distribution: {col}",
            barmode="overlay",
            height=300,
            template="plotly_white",
            legend=dict(orientation="h")
        )
        plots_html += pio.to_html(
            fig, full_html=False, include_plotlyjs=False)

    # Full HTML
    overall_status = "⚠️ DRIFT DETECTED" \
                     if n_drifted > n_cols * 0.5 else "✅ NO DRIFT"
    status_color   = "#e74c3c" \
                     if n_drifted > n_cols * 0.5 else "#2ecc71"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>FraudGuard — Drift Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif;
               margin: 40px; background: #f5f5f5; }}
        .card {{ background: white; padding: 24px;
                border-radius: 8px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a3a5c; }}
        h2 {{ color: #2c5f8a; }}
        .status {{ font-size: 24px; font-weight: bold;
                  color: {status_color}; }}
        .metric {{ display: inline-block; margin: 10px 20px;
                  text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold;
                        color: #1a3a5c; }}
        .metric-label {{ color: #666; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #1a3a5c; color: white;
             padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🛡️ FraudGuard — Data Drift Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p class="status">{overall_status}</p>
        <div>
            <div class="metric">
                <div class="metric-value">{n_cols}</div>
                <div class="metric-label">Total Features</div>
            </div>
            <div class="metric">
                <div class="metric-value"
                     style="color:#e74c3c">{n_drifted}</div>
                <div class="metric-label">Drifted Features</div>
            </div>
            <div class="metric">
                <div class="metric-value">{drift_pct:.1f}%</div>
                <div class="metric-label">Drift Share</div>
            </div>
            <div class="metric">
                <div class="metric-value">5,000</div>
                <div class="metric-label">Samples Compared</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Feature Distributions (Top Drifted)</h2>
        {plots_html}
    </div>

    <div class="card">
        <h2>Feature Drift Details (KS Test)</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>KS Statistic</th>
                <th>P-Value</th>
                <th>Status</th>
            </tr>
            {table_rows}
        </table>
    </div>
</body>
</html>"""

    with open(report_path, "w") as f:
        f.write(html)
    print(f"HTML report saved: {report_path} ✓")

def run_monitoring_pipeline():
    """Full monitoring pipeline"""
    print("=" * 50)
    print("FraudGuard — Drift Monitoring Pipeline")
    print("=" * 50)

    reference = load_reference_data()
    current   = load_current_data()

    print("\nRunning KS drift tests...")
    drift_results, drifted_cols = run_ks_drift(reference, current)

    n_cols      = len(reference.columns)
    n_drifted   = len(drifted_cols)
    drift_share = n_drifted / n_cols

    print(f"\nDrift Summary:")
    print(f"  Drifted columns: {n_drifted}/{n_cols}")
    print(f"  Drift share:     {drift_share:.1%}")

    drift_detected = drift_share > 0.5
    if drift_detected:
        print("\n⚠️  ALERT: Significant drift detected!")
        print("   Consider retraining the model.")
    else:
        print("\n✓  No significant drift detected.")

    if drifted_cols:
        print(f"\nDrifted features: {drifted_cols[:5]}")

    # Generate HTML report
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        REPORTS_DIR, f"drift_report_{timestamp}.html")
    generate_html_report(
        reference, current, drift_results,
        drifted_cols, report_path)

    # Save JSON summary
    summary = {
        "timestamp":      datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "drift_share":    round(drift_share, 4),
        "drifted_columns": drifted_cols,
        "total_columns":  n_cols,
        "report_path":    report_path,
        "status":         "ALERT" if drift_detected else "OK"
    }

    summary_path = os.path.join(REPORTS_DIR, "latest_drift.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved ✓")
    print("\nMonitoring pipeline complete ✓")
    return summary

if __name__ == "__main__":
    run_monitoring_pipeline()