import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.ensemble import IsolationForest
import datetime

st.set_page_config(page_title="Fee + Valuation Anomaly Detector", layout="wide")
st.title("Fee & Valuation Discrepancy Detector")

# Upload Fee Data
st.header("Upload Fee Data (for Anomaly Detection)")
fee_file = st.file_uploader("Upload a CSV with columns: AccountID, Valuation, FeeAmount", type="csv")

# Upload Valuation Data
st.header("Upload Daily Valuation Data (for Discrepancy Detection)")
val_file = st.file_uploader("Upload a CSV with columns: AccountID, Date, Valuation", type="csv")

# Process Fee Anomalies
if fee_file:
    df_fee = pd.read_csv(fee_file)
    df_fee['FeeToValuationRatio'] = df_fee['FeeAmount'] / df_fee['Valuation']
    model = IsolationForest(contamination=0.05, random_state=42)
    df_fee['anomaly'] = model.fit_predict(df_fee[['FeeToValuationRatio']])
    df_fee['is_anomaly'] = df_fee['anomaly'] == -1

    st.subheader("Fee Anomalies Detected")
    st.dataframe(df_fee[df_fee['is_anomaly']])

# Process Valuation Discrepancies
if val_file:
    df_val = pd.read_csv(val_file, parse_dates=["Date"])
    issues = []
    full_q1 = pd.date_range(start="2025-01-01", end="2025-03-31", freq="B")

    for account_id, group in df_val.groupby("AccountID"):
        group = group.sort_values("Date")
        expected_dates = full_q1
        actual_dates = set(group["Date"])
        missing_dates = sorted(set(expected_dates) - actual_dates)
        if missing_dates:
            issues.append({
                "AccountID": account_id,
                "Issue": "Missing valuation days",
                "Details": f"{len(missing_dates)} missing days in Q1"
            })

        for month in [1, 2, 3]:
            month_data = group[group["Date"].dt.month == month]
            if not month_data.empty and month_data["Valuation"].nunique() == 1:
                issues.append({
                    "AccountID": account_id,
                    "Issue": "Flat valuation in month",
                    "Details": f"All values same in {month_data['Date'].iloc[0].strftime('%B')}"
                })

        pct_change = group["Valuation"].pct_change().abs()
        if (pct_change > 0.2).any():
            spikes = group[pct_change > 0.2]
            for _, row in spikes.iterrows():
                issues.append({
                    "AccountID": account_id,
                    "Issue": "Irregular valuation change",
                    "Details": f"Valuation jump >20% on {row['Date'].date()}"
                })

    issues_df = pd.DataFrame(issues)
    st.subheader("Valuation Discrepancies")
    st.dataframe(issues_df)