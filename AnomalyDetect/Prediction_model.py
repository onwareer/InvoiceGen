import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Reuse or create mock valuation data
dates_q1 = pd.date_range(start="2025-01-01", end="2025-03-31", freq="B")
accounts = [f"A{str(i).zfill(3)}" for i in range(1, 11)]
data = []

for account in accounts:
    for date in dates_q1:
        if account == "A001" and date.month == 2 and date.day in [5, 6, 14, 21]:
            continue  # simulate missing data
        if account == "A002" and date.month == 3:
            valuation = 1_200_000  # flat valuation
        else:
            base_val = np.random.choice([1_000_000, 1_100_000, 1_200_000])
            variation = np.random.uniform(-10000, 10000)
            valuation = base_val + variation
        if account == "A003" and date == pd.Timestamp("2025-03-14"):
            valuation *= 1.5  # spike anomaly
        data.append({"AccountID": account, "Date": date, "Valuation": round(valuation, 2)})

# Convert to DataFrame
df = pd.DataFrame(data)
df["Month"] = df["Date"].dt.month

# Feature engineering per AccountID per month
features = []
labels = {}

for (account, month), group in df.groupby(["AccountID", "Month"]):
    group = group.sort_values("Date")
    key = f"{account}-{month}"
    expected_days = pd.date_range(group["Date"].min(), group["Date"].max(), freq="B")
    missing_days = len(set(expected_days) - set(group["Date"]))
    pct_change = group["Valuation"].pct_change().abs()
    pct_flat = (group["Valuation"].diff().abs() < 1e-3).mean()
    volatility = group["Valuation"].std()

    feature = {
        "AccountMonth": key,
        "AccountID": account,
        "Month": month,
        "MissingDays": missing_days,
        "MaxPctChange": pct_change.max(),
        "PctFlatDays": pct_flat,
        "ValuationVolatility": volatility
    }

    # Simulated labels
    if missing_days >= 3 or pct_change.max() > 0.2 or pct_flat > 0.9 or volatility < 500:
        labels[key] = 1  # Anomalous
    else:
        labels[key] = 0  # Normal

    features.append(feature)

# Prepare training dataset
features_df = pd.DataFrame(features)
features_df["Label"] = features_df["AccountMonth"].map(labels)

X = features_df[["MissingDays", "MaxPctChange", "PctFlatDays", "ValuationVolatility"]]
y = features_df["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))