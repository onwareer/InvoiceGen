from sklearn.ensemble import IsolationForest
import pandas as pd

df = pd.read_csv("C:\\Python\\InvoiceGen\\AnomalyDetect\\sample_fee_data.csv")

df['FeeToValuationRatio'] = df['FeeAmount'] / df['Valuation']
features = df[['FeeToValuationRatio']]

model = IsolationForest(contamination=0.05)
df['anomaly'] = model.fit_predict(features)
df['is_anomaly'] = df['anomaly'] == -1

df[df['is_anomaly']].to_csv("flagged_anomalies.csv", index=False)