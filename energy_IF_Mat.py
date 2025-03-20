import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('elmeasure_energy_meter.csv')
print(df.head())
df['timestamp'] = pd.to_datetime(df['timestamp'])

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'frequency', 'kwh_imp']
X = df[features]

iso_forest = IsolationForest(contamination=0.002, random_state=42)
y_pred = iso_forest.fit_predict(X)

df['anomaly'] = y_pred

anomalies = df[df['anomaly'] == -1]

anomalies.to_csv('anomalies.csv', index=False)

plt.figure(figsize=(20, 6))
plt.plot(anomalies['timestamp'], anomalies['watts_total'], color='red', label='Anomaly Data (Line)', linestyle=':', marker='*')
plt.title('Anomaly Detection using Isolation Forest (with Time and Watts Total)')
plt.xlabel('Timestamp')
plt.ylabel('Watts Total')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()