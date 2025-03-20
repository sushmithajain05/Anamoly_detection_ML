import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv('elmeasure_energy_meter.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

contamination = 0.001  
iso_forest = IsolationForest(contamination=contamination, random_state=42)
df['iso_anomaly'] = iso_forest.fit_predict(df[['watts_total']])

df['iso_anomaly'] = df['iso_anomaly'] == -1
plt.figure(figsize=(12, 6))

plt.scatter(df['timestamp'][df['iso_anomaly']], df['watts_total'][df['iso_anomaly']], color='red', label='Point Anomaly', zorder=5)

plt.xticks(rotation=45, ha='right') 
plt.xlabel('Timestamp')
plt.ylabel('Watts Total')
plt.title('Point Anomaly Detection using Isolation Forest')
plt.legend()
plt.tight_layout()
plt.show()
print(df[['timestamp', 'device_id', 'watts_total', 'iso_anomaly']])
