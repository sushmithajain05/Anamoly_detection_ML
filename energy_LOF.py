import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv('elmeasure_energy_meter.csv')

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'kwh_imp']
X = df[features]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X)

df['anomaly_lof'] = y_pred_lof
anomalies_lof = df[df['anomaly_lof'] == -1]
anomalies_lof.to_csv('anomalies_lof.csv', index=False)

plt.figure(figsize=(10, 6))

plt.scatter(df[df['anomaly_lof'] == 1].index, df[df['anomaly_lof'] == 1]['watts_total'], color='blue', label='Normal', alpha=0.6)

plt.scatter(anomalies_lof.index, anomalies_lof['watts_total'], color='red', label='Anomaly', alpha=0.8)

plt.title('Anomaly Detection using Local Outlier Factor (LOF)')
plt.xlabel('Index')
plt.ylabel('Watts Total')
plt.legend()
plt.show()
