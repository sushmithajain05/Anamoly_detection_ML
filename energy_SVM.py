import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

df = pd.read_csv('elmeasure_energy_meter.csv')

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'frequency', 'kwh_imp']
X = df[features]

one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
y_pred_svm = one_class_svm.fit_predict(X)

df['anomaly_svm'] = y_pred_svm
anomalies_svm = df[df['anomaly_svm'] == -1]
anomalies_svm.to_csv('anomalies_svm.csv', index=False)
plt.figure(figsize=(10, 6))
plt.scatter(df[df['anomaly_svm'] == 1].index, df[df['anomaly_svm'] == 1]['watts_total'], color='blue', label='Normal', alpha=0.6)

plt.scatter(anomalies_svm.index, anomalies_svm['watts_total'], color='red', label='Anomaly', alpha=0.8)
plt.title('Anomaly Detection using One-Class SVM')
plt.xlabel('Index')
plt.ylabel('Watts Total')
plt.legend()
plt.show()
