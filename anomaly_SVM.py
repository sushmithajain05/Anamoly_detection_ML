import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.describe())

data = df.drop(columns=['Time', 'Class'])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')  # nu is the fraction of outliers expected
svm.fit(data_scaled)

df['Predicted_Label'] = np.where(svm.predict(data_scaled) == -1, 'Anomaly', 'Normal')

anomalies = df[df['Predicted_Label'] == 'Anomaly']
anomalies.to_csv("detected_anomalies_svm.csv", index=False)
print(f"Anomalies detected: {len(anomalies)} (Saved to detected_anomalies_svm.csv)")

plt.figure(figsize=(10, 6))
plt.scatter(df['V1'], df['V2'], c=df['Predicted_Label'].map({'Normal': 'blue', 'Anomaly': 'red'}), edgecolors='k')
plt.title("Anomaly Detection using One-Class SVM")
plt.xlabel('V1')
plt.ylabel('V2')
plt.show()

print(f"Anomalies detected: {sum(df['Predicted_Label'] == 'Anomaly')}")
