import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.describe())

data = df.drop(columns=['Time', 'Class'])

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

predictions = model.predict(data)
print(predictions)
df['Predicted_Label'] = predictions
df['Predicted_Label'] = df['Predicted_Label'].replace({1: 'Normal', -1: 'Anomaly'})

anomalies = df[df['Predicted_Label'] == 'Anomaly']
anomalies.to_csv("detected_anomalies.csv", index=False)
print(f"Anomalies detected: {len(anomalies)} (Saved to detected_anomalies.csv)")

plt.figure(figsize=(10, 6))
plt.scatter(df['V1'], df['V2'], c=df['Predicted_Label'].map({'Normal': 'blue', 'Anomaly': 'red'}), edgecolors='k')
plt.title("Anomaly Detection")
plt.xlabel('V1')
plt.ylabel('V2')
plt.show()

print(f"Anomalies detected: {sum(df['Predicted_Label'] == 'Anomaly')}")

