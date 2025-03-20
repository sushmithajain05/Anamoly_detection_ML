import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

df = pd.read_csv('healthcare_dataset.csv')

print(df.head())

df_numeric = df[['Age', 'Billing Amount', 'Room Number']] 

df_numeric = df_numeric.fillna(df_numeric.mean())  

scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_numeric)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005)  
outliers = lof.fit_predict(features_scaled)

df['Outlier'] = outliers

print(f"Number of anomalies detected: {sum(df['Outlier'] == -1)}")

anomalies = df[df['Outlier'] == -1]
anomalies.to_csv("detected_anomalies1.csv", index=False)
print(f"Anomalies detected: {len(anomalies)} (Saved to detected_anomalies.csv)")

colors = ['blue' if o == 1 else 'red' for o in df['Outlier']]

plt.scatter(df['Age'], df['Billing Amount'], c=colors, alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Billing Amount')
plt.title('Anomaly Detection using LOF')

plt.show()
