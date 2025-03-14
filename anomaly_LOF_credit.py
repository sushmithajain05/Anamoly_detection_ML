import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.describe())

if 'V1' not in df.columns or 'V2' not in df.columns:
    print("Error: V1 and V2 columns are missing from the dataset.")
else:
    data = df.drop(columns=['Time', 'Class'])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)  
    predictions = lof.fit_predict(data_scaled)

    df['Predicted_Label'] = np.where(predictions == -1, 'Anomaly', 'Normal')

    anomalies = df[df['Predicted_Label'] == 'Anomaly']
    print(f"Anomalies detected: {len(anomalies)}")

    anomalies.to_csv("detected_anomalies_lof.csv", index=False)

    plt.figure(figsize=(10, 6))
    if 'V1' in df.columns and 'V2' in df.columns:
        plt.scatter(df['V1'], df['V2'], c=df['Predicted_Label'].map({'Normal': 'blue', 'Anomaly': 'red'}), edgecolors='k')
        plt.title("Anomaly Detection using LOF")
        plt.xlabel('V1')
        plt.ylabel('V2')
        plt.show()
    else:
        print("V1 or V2 columns are not available for plotting.")

    print(f"Anomalies detected: {sum(df['Predicted_Label'] == 'Anomaly')}")
