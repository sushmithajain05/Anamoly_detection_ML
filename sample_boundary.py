import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

np.random.seed(42)
num_points = 200  
dates = [datetime.now() - timedelta(days=i) for i in range(num_points)]
values = np.sin(np.linspace(0, 20, num_points)) + np.random.normal(0, 0.2, num_points)

anomaly_indices = np.random.choice(num_points, 10, replace=False)
values[anomaly_indices] += np.random.uniform(2, 4, size=len(anomaly_indices))

df = pd.DataFrame({"timestamp": dates, "value": values})
df = df.sort_values("timestamp") 

window_size = 10  
df["expected_value"] = df["value"].rolling(window=window_size, center=True).mean()
df["std_dev"] = df["value"].rolling(window=window_size, center=True).std()
df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

model = IsolationForest(contamination=0.05, random_state=42)  
df["anomaly_score"] = model.fit_predict(df[["value"]])

plt.figure(figsize=(12, 6))

plt.plot(df["timestamp"], df["value"], label="Value", color="blue", linewidth=1.5)

plt.plot(df["timestamp"], df["expected_value"], label="Expected Value", color="green", linestyle="dashed", linewidth=1.5)

plt.fill_between(df["timestamp"], df["lower_boundary"], df["upper_boundary"], color='lightblue', alpha=0.4, label="Boundary")

plt.scatter(df["timestamp"][df["anomaly_score"] == -1], 
            df["value"][df["anomaly_score"] == -1], 
            color="red", label="Anomaly", marker="o", s=100)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Anomaly Detection with Boundaries (Using Isolation Forest & Moving Average)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
