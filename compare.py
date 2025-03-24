import pandas as pd

df1 = pd.read_csv("anomalies_detected_hybrid.csv")
df2 = pd.read_csv("anomalies_detected.csv")

df1["timestamp"] = pd.to_datetime(df1["id"], errors="coerce")
df2["timestamp"] = pd.to_datetime(df2["id"], errors="coerce")

df1_anomalies = df1[df1["final_anomaly"] == True][["id"]]
df2_anomalies = df2[df2["anomaly_score"] == -1][["id"]]

common_anomalies = pd.merge(df1_anomalies, df2_anomalies, on=["id"], how="inner")

print(f"Number of common anomalies: {len(common_anomalies)}")
common_anomalies.to_csv("common_anomalies.csv", index=False)
