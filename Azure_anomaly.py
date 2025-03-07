
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

file_path = "EB_2021.csv"  
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df = df.dropna(subset=["power_load"])

window_size = 100  
df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)  
df["anomaly_score"] = model.fit_predict(df[["power_load"]])

df_anomalies = df[df["anomaly_score"] == -1]
df_anomalies["deviation"] = df_anomalies["power_load"] - df_anomalies["expected_value"]

def determine_anomaly_reason(row):
    if row["power_load"] > row["upper_boundary"]:
        return "Unusual surge in power load (Above expected range)"
    elif row["power_load"] < row["lower_boundary"]:
        return "Unexpected drop in power load (Below expected range)"
    else:
        return "Detected as anomaly by Isolation Forest"

df_anomalies["anomaly_reason"] = df_anomalies.apply(determine_anomaly_reason, axis=1)

df_anomalies["hover_text"] = df_anomalies.apply(
    lambda row: f"Date & Time: {row['timestamp']}<br>"
                f"Feature: Power Load<br>"
                f"Expected Value: {row['expected_value']:.2f}<br>"
                f"Actual Value: {row['power_load']:.2f}<br>"
                f"Deviation: {row['deviation']:.2f}<br>"
                f"Reason: {row['anomaly_reason']}", axis=1
)

df_anomalies.to_csv("detected_anomalies_with_reasons.csv", index=False)

df_sampled = df.iloc[::40]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_sampled["timestamp"], y=df_sampled["power_load"],
    mode="lines", name="Actual value ",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=df_sampled["timestamp"], y=df_sampled["expected_value"],
    mode="lines", name="Expected Value",
    line=dict(color="green", dash="dash")
))

fig.add_trace(go.Scatter(
    x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
    y=df_sampled["upper_boundary"].tolist() + df_sampled["lower_boundary"].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(173, 216, 230, 0.4)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name="Boundary"
))

fig.add_trace(go.Scatter(
    x=df_anomalies["timestamp"], y=df_anomalies["power_load"],
    mode="markers", name="Anomaly",
    marker=dict(color="red", size=6),
    text=df_anomalies["hover_text"],
    hoverinfo="text"
))

fig.update_layout(
    title="Anomaly Detection on power load",
    xaxis_title="Timestamp",
    yaxis_title="Power Load",
    legend=dict(x=0, y=1),
    xaxis=dict(tickangle=45),
    template="plotly_white"
)

fig.show()