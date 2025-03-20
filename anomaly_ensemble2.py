import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import t
import dash
from dash import dcc, html, Input, Output

SI_UNITS = {
    "power_load": "kW"  
}
file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

df = df.set_index("timestamp").resample("1T").mean(numeric_only=True).interpolate().reset_index()

window_size = 100
df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

def generalized_esd_test(series, alpha=0.05):
    anomalies = []
    temp_series = series.copy()
    for _ in range(10):  
        mean = temp_series.mean()
        std_dev = temp_series.std()
        max_dev = (temp_series - mean).abs().max()
        test_stat = max_dev / std_dev
        critical_value = t.ppf(1 - alpha / (2 * len(temp_series)), len(temp_series) - 1)
        if test_stat > critical_value:
            anomaly_index = (temp_series - mean).abs().idxmax()
            anomalies.append(anomaly_index)
            temp_series = temp_series.drop(anomaly_index)
        else:
            break
    return anomalies

anomaly_indices = generalized_esd_test(df["power_load"])
df["anomaly_score"] = 0
df.loc[anomaly_indices, "anomaly_score"] = 1

df_anomalies = df[df["anomaly_score"] == 1]
df_anomalies["deviation"] = df_anomalies["power_load"] - df_anomalies["expected_value"]

def get_anomaly_reason(row):
    if row["power_load"] > row["upper_boundary"]:
        return "Sudden spike: Power load exceeded expected range."
    elif row["power_load"] < row["lower_boundary"]:
        return "Sudden drop: Power load fell below expected range."
    else:
        return "Unusual fluctuation detected."

df_anomalies["reason"] = df_anomalies.apply(get_anomaly_reason, axis=1)

df_anomalies["hover_text"] = df_anomalies.apply(
    lambda row: f"Date & Time: {row['timestamp']}<br>"
                f"Feature: Power Load ({SI_UNITS['power_load']})<br>"
                f"Expected Value: {row['expected_value']:.2f} {SI_UNITS['power_load']}<br>"
                f"Actual Value: {row['power_load']:.2f} {SI_UNITS['power_load']}<br>"
                f"Deviation: {row['deviation']:.2f} {SI_UNITS['power_load']}<br>"
                f"Reason: {row['reason']}", axis=1
)

df_anomalies.to_csv("anomalies_detected.csv", index=False)
print("Anomalies saved to anomalies_detected.csv")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Anomaly Detection on Power Load"),
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=df['timestamp'].min().date(),
        max_date_allowed=df['timestamp'].max().date(),
        start_date=df['timestamp'].min().date(),
        end_date=df['timestamp'].max().date()
    ),
    dcc.Graph(id='anomaly-graph')
])

@app.callback(
    Output('anomaly-graph', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_graph(start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    filtered_anomalies = df_anomalies[(df_anomalies['timestamp'] >= start_date) & (df_anomalies['timestamp'] <= end_date)]
    df_sampled = filtered_df.iloc[::40]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled["power_load"],
                             mode="lines", name=f"Actual Value ({SI_UNITS['power_load']})", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled["expected_value"],
                             mode="lines", name=f"Expected Value ({SI_UNITS['power_load']})", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
                             y=df_sampled["upper_boundary"].tolist() + df_sampled["lower_boundary"].tolist()[::-1],
                             fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                             line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies["power_load"],
                             mode="markers", name=f"Anomaly ({SI_UNITS['power_load']})", marker=dict(color="red", size=6),
                             text=filtered_anomalies["hover_text"], hoverinfo="text"))
    
    fig.update_layout(title="Anomaly Detection on Power Load",
                      xaxis_title="Timestamp", 
                      yaxis_title=f"Power Load ({SI_UNITS['power_load']})",
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
