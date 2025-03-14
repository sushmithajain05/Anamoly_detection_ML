import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

window_size = 100
df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
df["z_score"] = (df["power_load"] - df["expected_value"]) / df["std_dev"]

df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

threshold = 3
df_anomalies = df[(df["z_score"].abs() > threshold)]
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
                f"Feature: Power Load<br>"
                f"Expected Value: {row['expected_value']:.2f}<br>"
                f"Actual Value: {row['power_load']:.2f}<br>"
                f"Deviation: {row['deviation']:.2f}<br>"
                f"Reason: {row['reason']}", axis=1
)

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
                             mode="lines", name="Actual Value", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled["expected_value"],
                             mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
                             y=df_sampled["upper_boundary"].tolist() + df_sampled["lower_boundary"].tolist()[::-1],
                             fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                             line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies["power_load"],
                             mode="markers", name="Anomaly", marker=dict(color="red", size=6),
                             text=filtered_anomalies["hover_text"], hoverinfo="text"))
    
    fig.update_layout(title="Anomaly Detection on Power Load",
                      xaxis_title="Timestamp", yaxis_title="Power Load",
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)