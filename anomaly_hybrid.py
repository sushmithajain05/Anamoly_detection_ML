import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fftpack import fft
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
import dash
from dash import dcc, html, Input, Output

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

fft_values = np.abs(fft(df["power_load"].to_numpy()))
threshold_fft = np.percentile(fft_values, 95)
df["fft_anomaly"] = fft_values > threshold_fft

stl = STL(df["power_load"], period=1440, robust=True)
stl_result = stl.fit()
df["trend"] = stl_result.trend
df["seasonal"] = stl_result.seasonal
df["residual"] = stl_result.resid
df["stl_anomaly"] = np.abs(df["residual"]) > (2 * np.std(df["residual"]))


df["z_score"] = zscore(df["power_load"])
df["esd_anomaly"] = np.abs(df["z_score"]) > 3

window_size = 100
df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])
df["dynamic_anomaly"] = (df["power_load"] > df["upper_boundary"]) | (df["power_load"] < df["lower_boundary"])

df["final_anomaly"] = df[["fft_anomaly", "stl_anomaly", "esd_anomaly", "dynamic_anomaly"]].any(axis=1)
df_anomalies = df[df["final_anomaly"]]

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Hybrid Anomaly Detection on Power Load"),
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
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies["power_load"],
                             mode="markers", name="Anomaly", marker=dict(color="red", size=6)))
    fig.update_layout(title="Hybrid Anomaly Detection on Power Load",
                      xaxis_title="Timestamp", yaxis_title="Power Load",
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)