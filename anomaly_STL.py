import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from statsmodels.tsa.seasonal import STL

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

def detect_anomalies_stl():
    period = 1440 
    
    df_sorted = df.set_index("timestamp")
    numeric_cols = df_sorted.select_dtypes(include=["number"]).columns
    df_resampled = df_sorted[numeric_cols].resample("T").mean().interpolate() 
    
    if df_resampled.index.freq is None:
        df_resampled.index.freq = "T"  

    if len(df_resampled) < period:
        period = max(2, len(df_resampled) // 10)  

    df_resampled["power_load"] = pd.to_numeric(df_resampled["power_load"], errors="coerce")
    df_resampled = df_resampled.dropna()


    stl = STL(df_resampled["power_load"], seasonal=period)
    result = stl.fit()

    residual = result.resid
    threshold = 3 * residual.std()

    anomalies = df_resampled[np.abs(residual) > threshold].reset_index()
    anomalies["deviation"] = anomalies["power_load"] - df_resampled["power_load"].rolling(window=100, center=True, min_periods=1).mean()

    anomalies["hover_text"] = anomalies.apply(
        lambda row: f"Date & Time: {row['timestamp']}<br>"
                    f"Feature: Power Load<br>"
                    f"Expected Value: {row['power_load']:.2f}<br>"
                    f"Actual Value: {row['power_load']:.2f}<br>"
                    f"Deviation: {row['deviation']:.2f}", axis=1
    )
    return anomalies


def save_anomalies_to_csv():
    anomalies_df = detect_anomalies_stl()
    anomalies_df.to_csv("anomalies_detected_stl.csv", index=False)
    print("Anomalies saved to anomalies_detected_stl.csv")

save_anomalies_to_csv()

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Anomaly Detection using STL"),
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
    df_anomalies = detect_anomalies_stl()
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
    
    fig.update_layout(title="Anomaly Detection using STL",
                      xaxis_title="Timestamp", yaxis_title="Power Load",
                      xaxis=dict(tickangle=45), template="plotly_white")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
