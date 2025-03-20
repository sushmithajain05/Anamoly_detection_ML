import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output

SI_UNITS = {"power_load": "kW"}
file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

ALGORITHMS = ["Isolation Forest", "Fourier Transform", "ESD", "STL", "Dynamic Threshold", "Z-score", "SR-CNN"]

window_size = 100
df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
df["iso_anomaly_score"] = iso_forest.fit_predict(df[["power_load"]])
df["iso_anomaly"] = df["iso_anomaly_score"] == -1

def detect_anomalies(method):
    df_anomalies = df.copy()
    if method == "Isolation Forest":
        df_anomalies = df[df["iso_anomaly"]]
    elif method == "Fourier Transform":
        pass  
    elif method == "ESD":
        pass 
    elif method == "STL":
        pass 
    elif method == "Dynamic Threshold":
        pass 
    elif method == "Z-score":
        df["z_score"] = (df["power_load"] - df["expected_value"]) / df["std_dev"]
        df_anomalies = df[np.abs(df["z_score"]) > 2]
    elif method == "SR-CNN":
        pass
    df_anomalies["hover_text"] = df_anomalies.apply(
        lambda row: f"Date & Time: {row['timestamp']}<br>"
                    f"Feature: Power Load ({SI_UNITS['power_load']})<br>"
                    f"Expected Value: {row['expected_value']:.2f} {SI_UNITS['power_load']}<br>"
                    f"Actual Value: {row['power_load']:.2f} {SI_UNITS['power_load']}<br>", axis=1
    )
    return df_anomalies

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Anomaly Detection on Power Load"),
    dcc.Dropdown(
        id='algorithm-selector',
        options=[{'label': algo, 'value': algo} for algo in ALGORITHMS],
        value='Isolation Forest',
        clearable=False
    ),
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
    Input('date-picker', 'end_date'),
    Input('algorithm-selector', 'value')
)
def update_graph(start_date, end_date, algorithm):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    filtered_anomalies = detect_anomalies(algorithm)
    filtered_anomalies = filtered_anomalies[(filtered_anomalies['timestamp'] >= start_date) & (filtered_anomalies['timestamp'] <= end_date)]
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