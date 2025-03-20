import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from scipy.stats import zscore

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

df = df.set_index("timestamp").resample("1T").mean(numeric_only=True).interpolate().reset_index()


df["z_score"] = zscore(df["power_load"])
df["anomaly"] = (abs(df["z_score"]) > 3).astype(int)  

df_anomalies = df[df["anomaly"] == 1]
df_anomalies["deviation"] = df_anomalies["power_load"] - df_anomalies["power_load"].mean()

def get_anomaly_reason(row):
    if row["z_score"] > 3:
        return "Sudden spike detected."
    elif row["z_score"] < -3:
        return "Sudden drop detected."
    else:
        return "Unusual fluctuation."

df_anomalies["reason"] = df_anomalies.apply(get_anomaly_reason, axis=1)

df_anomalies.to_csv("anomalies_detected_ensemble3.csv", index=False)
print("Anomalies saved to anomalies_detected_ensemble3.csv")


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Anomaly Detection on Power Load (Ensemble 3)"),
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
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies["power_load"],
                             mode="markers", name="Anomaly", marker=dict(color="red", size=6),
                             text=filtered_anomalies["reason"], hoverinfo="text"))
    
    fig.update_layout(title="Anomaly Detection on Power Load (Ensemble 3)",
                      xaxis_title="Timestamp", 
                      yaxis_title="Power Load (kW)",
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
