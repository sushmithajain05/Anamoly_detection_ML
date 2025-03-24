import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

features = [
    "energy_solar", "energy_gridgenset", "energy_load", "energy_batterycharge", 
    "energy_batterydischarge", "battery_power", "battery_voltage", "battery_current", 
    "battery_soc", "battery_temperature", "power_solar", "power_gridgenset", "power_load"
]

window_size = 500
for feature in features:
    df[f"{feature}_expected"] = df[feature].rolling(window=window_size, center=True, min_periods=1).mean()
    df[f"{feature}_std_dev"] = df[feature].rolling(window=window_size, center=True, min_periods=1).std()
    df[f"{feature}_upper_boundary"] = df[f"{feature}_expected"] + (2 * df[f"{feature}_std_dev"])
    df[f"{feature}_lower_boundary"] = df[f"{feature}_expected"] - (2 * df[f"{feature}_std_dev"])

model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
df_anomalies = {}
anomaly_list = []

for feature in features:
    df[f"{feature}_anomaly_score"] = model.fit_predict(df[[feature]])
    anomaly_df = df[df[f"{feature}_anomaly_score"] == -1].copy()
    anomaly_df["deviation"] = anomaly_df[feature] - anomaly_df[f"{feature}_expected"]
    
    anomaly_df["hover_text"] = anomaly_df.apply(
        lambda row: f"Date & Time: {row['timestamp']}<br>"
                    f"Feature: {feature}<br>"
                    f"Expected Value: {row[f'{feature}_expected']:.2f}<br>"
                    f"Actual Value: {row[feature]:.2f}<br>"
                    f"Deviation: {row['deviation']:.2f}", axis=1
    )
    
    df_anomalies[feature] = anomaly_df
    anomaly_df["feature"] = feature
    anomaly_list.append(anomaly_df)

final_anomalies_df = pd.concat(anomaly_list, ignore_index=True)
final_anomalies_df = final_anomalies_df.drop_duplicates(subset=["timestamp"])  # Remove duplicate anomalies
final_anomalies_df.to_csv("anomalies_detected_3.csv", index=False)
print("Anomalies saved to anomalies_detected_3.csv")

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Anomaly Detection on Multiple Features"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in features],
        value=features[0]
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
    Input('feature-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_graph(selected_feature, start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    filtered_anomalies = df_anomalies[selected_feature][
        (df_anomalies[selected_feature]['timestamp'] >= start_date) & 
        (df_anomalies[selected_feature]['timestamp'] <= end_date)
    ]
    df_sampled = filtered_df.iloc[::40]  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature],
                             mode="lines", name="Actual Value", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_expected"],
                             mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_upper_boundary"],
                             mode="lines", name="Upper Boundary", line=dict(color="orange", dash="dot")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_lower_boundary"],
                             mode="lines", name="Lower Boundary", line=dict(color="purple", dash="dot")))
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies[selected_feature],
                         mode="markers", name="Anomaly", marker=dict(color="red", size=6),
                         text=filtered_anomalies["hover_text"], hoverinfo="text"))

    
    fig.update_layout(title=f"Anomaly Detection on {selected_feature}",
                      xaxis_title="Timestamp", yaxis_title=selected_feature,
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run(debug=True)