import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

features = [
    "energy_solar", "energy_load", "energy_batterycharge", 
    "energy_batterydischarge", "battery_voltage",  
    "battery_soc", "battery_temperature", "power_solar",  "power_load"
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
    if feature in df.columns:
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

final_anomalies_df = pd.concat(anomaly_list, ignore_index=True) if anomaly_list else pd.DataFrame()
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
    print(f"Selected feature: {selected_feature}")

    if start_date is None or end_date is None:
        print("Invalid date range selected.")
        return go.Figure()

    start_date = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)

    if selected_feature not in df.columns:
        print(f"Feature '{selected_feature}' not found in dataset.")
        return go.Figure()

    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    if filtered_df.empty:
        print("No data available in selected range")
        return go.Figure()

    if selected_feature in df_anomalies:
        filtered_anomalies = df_anomalies[selected_feature].copy()
        filtered_anomalies = filtered_anomalies[
            (filtered_anomalies['timestamp'] >= start_date) &
            (filtered_anomalies['timestamp'] <= end_date)
        ]
    else:
        print("No anomalies found for this feature.")
        filtered_anomalies = pd.DataFrame(columns=["timestamp", selected_feature])

    if f"{selected_feature}_upper_boundary" not in filtered_df.columns:
        print(f"Missing boundary columns for {selected_feature}")
        return go.Figure()

    df_sampled = filtered_df.iloc[::10]  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature], mode="lines", name="Actual Value", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_expected"], mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
                             y=df_sampled[f"{selected_feature}_upper_boundary"].tolist() + df_sampled[f"{selected_feature}_lower_boundary"].tolist()[::-1],
                             fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                             line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
    if not filtered_anomalies.empty:
        fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies[selected_feature],
                                 mode="markers", name="Anomaly", marker=dict(color="red", size=6),
                                 text=filtered_anomalies["hover_text"], hoverinfo="text"))

    fig.update_layout(title=f"Anomaly Detection on {selected_feature}", xaxis_title="Timestamp", yaxis_title=selected_feature, xaxis=dict(tickangle=45), template="plotly_white")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)