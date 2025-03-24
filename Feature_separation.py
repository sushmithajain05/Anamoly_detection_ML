import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output

df = pd.read_csv("EB_2021.csv")
df.columns = df.columns.str.strip()  
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

ALL_FEATURES = list(df.columns)
FEATURE_GROUPS = {
    "Energy Features": [col for col in ["Energy Solar", "Energy GridGenset", "Energy Load", "Energy BatteryCharge", "Energy BatteryDischarge"] if col in ALL_FEATURES],
    "Battery Features": [col for col in ["Battery Power", "Battery Voltage", "Battery Current", "Battery SOC", "Battery Temperature"] if col in ALL_FEATURES],
    "Power Features": [col for col in ["Power Solar", "Power GridGenset", "Power Load"] if col in ALL_FEATURES]
}

SI_UNITS = {
    "Energy Solar": "kWh", "Energy GridGenset": "kWh", "Energy Load": "kWh", 
    "Energy BatteryCharge": "kWh", "Energy BatteryDischarge": "kWh",
    "Battery Power": "W", "Battery Voltage": "V", "Battery Current": "A", 
    "Battery SOC": "%", "Battery Temperature": "°C",
    "Power Solar": "kW", "Power GridGenset": "kW", "Power Load": "kW"
}

print("Available Features in Dataset:", ALL_FEATURES)
print("Filtered Feature Groups:", FEATURE_GROUPS)

df_anomalies = {}
window_size = 100

for group, features in FEATURE_GROUPS.items():
    for feature in features:
        if feature in df.columns: 
            df[feature + "_expected"] = df[feature].rolling(window=window_size, center=True, min_periods=1).mean()
            df[feature + "_std_dev"] = df[feature].rolling(window=window_size, center=True, min_periods=1).std()
            df[feature + "_upper"] = df[feature + "_expected"] + (2 * df[feature + "_std_dev"])
            df[feature + "_lower"] = df[feature + "_expected"] - (2 * df[feature + "_std_dev"])
            
            model = IsolationForest(random_state=42, n_jobs=-1)
            df[feature + "_anomaly"] = model.fit_predict(df[[feature]])
            anomalies = df[df[feature + "_anomaly"] == -1].copy()
            anomalies["deviation"] = anomalies[feature] - anomalies[feature + "_expected"]
            
            anomalies["hover_text"] = anomalies.apply(
                lambda row: f"Date & Time: {row['timestamp']}<br>"
                            f"Feature: {feature} ({SI_UNITS.get(feature, '')})<br>"
                            f"Expected Value: {row[feature + '_expected']:.2f} {SI_UNITS.get(feature, '')}<br>"
                            f"Actual Value: {row[feature]:.2f} {SI_UNITS.get(feature, '')}<br>"
                            f"Deviation: {row['deviation']:.2f} {SI_UNITS.get(feature, '')}", axis=1
            )
            df_anomalies[feature] = anomalies

if df_anomalies:
    all_anomalies = pd.concat(df_anomalies.values())
    all_anomalies.to_csv("anomalies_detected.csv", index=False)
    print("Anomalies saved to anomalies_detected.csv")
else:
    print("No anomalies detected.")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard"),
    dcc.Dropdown(
        id="feature-group-dropdown",
        options=[{"label": key, "value": key} for key in FEATURE_GROUPS.keys()],
        value=list(FEATURE_GROUPS.keys())[0] if FEATURE_GROUPS else None,
        clearable=False
    ),
    dcc.Dropdown(id="feature-dropdown", clearable=False),
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
    Output("feature-dropdown", "options"),
    Output("feature-dropdown", "value"),
    Input("feature-group-dropdown", "value")
)
def update_feature_dropdown(selected_group):
    valid_features = FEATURE_GROUPS.get(selected_group, [])
    options = [{"label": feature, "value": feature} for feature in valid_features]
    return options, options[0]["value"] if options else None

@app.callback(
    Output("anomaly-graph", "figure"),
    Input("feature-dropdown", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date")
)
def update_graph(selected_feature, start_date, end_date):
    if not selected_feature or selected_feature not in df.columns:
        return go.Figure()
    
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    filtered_anomalies = df_anomalies[selected_feature][(df_anomalies[selected_feature]['timestamp'] >= start_date) &
                                                         (df_anomalies[selected_feature]['timestamp'] <= end_date)]
    df_sampled = filtered_df.iloc[::40]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature],
                             mode="lines", name=f"Actual Value ({SI_UNITS.get(selected_feature, '')})", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature + "_expected"],
                             mode="lines", name=f"Expected Value ({SI_UNITS.get(selected_feature, '')})", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies[selected_feature],
                             mode="markers", name=f"Anomaly ({SI_UNITS.get(selected_feature, '')})", marker=dict(color="red", size=6),
                             text=filtered_anomalies["hover_text"], hoverinfo="text"))
    
    fig.update_layout(title=f"Anomaly Detection on {selected_feature}",
                      xaxis_title="Timestamp", 
                      yaxis_title=f"{selected_feature} ({SI_UNITS.get(selected_feature, '')})",
                      xaxis=dict(tickangle=45), template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)