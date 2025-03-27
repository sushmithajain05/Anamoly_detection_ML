# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from sklearn.ensemble import IsolationForest
# import dash
# from dash import dcc, html, Input, Output

# file_path = "EB_2021.csv"
# df = pd.read_csv(file_path)

# df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

# features = [
#     "energy_solar", "energy_load", "energy_batterycharge", 
#     "energy_batterydischarge", "battery_voltage", "battery_current",
#     "battery_soc", "battery_temperature", "power_solar", "power_load"
# ]

# window_size = 500
# for feature in features:
#     df[f"{feature}_expected"] = df[feature].rolling(window=window_size, center=True, min_periods=1).mean()
#     df[f"{feature}_std_dev"] = df[feature].rolling(window=window_size, center=True, min_periods=1).std()
#     df[f"{feature}_upper_boundary"] = df[f"{feature}_expected"] + (2 * df[f"{feature}_std_dev"])
#     df[f"{feature}_lower_boundary"] = df[f"{feature}_expected"] - (2 * df[f"{feature}_std_dev"])

# model = IsolationForest(random_state=42, n_jobs=-1)
# df["anomaly_score"] = model.fit_predict(df[features])

# anomaly_df = df[df["anomaly_score"] == -1].copy()

# # Identify the exact feature causing the anomaly
# anomaly_df["feature"] = anomaly_df.apply(
#     lambda row: ", ".join(
#         [feature for feature in features 
#          if row[feature] > row[f"{feature}_upper_boundary"] or row[feature] < row[f"{feature}_lower_boundary"]]
#     ), axis=1
# )

# anomaly_df.to_csv("anomalies_detected_complete.csv", index=False)
# print(f"Anomalies saved. Unique anomaly count: {len(anomaly_df)}")

# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Anomaly Detection on Multiple Features"),
#     dcc.Dropdown(
#         id='feature-dropdown',
#         options=[{'label': feature, 'value': feature} for feature in features],
#         value=features[0]
#     ),
#     dcc.DatePickerRange(
#         id='date-picker',
#         min_date_allowed=df['timestamp'].min().date(),
#         max_date_allowed=df['timestamp'].max().date(),
#         start_date=df['timestamp'].min().date(),
#         end_date=df['timestamp'].max().date()
#     ),
#     dcc.Graph(id='anomaly-graph')
# ])

# @app.callback(
#     Output('anomaly-graph', 'figure'),
#     Input('feature-dropdown', 'value'),
#     Input('date-picker', 'start_date'),
#     Input('date-picker', 'end_date')
# )
# def update_graph(selected_feature, start_date, end_date):
#     if start_date is None or end_date is None:
#         return go.Figure()

#     start_date = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)

#     filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
#     if filtered_df.empty:
#         return go.Figure()

#     filtered_anomalies = anomaly_df[
#         (anomaly_df['timestamp'] >= start_date) & 
#         (anomaly_df['timestamp'] <= end_date) &
#         (anomaly_df[selected_feature] > anomaly_df[f"{selected_feature}_upper_boundary"]) | 
#         (anomaly_df[selected_feature] < anomaly_df[f"{selected_feature}_lower_boundary"])
#     ]
    
#     df_sampled = filtered_df.iloc[::10]  

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature], 
#                              mode="lines", name="Actual Value", line=dict(color="blue")))
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_expected"], 
#                              mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
#                              y=df_sampled[f"{selected_feature}_upper_boundary"].tolist() + df_sampled[f"{selected_feature}_lower_boundary"].tolist()[::-1],
#                              fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
#                              line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
    
#     if not filtered_anomalies.empty:
#         hover_texts = [
#             f"Date & Time: {row['timestamp']}<br>"
#             f"Feature: {selected_feature}<br>"
#             f"Expected Value: {row[f'{selected_feature}_expected']:.2f} kW<br>"
#             f"Actual Value: {row[selected_feature]:.2f} kW<br>"
#             f"Deviation: {abs(row[selected_feature] - row[f'{selected_feature}_expected']):.2f} kW<br>"
#             f"Reason: {'High Deviation' if row[selected_feature] > row[f'{selected_feature}_upper_boundary'] else 'Low Deviation'}"
#             for _, row in filtered_anomalies.iterrows()
#         ]

#         fig.add_trace(go.Scatter(
#             x=filtered_anomalies["timestamp"], y=filtered_anomalies[selected_feature],
#             mode="markers", name="Anomaly", marker=dict(color="red", size=6),
#             text=hover_texts, hoverinfo="text"
#         ))

#     fig.update_layout(title=f"Anomaly Detection on {selected_feature}",
#                       xaxis_title="Timestamp", yaxis_title=selected_feature, 
#                       xaxis=dict(tickangle=45), template="plotly_white")
    
#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=False)


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
    "energy_batterydischarge", "battery_voltage", "battery_current",
    "battery_soc", "battery_temperature", "power_solar", "power_load"
]

window_size = 500
for feature in features:
    df[f"{feature}_expected"] = df[feature].rolling(window=window_size, center=True, min_periods=1).mean()
    df[f"{feature}_std_dev"] = df[feature].rolling(window=window_size, center=True, min_periods=1).std()
    df[f"{feature}_upper_boundary"] = df[f"{feature}_expected"] + (2 * df[f"{feature}_std_dev"])
    df[f"{feature}_lower_boundary"] = df[f"{feature}_expected"] - (2 * df[f"{feature}_std_dev"])

model = IsolationForest(random_state=42, n_jobs=-1)
df["anomaly_score"] = model.fit_predict(df[features])

anomaly_df = df[df["anomaly_score"] == -1].copy()

anomaly_df["feature"] = anomaly_df.apply(
    lambda row: ", ".join(
        [feature for feature in features 
         if row[feature] > row[f"{feature}_upper_boundary"] or row[feature] < row[f"{feature}_lower_boundary"]]
    ) if any(
        row[feature] > row[f"{feature}_upper_boundary"] or row[feature] < row[f"{feature}_lower_boundary"]
        for feature in features
    ) else "No Specific Feature Identified",
    axis=1
)

anomaly_df.to_csv("anomalies_detected_complete.csv", index=False)
print(f"Anomalies saved. Unique anomaly count: {len(anomaly_df)}")

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
    if start_date is None or end_date is None:
        return go.Figure()

    start_date = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)

    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    if filtered_df.empty:
        return go.Figure()

    filtered_anomalies = anomaly_df[
        (anomaly_df['timestamp'] >= start_date) & 
        (anomaly_df['timestamp'] <= end_date) &
        (anomaly_df[selected_feature] > anomaly_df[f"{selected_feature}_upper_boundary"]) | 
        (anomaly_df[selected_feature] < anomaly_df[f"{selected_feature}_lower_boundary"])
    ]
    
    df_sampled = filtered_df.iloc[::10]  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[selected_feature], 
                             mode="lines", name="Actual Value", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled[f"{selected_feature}_expected"], 
                             mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
                             y=df_sampled[f"{selected_feature}_upper_boundary"].tolist() + df_sampled[f"{selected_feature}_lower_boundary"].tolist()[::-1],
                             fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                             line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
    
    if not filtered_anomalies.empty:
        hover_texts = [
            f"Date & Time: {row['timestamp']}<br>"
            f"Feature: {selected_feature}<br>"
            f"Expected Value: {row[f'{selected_feature}_expected']:.2f} kW<br>"
            f"Actual Value: {row[selected_feature]:.2f} kW<br>"
            f"Deviation: {abs(row[selected_feature] - row[f'{selected_feature}_expected']):.2f} kW<br>"
            f"Reason: {'High Deviation' if row[selected_feature] > row[f'{selected_feature}_upper_boundary'] else 'Low Deviation'}"
            for _, row in filtered_anomalies.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=filtered_anomalies["timestamp"], y=filtered_anomalies[selected_feature],
            mode="markers", name="Anomaly", marker=dict(color="red", size=6),
            text=hover_texts, hoverinfo="text"
        ))

    fig.update_layout(title=f"Anomaly Detection on {selected_feature}",
                      xaxis_title="Timestamp", yaxis_title=selected_feature, 
                      xaxis=dict(tickangle=45), template="plotly_white")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
