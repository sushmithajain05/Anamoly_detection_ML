# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output
# from scipy.stats import t

# file_path = "EB_2021.csv"
# df = pd.read_csv(file_path)
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")
# df = df.dropna(subset=["power_load"])

# window_size = 50
# df["expected_value"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).mean()
# df["std_dev"] = df["power_load"].rolling(window=window_size, center=True, min_periods=1).std()
# df["upper_boundary"] = df["expected_value"] + (2 * df["std_dev"])
# df["lower_boundary"] = df["expected_value"] - (2 * df["std_dev"])

# def detect_anomalies_esd(max_outliers=100, alpha=0.05):
#     series = df["power_load"].copy()
#     anomalies = []
#     n = len(series)

#     for i in range(max_outliers):
#         mean = series.mean()
#         std_dev = series.std()
#         abs_deviation = np.abs(series - mean)
#         max_deviation = abs_deviation.max()
#         max_index = abs_deviation.idxmax()

#         test_statistic = max_deviation / std_dev
#         critical_value = ((n - i - 1) * t.ppf(1 - alpha / (2 * (n - i)), n - i - 2)) / np.sqrt(
#             (n - i - 2) + t.ppf(1 - alpha / (2 * (n - i)), n - i - 2) ** 2
#         )

#         if test_statistic > critical_value:
#             anomalies.append(max_index)
#             series = series.drop(max_index)  # Fix: Drop without modifying original
#         else:
#             break

#     df_anomalies = df.loc[anomalies].copy() if anomalies else pd.DataFrame(columns=["timestamp", "power_load", "expected_value", "deviation", "hover_text"])
#     if not df_anomalies.empty:
#         df_anomalies["deviation"] = df_anomalies["power_load"] - df_anomalies["expected_value"]
#         df_anomalies["hover_text"] = df_anomalies.apply(
#             lambda row: f"Date & Time: {row['timestamp']}<br>"
#                         f"Feature: Power Load<br>"
#                         f"Expected Value: {row['expected_value']:.2f}<br>"
#                         f"Actual Value: {row['power_load']:.2f}<br>"
#                         f"Deviation: {row['deviation']:.2f}", axis=1
#         )
    
#     print(f"Number of detected anomalies: {len(df_anomalies)}")  # Debugging output
#     return df_anomalies

# df_anomalies_global = detect_anomalies_esd()

# def save_anomalies_to_csv():
#     df_anomalies_global.to_csv("anomalies_detected_esd.csv", index=False)
#     print("Anomalies saved to anomalies_detected_esd.csv")

# save_anomalies_to_csv()

# app = Dash(__name__)
# app.layout = html.Div([
#     html.H1("Anomaly Detection using ESD"),
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
#     Input('date-picker', 'start_date'),
#     Input('date-picker', 'end_date')
# )
# def update_graph(start_date, end_date):
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)
    
#     filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
#     filtered_anomalies = df_anomalies_global[(df_anomalies_global['timestamp'] >= start_date) & (df_anomalies_global['timestamp'] <= end_date)]
#     df_sampled = filtered_df.iloc[::40]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled["power_load"],
#                              mode="lines", name="Actual Value", line=dict(color="blue")))
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"], y=df_sampled["expected_value"],
#                              mode="lines", name="Expected Value", line=dict(color="green", dash="dash")))
#     fig.add_trace(go.Scatter(x=df_sampled["timestamp"].tolist() + df_sampled["timestamp"].tolist()[::-1],
#                              y=df_sampled["upper_boundary"].tolist() + df_sampled["lower_boundary"].tolist()[::-1],
#                              fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
#                              line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Boundary"))
#     fig.add_trace(go.Scatter(x=filtered_anomalies["timestamp"], y=filtered_anomalies["power_load"],
#                              mode="markers", name="Anomaly", marker=dict(color="red", size=6),
#                              text=filtered_anomalies["hover_text"], hoverinfo="text"))
    
#     fig.update_layout(title="Anomaly Detection using ESD",
#                       xaxis_title="Timestamp", yaxis_title="Power Load",
#                       xaxis=dict(tickangle=45), template="plotly_white")
#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=True)