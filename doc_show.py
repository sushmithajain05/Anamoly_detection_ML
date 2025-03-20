import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv("EB_2021.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')

# Feature selection and unit definitions
features = [
    'energy_solar', 'energy_gridgenset', 'energy_load', 'energy_batterycharge', 
    'energy_batterydischarge', 'battery_power', 'battery_voltage', 'battery_current', 
    'battery_soc', 'battery_temperature', 'power_solar', 'power_gridgenset', 'power_load'
]

si_units = {
    'energy_solar': 'kWh', 'energy_gridgenset': 'kWh', 'energy_load': 'kWh',
    'energy_batterycharge': 'kWh', 'energy_batterydischarge': 'kWh',
    'battery_power': 'kW', 'battery_voltage': 'V', 'battery_current': 'A',
    'battery_soc': '%', 'battery_temperature': '°C', 'power_solar': 'kW',
    'power_gridgenset': 'kW', 'power_load': 'kW'
}

# Normalize data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data[features])

# Extract only anomaly points
anomalies = data[data['anomaly'] == -1].copy()

# Inverse scaling for visualization
anomalies[features] = scaler.inverse_transform(anomalies[features])

# Save anomalies to CSV
anomalies.to_csv("anomalies_detected.csv", index=False)

# Plot anomalies
for feature in features:
    feature_with_unit = f"{feature} ({si_units.get(feature, '')})"

    fig = px.scatter(
        anomalies, x='timestamp', y=feature, color='anomaly',
        title=f'Anomaly Detection in {feature_with_unit}',
        labels={'timestamp': 'Timestamp', feature: feature_with_unit},
        template='plotly_white'
    )
    
    fig.update_traces(marker=dict(color='red', size=6, symbol='circle'))

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title=feature_with_unit, 
        xaxis={'rangeslider': {'visible': True}, 'rangeselector': {'buttons': [
            {'count': 1, 'label': '1d', 'step': 'day', 'stepmode': 'backward'},
            {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
            {'step': 'all'}
        ]}}
    )
    
    fig.show()
