
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("EB_2021.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')

features = [
    'energy_solar', 'energy_gridgenset', 'energy_load', 'energy_batterycharge', 
    'energy_batterydischarge', 'battery_power', 'battery_voltage', 'battery_current', 
    'battery_soc', 'battery_temperature', 'power_solar', 'power_gridgenset', 'power_load'
]
si_units = {
    'energy_solar': 'kWh',
    'energy_gridgenset': 'kWh',
    'energy_load': 'kWh',
    'energy_batterycharge': 'kWh',
    'energy_batterydischarge': 'kWh',
    'battery_power': 'kW',
    'battery_voltage': 'V',
    'battery_current': 'A',
    'battery_soc': '%',
    'battery_temperature': '°C',
    'power_solar': 'kW',
    'power_gridgenset': 'kW',
    'power_load': 'kW'
}


scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data[features])
data['anomaly_score'] = iso_forest.decision_function(data[features])


anomalies = data[data['anomaly'] == -1].copy()

for feature in features:
    anomalies[f'expected_{feature}'] = anomalies[feature].rolling(window=10, min_periods=1).mean()


def get_anomaly_reason(row):
    reasons = []
    for feature in features:
        mean_value = data[feature].mean()
        std_dev = data[feature].std()
        if abs(row[feature] - mean_value) > 2 * std_dev:
            reasons.append(f"{feature} deviation: {row[feature]:.2f} {si_units[feature]}")
    return ", ".join(reasons) if reasons else "Unknown"

anomalies['reason'] = anomalies.apply(get_anomaly_reason, axis=1)

anomalies[features] = scaler.inverse_transform(anomalies[features])
anomalies[[f'expected_{feature}' for feature in features]] = scaler.inverse_transform(anomalies[[f'expected_{feature}' for feature in features]])

anomalies.to_csv("anomalies_detected1.csv", index=False)

for feature in features:
    feature_with_unit = f"{feature} ({si_units.get(feature, '')})"

    fig = px.area(
        anomalies, x='timestamp', y=feature, color='anomaly',
        title=f'Anomaly Detection in {feature_with_unit}',
        labels={'timestamp': 'Timestamp', feature: feature_with_unit},
        template='plotly_white',
        hover_data={'anomaly_score': True, 'reason': True, 'expected_' + feature: ':.2f', feature: ':.2f'}
    )
    
    fig.update_traces(fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=2))

    fig.add_trace(go.Scatter(
        x=anomalies['timestamp'], 
        y=anomalies[f'expected_{feature}'], 
        mode='lines+markers',  
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=6, symbol='circle'),
        name=f"Expected {feature_with_unit}"
    ))
    
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
