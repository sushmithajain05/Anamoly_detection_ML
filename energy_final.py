
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

data = pd.read_csv("elmeasure_energy_meter1.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'kwh_imp']
si_units = {
    'watts_total': 'W',  
    'pf_avg': '',  
    'vln_avg': 'V',  
    'amps_avg': 'A',  
    'kwh_imp': 'kWh'  
}

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data[features])
data['anomaly_score'] = iso_forest.decision_function(data[features])

anomalies = data[data['anomaly'] == -1].copy()
normal_data = data[data['anomaly'] == 1].copy()

for feature in features:
    normal_data.loc[:, f'expected_{feature}'] = normal_data[feature].rolling(window=10, min_periods=1).mean()

def get_anomaly_reason(row):
    reasons = []
    for feature in features:
        mean_value = data[feature].mean()
        std_dev = data[feature].std()
        if abs(row[feature] - mean_value) > 2 * std_dev:
            reasons.append(f"{feature} deviation: {row[feature]:.2f} {si_units[feature]}")
    return ", ".join(reasons) if reasons else "Unknown"

anomalies.loc[:, 'reason'] = anomalies.apply(get_anomaly_reason, axis=1)
for feature in features:
    anomalies.loc[:, f'expected_{feature}'] = anomalies[feature].rolling(window=10, min_periods=1).mean()

anomalies.to_csv("anomalies_detected.csv", index=False)

for feature in features:
    fig = px.area(
        anomalies, x='timestamp', y=feature, color='anomaly',
        title=f'Anomaly Detection in {feature} ({si_units[feature]})',
        labels={
            feature: f'{feature} ({si_units[feature]})', 
            'timestamp': 'Timestamp'
        },
        template='plotly_white',
        hover_data={
            'anomaly_score': True, 
            'reason': True,
            'device_id': True,
            'expected_' + feature: ':.2f',
            feature: ':.2f',
        }
    )

    fig.update_traces(fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=2))
    fig.add_scatter(
        x=normal_data['timestamp'], 
        y=normal_data[f'expected_{feature}'], 
        mode='lines', 
        line=dict(color='blue', dash='dash'),
        name="Expected Trend",
        hovertext=[f"Device: {device} | Expected {feature}: {val:.2f} {si_units[feature]}" for device, val in zip(normal_data['device_id'], normal_data[f'expected_{feature}'])]

    )  

    fig.update_layout(
        xaxis={
            'rangeslider': {'visible': True}, 
            'rangeselector': {'buttons': [
                {'count': 1, 'label': '1d', 'step': 'day', 'stepmode': 'backward'},
                {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
                {'step': 'all'}
            ]}
        }
    )
    fig.show()


