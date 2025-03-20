import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Read and prepare data
data = pd.read_csv("anomalies_detected.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')

# Features and their respective SI units
features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'kwh_imp']
si_units = {
    'watts_total': 'W',  
    'pf_avg': '',  
    'vln_avg': 'V',  
    'amps_avg': 'A',  
    'kwh_imp': 'kWh'  
}

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
data['anomaly'] = iso_forest.fit_predict(data[features])
data['anomaly_score'] = iso_forest.decision_function(data[features])

# Separate anomalies and normal data
anomalies = data[data['anomaly'] == -1].copy()
normal_data = data[data['anomaly'] == 1].copy()

# Calculate expected values for normal data
for feature in features:
    normal_data.loc[:, f'expected_{feature}'] = normal_data[feature].rolling(window=10, min_periods=1).mean()

# Function to generate anomaly reason based on feature deviation
def get_anomaly_reason(row):
    reasons = []
    for feature in features:
        mean_value = data[feature].mean()
        std_dev = data[feature].std()
        if abs(row[feature] - mean_value) > 2 * std_dev:
            reasons.append(f"{feature} deviation: {row[feature]:.2f} {si_units[feature]}")
    return ", ".join(reasons) if reasons else "Unknown"

# Assign reason for each anomaly
anomalies.loc[:, 'reason'] = anomalies.apply(get_anomaly_reason, axis=1)

# Save the anomalies to CSV
anomalies.to_csv("anomalies_detected.csv", index=False)

# Function to break down anomalies into smaller groups (branches of 20 anomalies)
def split_into_branches(anomalies, feature, branch_size=20):
    # Split the anomalies data into smaller chunks (branches) for better readability
    return [anomalies.iloc[i:i + branch_size] for i in range(0, anomalies.shape[0], branch_size)]

# Create plots for each feature
for feature in features:
    # Split anomalies into smaller branches
    feature_anomalies = anomalies[anomalies[feature].notna()]
    branches = split_into_branches(feature_anomalies, feature, branch_size=20)
    
    # Create the plot for each branch
    for i, branch in enumerate(branches):
        fig = px.area(
            branch, x='timestamp', y=feature, color='anomaly',
            title=f'Anomaly Detection in {feature} ({si_units[feature]}) - Branch {i+1}',
            labels={feature: f'{feature} ({si_units[feature]})', 'timestamp': 'Timestamp'},
            template='plotly_white',
            hover_data={
                'anomaly_score': True, 
                'reason': True,
                'device_id': True,
                'expected_' + feature: ':.2f',
                feature: ':.2f',
            }
        )

        # Highlight anomalies with a red fill
        fig.update_traces(fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=2))

        # Add the expected trend line for normal data
        fig.add_scatter(
            x=normal_data['timestamp'], 
            y=normal_data[f'expected_{feature}'], 
            mode='lines', 
            line=dict(color='blue', dash='dash'),
            name="Expected Trend",
            hovertext=[f"Device: {device} | Expected {feature}: {val:.2f} {si_units[feature]}" for device, val in zip(normal_data['device_id'], normal_data[f'expected_{feature}'])]
        )

        # Update the layout with rangeslider and selector
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
        
        # Show the plot for this branch
        fig.show()
