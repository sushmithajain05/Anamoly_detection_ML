import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

df = pd.read_csv('elmeasure_energy_meter.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'frequency', 'kwh_imp']
X = df[features]

iso_forest = IsolationForest(contamination=0.002, random_state=42)
y_pred = iso_forest.fit_predict(X)
df['anomaly'] = y_pred

anomalies = df[df['anomaly'] == -1]
anomalies.to_csv('anomalies.csv', index=False)

fig = px.scatter(anomalies, x='timestamp', y='watts_total', 
                 title='Anomaly Detection using Isolation Forest (Watts Total Over Time)', 
                 labels={'timestamp': 'Time', 'watts_total': 'Total Watts'},
                 hover_data=['timestamp', 'watts_total'])

fig.update_traces(marker=dict(size=8, color='red', opacity=0.7))
fig.show()
