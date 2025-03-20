import pandas as pd
import altair as alt
from sklearn.ensemble import IsolationForest
import altair_viewer  

df = pd.read_csv('elmeasure_energy_meter.csv')
df['timestamp'] = pd.to_datetime(df['timestamp']) 

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'kwh_imp']
X = df[features]

iso_forest = IsolationForest(random_state=42)
y_pred = iso_forest.fit_predict(X)
df['anomaly'] = y_pred

anomalies = df[df['anomaly'] == -1]
anomalies.to_csv('anomalies.csv', index=False)  

alt.data_transformers.disable_max_rows()

chart = alt.Chart(anomalies).mark_circle(size=60, color='red').encode(
    x='timestamp:T',
    y='watts_total:Q',
    tooltip=['timestamp', 'watts_total']
).properties(
    title='Anomaly Detection using Isolation Forest (Watts Total Over Time)',
    width=800,
    height=400
)
line_chart = alt.Chart(anomalies).mark_line(color='blue', size=2).encode(
    x='timestamp:T',
    y='watts_total:Q'
)
interactive_chart = chart + line_chart
interactive_chart = interactive_chart.interactive()
altair_viewer.show(interactive_chart)





