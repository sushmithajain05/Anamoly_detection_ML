import pandas as pd
import altair as alt
from sklearn.ensemble import IsolationForest
import altair_viewer  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('elmeasure_energy_meter.csv')
df['timestamp'] = pd.to_datetime(df['timestamp']) 

features = ['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'kwh_imp']
X = df[features]

iso_forest = IsolationForest(contamination=0.10, random_state=42)
y_pred = iso_forest.fit_predict(X)
df['anomaly'] = y_pred

df['reason'] = "Normal"
df.loc[(df['watts_total'].pct_change().abs() > 0.3) & (df['anomaly'] == -1), 'reason'] = "Sudden spike in power usage"
df.loc[(df['pf_avg'] < 0.7) & (df['anomaly'] == -1), 'reason'] = "Low power factor detected"
df.loc[(df['amps_avg'].pct_change().abs() > 0.3) & (df['anomaly'] == -1), 'reason'] = "Unusual current fluctuation"

df.loc[(df['watts_total'].pct_change().abs() > 0.3) & (df['anomaly'] == -1), 'anomaly_feature'] = 'watts_total'
df.loc[(df['pf_avg'] < 0.7) & (df['anomaly'] == -1), 'anomaly_feature'] = 'pf_avg'
df.loc[(df['amps_avg'].pct_change().abs() > 0.3) & (df['anomaly'] == -1), 'anomaly_feature'] = 'amps_avg'

df['expected_watts'] = df['watts_total'].rolling(window=5, min_periods=1).median()
df['expected_pf'] = df['pf_avg'].rolling(window=5, min_periods=1).median()
df['expected_amps'] = df['amps_avg'].rolling(window=5, min_periods=1).median()

anomalies = df[df['anomaly'] == -1]
print("Anomaly counts by feature:\n", anomalies['anomaly_feature'].value_counts().to_string())

anomalies.to_csv('anomalies1.csv', index=False)

alt.data_transformers.disable_max_rows()

def create_chart(feature, color, expected_feature, title):
    filtered_anomalies = anomalies[anomalies['anomaly_feature'] == feature]
    anomaly_batches = [filtered_anomalies.iloc[i:i+30] for i in range(0, len(filtered_anomalies), 30)]
    
    batch_charts = []
    for i, batch in enumerate(anomaly_batches):
        scatter_chart = alt.Chart(batch).mark_circle(size=40, color='red').encode(
            x='timestamp:T',
            y=f'{feature}:Q',
            tooltip=['timestamp', feature, 'anomaly_feature', 'reason']
        ).properties(
            width=400,  
            height=200,  
            title=f"{title} - Batch {i+1}"
        )
        
        line_chart = alt.Chart(batch).mark_line(color='red', size=1).encode(
            x='timestamp:T',
            y=f'{feature}:Q'
        )
        
        expected_values = batch[['timestamp', expected_feature]].drop_duplicates()

        expected_line = alt.Chart(expected_values).mark_line(color='green', size=2).encode(
            x='timestamp:T',
            y=f'{expected_feature}:Q'
        )

        expected_points = alt.Chart(expected_values).mark_circle(size=60, color='green').encode(
            x='timestamp:T',
            y=f'{expected_feature}:Q',
            tooltip=['timestamp', expected_feature]
        )

        legend = alt.Chart(pd.DataFrame({
            'color': ['red', 'green'],
            'description': ['Anomalies', 'Expected Trend']
        })).mark_point().encode(
            y=alt.Y('description:N', title='Legend', axis=alt.Axis(orient='right')),
            color=alt.Color('color:N', scale=None)
        )

        batch_charts.append((scatter_chart + line_chart + expected_line + expected_points).interactive() | legend)
    
    return alt.vconcat(*batch_charts)


anomaly_batches = [anomalies.iloc[i:i+30] for i in range(0, len(anomalies), 30)]

all_anomalies_charts = []
for i, batch in enumerate(anomaly_batches):
    batch_chart = alt.Chart(batch).mark_circle(size=40, color='red').encode(
        x='timestamp:T',
        y='anomaly_feature:N',
        tooltip=['timestamp', 'anomaly_feature', 'reason']
    ).properties(
        width=600,
        height=300,
        title=f'Anomalies Batch {i+1}'
    )
    
    batch_line = alt.Chart(batch).mark_line(color='red', size=1).encode(
        x='timestamp:T',
        y='anomaly_feature:N'
    )
    
    all_anomalies_charts.append(batch_chart + batch_line)

watts_chart = create_chart('watts_total', 'blue', 'expected_watts', 'Watts Total Anomalies')
pf_chart = create_chart('pf_avg', 'red', 'expected_pf', 'Power Factor Anomalies')
amps_chart = create_chart('amps_avg', 'red', 'expected_amps', 'Current (Amps) Anomalies')

final_chart = alt.vconcat(
    *all_anomalies_charts,
    watts_chart, 
    pf_chart, 
    amps_chart
).properties(
    title="Anomaly Detection in Power Consumption"
)

altair_viewer.show(final_chart)
