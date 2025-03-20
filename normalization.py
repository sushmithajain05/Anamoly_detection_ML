import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

numerical_cols = df.select_dtypes(include=['number']).columns

scaler = MinMaxScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
output_file = "EB_2021_normalized.csv"
df.to_csv(output_file, index=False)

print(f"Normalized data saved to {output_file}")
