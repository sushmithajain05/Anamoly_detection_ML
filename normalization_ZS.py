import pandas as pd
from sklearn.preprocessing import StandardScaler
file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

numerical_cols = df.select_dtypes(include=['number']).columns

scaler = StandardScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

output_file = "EB_2021_standardized.csv"
df.to_csv(output_file, index=False)

print(f"Standardized data saved to {output_file}")
