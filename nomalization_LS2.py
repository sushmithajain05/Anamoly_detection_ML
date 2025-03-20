import pandas as pd
import numpy as np

file_path = "EB_2021.csv"
df = pd.read_csv(file_path)

numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = np.log1p(df[numerical_cols]) 
output_file = "EB_2021_log_scaled.csv"
df.to_csv(output_file, index=False)

print(f"Log-scaled data saved to {output_file}")
