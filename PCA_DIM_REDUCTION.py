import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "EB_2021.csv"  # Update path if needed
data = pd.read_csv(file_path)

# Selecting numerical features for PCA
numerical_features = [
    'energy_solar', 'energy_gridgenset', 'energy_load', 'energy_batterycharge', 
    'energy_batterydischarge', 'battery_power', 'battery_voltage', 'battery_current', 
    'battery_soc', 'battery_temperature', 'power_solar', 'power_gridgenset', 'power_load'
]

# Handle missing values by filling with column mean
data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[numerical_features])

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95)
pca_transformed = pca.fit_transform(scaled_data)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# Number of selected components
print(f"Optimal number of components: {pca.n_components_}")

# Convert PCA output to DataFrame
pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Save the transformed data
pca_df.to_csv("EB_2021_PCA_reduced.csv", index=False)
