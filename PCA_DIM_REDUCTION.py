# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler

# file_path = "EB_2021.csv" 
# data = pd.read_csv(file_path)

# numerical_features = [
#     'energy_solar', 'energy_gridgenset', 'energy_load', 'energy_batterycharge', 
#     'energy_batterydischarge', 'battery_power', 'battery_voltage', 'battery_current', 
#     'battery_soc', 'battery_temperature', 'power_solar', 'power_gridgenset', 'power_load'
# ]

# data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data[numerical_features])

# pca = PCA(n_components=0.95)
# pca_transformed = pca.fit_transform(scaled_data)

# plt.figure(figsize=(8, 5))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA Explained Variance')
# plt.grid(True)
# plt.show()

# print(f"Optimal number of components: {pca.n_components_}")

# pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
# pca_df.to_csv("EB_2021_PCA_reduced.csv", index=False)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

file_path = "EB_2021.csv" 
data = pd.read_csv(file_path)

numerical_features = [
    'energy_solar', 'energy_gridgenset', 'energy_load', 'energy_batterycharge', 
    'energy_batterydischarge', 'battery_power', 'battery_voltage', 'battery_current', 
    'battery_soc', 'battery_temperature', 'power_solar', 'power_gridgenset', 'power_load'
]

data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[numerical_features])

pca = PCA(n_components=0.95)
pca_transformed = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

print(f"Optimal number of components: {pca.n_components_}")

pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
pca_df.to_csv("EB_2021_PCA_reduced.csv", index=False)

loadings = pd.DataFrame(pca.components_, columns=numerical_features, index=[f'PC{i+1}' for i in range(pca.n_components_)])
print("PCA Component Loadings:")
print(loadings)

plt.figure(figsize=(10, 5))
plt.bar(numerical_features, pca.components_[0], color='b', alpha=0.7)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("PCA Component Loading (PC1)")
plt.title("Feature Importance in First Principal Component")
plt.grid()
plt.show()
