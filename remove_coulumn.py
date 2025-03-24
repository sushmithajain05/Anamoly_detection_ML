import pandas as pd

file_path = "anomalies_detected_3.csv"
df = pd.read_csv(file_path)

columns_to_remove = [
    "energy_solar_expected", "energy_solar_std_dev", "energy_solar_upper_boundary", "energy_solar_lower_boundary",
    "energy_gridgenset_expected", "energy_gridgenset_std_dev", "energy_gridgenset_upper_boundary", "energy_gridgenset_lower_boundary",
    "energy_load_expected", "energy_load_std_dev", "energy_load_upper_boundary", "energy_load_lower_boundary",
    "energy_batterycharge_expected", "energy_batterycharge_std_dev", "energy_batterycharge_upper_boundary", "energy_batterycharge_lower_boundary",
    "energy_batterydischarge_expected", "energy_batterydischarge_std_dev", "energy_batterydischarge_upper_boundary", "energy_batterydischarge_lower_boundary",
    "battery_power_expected", "battery_power_std_dev", "battery_power_upper_boundary", "battery_power_lower_boundary",
    "battery_voltage_expected", "battery_voltage_std_dev", "battery_voltage_upper_boundary", "battery_voltage_lower_boundary",
    "battery_current_expected", "battery_current_std_dev", "battery_current_upper_boundary", "battery_current_lower_boundary",
    "battery_soc_expected", "battery_soc_std_dev", "battery_soc_upper_boundary", "battery_soc_lower_boundary",
    "battery_temperature_expected", "battery_temperature_std_dev", "battery_temperature_upper_boundary", "battery_temperature_lower_boundary",
    "power_solar_expected", "power_solar_std_dev", "power_solar_upper_boundary", "power_solar_lower_boundary",
    "power_gridgenset_expected", "power_gridgenset_std_dev", "power_gridgenset_upper_boundary", "power_gridgenset_lower_boundary",
    "power_load_expected", "power_load_std_dev", "power_load_upper_boundary", "power_load_lower_boundary",
    "energy_solar_anomaly_score", "deviation", "hover_text", "feature", "energy_gridgenset_anomaly_score", "energy_load_anomaly_score",
    "energy_batterycharge_anomaly_score", "energy_batterydischarge_anomaly_score", "battery_power_anomaly_score", "battery_voltage_anomaly_score",
    "battery_current_anomaly_score", "battery_soc_anomaly_score", "battery_temperature_anomaly_score", "power_solar_anomaly_score",
    "power_gridgenset_anomaly_score", "power_load_anomaly_score"
]

df = df.drop(columns=columns_to_remove, errors='ignore')
df.to_csv("cleaned_3.csv", index=False)
print(df)
print(len(df))
print("Columns removed and cleaned dataset saved.")
