import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "temperature_data.csv"
df = pd.read_csv(file_path)

# Display basic information
print(df.info())
print(df.describe())

# Handling missing values (e.g., forward fill)
df.fillna(method='ffill', inplace=True)

# Convert time column if exists
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

# Plot temperature trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['temperature'], label="Temperature", color='blue')
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Time Series")
plt.legend()
plt.grid()
plt.show()

# Boxplot for detecting anomalies
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['temperature'])
plt.title("Temperature Distribution (Boxplot)")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['temperature'], bins=30, kde=True)
plt.title("Temperature Distribution (Histogram)")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

# Anomaly detection using 3-sigma rule
mean_temp = df['temperature'].mean()
std_temp = df['temperature'].std()
lower_bound = mean_temp - 3 * std_temp
upper_bound = mean_temp + 3 * std_temp

df['is_anomaly'] = (df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['temperature'], label="Temperature", color='blue')
plt.scatter(df.index[df['is_anomaly']], df['temperature'][df['is_anomaly']], color='red', label="Anomalies")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Anomalies Detection")
plt.legend()
plt.grid()
plt.show()
