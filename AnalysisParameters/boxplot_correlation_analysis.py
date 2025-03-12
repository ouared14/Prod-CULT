import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'data.csv' with actual file path)
data = pd.read_csv("data.csv")

# Define the relevant columns for analysis
parameters = ["Soil Fertility", "Crop Density", "Temperature", "Humidity", 
              "Soil Type", "Irrigation Efficiency", "Machinery Energy Consumption"]

# Create Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[parameters])
plt.xticks(rotation=45)
plt.title("Boxplot of Agricultural Parameters")
plt.xlabel("Parameters")
plt.ylabel("Values")
plt.show()

# Compute and plot the correlation matrix
plt.figure(figsize=(8, 6))
corr_matrix = data[parameters].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Parameters")
plt.show()
