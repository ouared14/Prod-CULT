import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from CSV (replace 'your_data.csv' with the actual filename)
df = pd.read_csv("your_data.csv")

# Select relevant columns for analysis (update column names based on your dataset)
columns_to_analyze = [
    "soil_fertility", "crop_density", "temperature", "humidity", 
    "soil_type", "irrigation_efficiency", "machinery_energy_consumption", "cxti"
]

# Drop any missing values to avoid issues
df = df[columns_to_analyze].dropna()

# Convert categorical variables (like soil type) into numerical values if needed
if "soil_type" in df.columns:
    df["soil_type"] = df["soil_type"].astype("category").cat.codes

# Plot distributions
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle("Statistical Data Distribution Analysis", fontsize=14)

for i, col in enumerate(columns_to_analyze):
    if i < len(axes.flatten()):
        sns.histplot(df[col], bins=30, kde=True, ax=axes.flatten()[i])
        axes.flatten()[i].set_title(col)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
