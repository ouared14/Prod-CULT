import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Charger les données
df = pd.read_csv("data.csv")

# Lissage des tendances avec une moyenne mobile
df["Yield_Smoothed"] = df["Yield"].rolling(window=5, min_periods=1).mean()

# Gestion des valeurs manquantes par interpolation
df.interpolate(method="linear", inplace=True)

# Extraction de caractéristiques : ajout de la saison de plantation
df["Planting_Season"] = pd.to_datetime(df["Date"]).dt.month
df["Planting_Season"] = df["Planting_Season"].apply(
    lambda x: "Spring" if 3 <= x <= 5 else "Summer" if 6 <= x <= 8 else "Autumn" if 9 <= x <= 11 else "Winter"
)

# Encodage des variables catégorielles
df = pd.get_dummies(df, columns=["Planting_Season"], drop_first=True)

# Normalisation des variables numériques
scaler = RobustScaler()
cols_to_scale = ["Yield", "Soil_Moisture", "Temperature", "Cost"]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[cols_to_scale])
df_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])

# Fusionner avec le jeu de données d'origine
df = pd.concat([df, df_pca], axis=1)

# Sauvegarde du jeu de données transformé
df.to_csv("processed_data.csv", index=False)

print("Prétraitement terminé. Fichier enregistré sous 'processed_data.csv'.")
