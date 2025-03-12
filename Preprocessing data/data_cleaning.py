import pandas as pd
import numpy as np

# Chargement des données
file_path = "data.csv"  # Remplacez par le chemin réel du fichier
df = pd.read_csv(file_path)

# --- Étape 1 : Vue d'ensemble du dataset ---
print("Aperçu des premières lignes :")
print(df.head())

print("\nRésumé des données :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe())

# --- Étape 2 : Nettoyage des données ---

# Suppression des enregistrements avec des valeurs manquantes sur les variables essentielles
variables_essentielles = ["temperature", "rainfall", "irrigation", "crop_yield"]
df = df.dropna(subset=variables_essentielles)

# Identification et suppression des valeurs aberrantes avec l'IQR
def filtrer_outliers(df, colonne):
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    return df[(df[colonne] >= borne_inf) & (df[colonne] <= borne_sup)]

for col in ["crop_yield", "temperature", "rainfall"]:
    df = filtrer_outliers(df, col)

# Sauvegarde du dataset nettoyé
df.to_csv("cleaned_data.csv", index=False)
print("\nNettoyage terminé. Données enregistrées dans 'cleaned_data.csv'.")
