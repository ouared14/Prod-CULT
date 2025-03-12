import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Génération de données aléatoires pour représenter les paramètres de production
np.random.seed(42)
N = 200  # Nombre d'échantillons
n_features = 30  # Nombre de paramètres initiaux

X = np.random.rand(N, n_features)
y = np.random.rand(N)  # Variable cible (ex: rendement ou coût de production)

# Nommage des colonnes
feature_names = [f'Param_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sélection des caractéristiques avec un modèle Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Extraction de l'importance des caractéristiques
feature_importances = rf_model.feature_importances_
sorted_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

# Mise en DataFrame pour analyse
importance_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])

# Réduction de dimension avec PCA
pca = PCA(n_components=17)  # Conservation des 17 paramètres les plus influents
X_pca = pca.fit_transform(X_scaled)

# Calcul de la variance expliquée cumulée
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Affichage des résultats
plt.figure(figsize=(12, 5))

# Graphique 1 : Importance des caractéristiques
plt.subplot(1, 2, 1)
plt.barh(importance_df["Feature"][:17], importance_df["Importance"][:17], color="skyblue")
plt.xlabel("Importance normalisée")
plt.ylabel("Paramètres")
plt.title("Importance des paramètres (Top 17)")

# Graphique 2 : Importance cumulée
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="-", color="red")
plt.axhline(y=0.95, color="gray", linestyle="--", label="Seuil 95%")
plt.xlabel("Nombre de paramètres")
plt.ylabel("Importance cumulative")
plt.title("Importance cumulée des paramètres")
plt.legend()

plt.tight_layout()
plt.show()

# Sauvegarde des résultats en CSV
importance_df.to_csv("feature_importance.csv", index=False)
pd.DataFrame(X_pca).to_csv("reduced_features.csv", index=False)
