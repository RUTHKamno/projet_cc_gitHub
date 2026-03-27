"""
Script d'exécution du notebook app.ipynb et sauvegarde du modèle.
Reproduit toutes les étapes du notebook.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ─────────────────────────────────────────
# 1. Chargement du dataset
# ─────────────────────────────────────────
print("📂 Chargement du dataset...")
df = pd.read_csv("valeurs_immobilieres_local.csv")
print(f"   Dataset chargé : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
print(f"   Colonnes : {list(df.columns)}")

# ─────────────────────────────────────────
# 2. Prétraitement
# ─────────────────────────────────────────
print("\n⚙️  Prétraitement...")
X = df.drop("prix_vente", axis=1)
y = df["prix_vente"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"   Features numériques  : {numeric_features}")
print(f"   Features catégorielles : {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Division 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Taille train : {len(X_train)} | Taille test : {len(X_test)}")

# ─────────────────────────────────────────
# 3. Entraînement du modèle
# ─────────────────────────────────────────
print("\n🤖 Entraînement du modèle RandomForestRegressor...")
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
])
model.fit(X_train, y_train)
print("   ✅ Modèle entraîné avec succès.")

# ─────────────────────────────────────────
# 4. Évaluation
# ─────────────────────────────────────────
print("\n📊 Évaluation des performances...")
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"   MAE  (Erreur Moyenne Absolue)              : {mae:.2f}")
print(f"   RMSE (Racine de l'erreur quadratique moy.) : {rmse:.2f}")
print(f"   Score R² (Précision)                       : {r2:.4f}")

# ─────────────────────────────────────────
# 5. Sauvegarde du modèle dans models/
# ─────────────────────────────────────────
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "random_forest_immobilier.joblib")
joblib.dump(model, model_path)
print(f"\n💾 Modèle sauvegardé → {model_path}")

# Sauvegarde d'un fichier de métriques
metrics_path = os.path.join(models_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("=== Métriques du modèle RandomForestRegressor ===\n\n")
    f.write(f"MAE  : {mae:.2f}\n")
    f.write(f"RMSE : {rmse:.2f}\n")
    f.write(f"R²   : {r2:.4f}\n")
    f.write(f"\nDataset : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
    f.write(f"Train   : {len(X_train)} exemples\n")
    f.write(f"Test    : {len(X_test)} exemples\n")
print(f"📄 Métriques sauvegardées → {metrics_path}")
print("\n✅ Exécution complète !")
