import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import lightgbm as lgb

# 1. Charger les données
df = pd.read_csv("data/processed/dataset_modelisation.csv")
df = df.dropna(subset=["rendement_t_ha"])
df_passe = df[df["annee"] <= 2024]

features_num = [
    "pluie_cumul_mm", "pause_pluvio_max_j", "temp_max_moy_c", 
    "ndvi_pic", "date_pic", "pente_croissance", "ndvi_auc", "engrais_kg_ha"
]
features_cat = ["culture", "type_sol", "semence_certifiee", "departement"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])

def creer_modele(alpha):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            objective='quantile', alpha=alpha,
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1
        ))
    ])

X_passe = df_passe[features_num + features_cat]
y_passe = df_passe["rendement_t_ha"]

# On recalcule la marge de conformité (comme dans calibrate.py)
print("Calcul de la marge de conformité...")
scores_erreurs = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, calib_idx in kf.split(X_passe):
    X_tr, y_tr = X_passe.iloc[train_idx], y_passe.iloc[train_idx]
    X_ca, y_ca = X_passe.iloc[calib_idx], y_passe.iloc[calib_idx]
    m_bas = creer_modele(0.1).fit(X_tr, y_tr)
    m_haut = creer_modele(0.9).fit(X_tr, y_tr)
    pred_b = m_bas.predict(X_ca)
    pred_h = m_haut.predict(X_ca)
    ecart = np.maximum(pred_b - y_ca, y_ca - pred_h)
    scores_erreurs.extend(ecart)

marge = np.quantile(scores_erreurs, 0.80)
print(f"Marge calculée : {marge:.3f}")

# On entraîne les 3 modèles finaux sur TOUTES les données passées (22-24)
print("Entraînement des modèles finaux...")
modele_bas = creer_modele(0.1).fit(X_passe, y_passe)
modele_med = creer_modele(0.5).fit(X_passe, y_passe)
modele_haut = creer_modele(0.9).fit(X_passe, y_passe)

# 2. Sauvegarde des modèles sur le disque
os.makedirs("models", exist_ok=True)
joblib.dump(modele_bas, "models/modele_bas.pkl")
joblib.dump(modele_med, "models/modele_med.pkl")
joblib.dump(modele_haut, "models/modele_haut.pkl")
joblib.dump(marge, "models/marge_conformite.pkl")

print("Modeles sauvegardes avec succes dans le dossier /models !")