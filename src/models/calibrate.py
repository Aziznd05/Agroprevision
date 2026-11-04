import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import lightgbm as lgb

# 1. Charger les données
df = pd.read_csv("data/processed/dataset_modelisation.csv")
df = df.dropna(subset=["rendement_t_ha"])

# On sépare les données passées (pour l'entraînement et la calibration) et le futur (test)
df_passe = df[df["annee"] <= 2024]
df_test = df[df["annee"] == 2025]

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

# --- ETAPE 1 : Cross-Conformal (Calibration robuste) ---
print("1. Calibration Croisée (Cross-Conformal) sur 2022-2024...")

X_passe = df_passe[features_num + features_cat]
y_passe = df_passe["rendement_t_ha"]

scores_erreurs = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, calib_idx in kf.split(X_passe):
    X_tr, y_tr = X_passe.iloc[train_idx], y_passe.iloc[train_idx]
    X_ca, y_ca = X_passe.iloc[calib_idx], y_passe.iloc[calib_idx]
    
    # On entraîne les modèles sur 80% des données passées
    m_bas = creer_modele(0.1).fit(X_tr, y_tr)
    m_haut = creer_modele(0.9).fit(X_tr, y_tr)
    
    # On prédit sur les 20% restants (qui n'ont pas servi à l'entraînement)
    pred_b = m_bas.predict(X_ca)
    pred_h = m_haut.predict(X_ca)
    
    # On calcule l'écart
    ecart = np.maximum(pred_b - y_ca, y_ca - pred_h)
    scores_erreurs.extend(ecart)

# On prend le quantile à 80% de toutes ces erreurs (la marge finale)
marge = np.quantile(scores_erreurs, 0.80)
print(f"   Marge de conformité finale : +{marge:.3f} t/ha")

# --- ETAPE 2 : Entraînement final sur TOUTES les données passées ---
print("2. Entraînement final sur 2022-2024...")
m_final_bas = creer_modele(0.1).fit(X_passe, y_passe)
m_final_med = creer_modele(0.5).fit(X_passe, y_passe)
m_final_haut = creer_modele(0.9).fit(X_passe, y_passe)

# --- ETAPE 3 : Test sur 2025 ---
print("3. Évaluation sur 2025...")
X_test = df_test[features_num + features_cat]
y_test = df_test["rendement_t_ha"]

pred_bas = m_final_bas.predict(X_test)
pred_med = m_final_med.predict(X_test)
pred_haut = m_final_haut.predict(X_test)

# On applique la marge
borne_basse = pred_bas - marge
borne_haute = pred_haut + marge

# Évaluation
dans_intervalle = (y_test >= borne_basse) & (y_test <= borne_haute)
couverture = dans_intervalle.mean() * 100

print("\n--- RESULTATS ---")
print(f"MAE (Médiane) : {mean_absolute_error(y_test, pred_med):.3f} t/ha")
print(f"Couverture de l'intervalle : {couverture:.1f} %")
print(f"(Cible : 76% à 84%)")