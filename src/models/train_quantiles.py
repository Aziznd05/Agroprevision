import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# 1. Charger les données
df = pd.read_csv("data/processed/dataset_modelisation.csv")
df = df.dropna(subset=["rendement_t_ha"])

# 2. Séparation Temporelle
train_df = df[df["annee"] <= 2024]
test_df = df[df["annee"] == 2025]

features_num = [
    "pluie_cumul_mm", "pause_pluvio_max_j", "temp_max_moy_c", 
    "ndvi_pic", "date_pic", "pente_croissance", "ndvi_auc", "engrais_kg_ha"
]
features_cat = ["culture", "type_sol", "semence_certifiee", "departement"]

# Préparation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])

X_train = train_df[features_num + features_cat]
y_train = train_df["rendement_t_ha"]
X_test = test_df[features_num + features_cat]
y_test = test_df["rendement_t_ha"]

def entrainer_quantile(alpha):
    """Entraîne un modèle LightGBM pour un quantile spécifique"""
    modele = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            objective='quantile', # On change l'objectif !
            alpha=alpha,          # Le quantile voulu (0.1, 0.5 ou 0.9)
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.05, 
            random_state=42,
            verbose=-1
        ))
    ])
    modele.fit(X_train, y_train)
    return modele

print("Entraînement des 3 modèles...")
# On entraîne les 3 modèles
modele_bas = entrainer_quantile(alpha=0.1)  # Borne basse (10%)
modele_med = entrainer_quantile(alpha=0.5)  # Médiane (50%)
modele_haut = entrainer_quantile(alpha=0.9) # Borne haute (90%)

# Prédictions
pred_bas = modele_bas.predict(X_test)
pred_med = modele_med.predict(X_test)
pred_haut = modele_haut.predict(X_test)

# --- ÉVALUATION DE LA COUVERTURE ---
# On regarde combien de fois le vrai rendement tombe dans notre intervalle
dans_intervalle = (y_test >= pred_bas) & (y_test <= pred_haut)
couverture = dans_intervalle.mean() * 100

print("\n--- RESULTATS ---")
print(f"MAE (Erreur de la médiane) : {mean_absolute_error(y_test, pred_med):.3f} t/ha")
print(f"Couverture de l'intervalle à 80% : {couverture:.1f} %")
print(f"(Cible exigée par le cahier des charges : entre 76% et 84%)")

# Affichage d'un exemple de prévision pour une parcelle
print("\n--- EXEMPLE DE PREVISION (Parcelle 1 du test) ---")
print(f"Rendement réel  : {y_test.iloc[0]:.2f} t/ha")
print(f"Borne basse     : {pred_bas[0]:.2f} t/ha")
print(f"Médiane prévue  : {pred_med[0]:.2f} t/ha")
print(f"Borne haute     : {pred_haut[0]:.2f} t/ha")