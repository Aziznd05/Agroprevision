import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# 1. Charger le jeu de données
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

# Préparation (identique à Ridge, sauf que LightGBM n'a pas besoin de normalisation)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])

# On crée le modèle LightGBM
# n_estimators = nombre d'arbres, max_depth = profondeur maximale
modele_lgbm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42,
        verbose=-1 # Pour cacher les messages d'avertissement
    ))
])

# Préparation des données
X_train = train_df[features_num + features_cat]
y_train = train_df["rendement_t_ha"]
X_test = test_df[features_num + features_cat]
y_test = test_df["rendement_t_ha"]

# Entraînement
modele_lgbm.fit(X_train, y_train)

# Prédiction
predictions_lgbm = modele_lgbm.predict(X_test)

# Évaluation
def evaluer(y_vrai, y_predit, nom_modele):
    mae = mean_absolute_error(y_vrai, y_predit)
    rmse = np.sqrt(mean_squared_error(y_vrai, y_predit))
    print(f"--- {nom_modele} ---")
    print(f"MAE  (Erreur moyenne) : {mae:.3f} t/ha")
    print(f"RMSE (Erreur globale) : {rmse:.3f} t/ha")
    print("")

evaluer(y_test, predictions_lgbm, "MODELE LIGHTGBM")