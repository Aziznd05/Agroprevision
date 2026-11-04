import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Charger le jeu de données
df = pd.read_csv("data/processed/dataset_modelisation.csv")
df = df.dropna(subset=["rendement_t_ha"])

# 2. Séparation Temporelle
train_df = df[df["annee"] <= 2024]
test_df = df[df["annee"] == 2025]

# 3. Baseline
baseline_moyennes = train_df.groupby(["departement", "culture"])["rendement_t_ha"].mean().reset_index()
baseline_moyennes = baseline_moyennes.rename(columns={"rendement_t_ha": "prediction_baseline"})
test_df = test_df.merge(baseline_moyennes, on=["departement", "culture"], how="left")
test_df["prediction_baseline"] = test_df["prediction_baseline"].fillna(train_df["rendement_t_ha"].mean())

# 4. Le Modèle Ridge (Version blindée)

features_num = [
    "pluie_cumul_mm", "pause_pluvio_max_j", "temp_max_moy_c", 
    "ndvi_pic", "date_pic", "pente_croissance", "ndvi_auc", "engrais_kg_ha"
]
features_cat = ["culture", "type_sol", "semence_certifiee", "departement"]

# On crée un pipeline pour les chiffres : on remplit les trous (médiane), puis on normalise
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# On assemble le tout
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])

# On crée le pipeline complet
modele_ridge = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# On prépare X et y
X_train = train_df[features_num + features_cat]
y_train = train_df["rendement_t_ha"]
X_test = test_df[features_num + features_cat]
y_test = test_df["rendement_t_ha"]

# On entraîne
modele_ridge.fit(X_train, y_train)
predictions_ridge = modele_ridge.predict(X_test)

# 5. Évaluation
def evaluer(y_vrai, y_predit, nom_modele):
    mae = mean_absolute_error(y_vrai, y_predit)
    rmse = np.sqrt(mean_squared_error(y_vrai, y_predit))
    print(f"--- {nom_modele} ---")
    print(f"MAE  (Erreur moyenne) : {mae:.3f} t/ha")
    print(f"RMSE (Erreur globale) : {rmse:.3f} t/ha")
    print("")

evaluer(y_test, test_df["prediction_baseline"], "BASELINE (Moyenne historique)")
evaluer(y_test, predictions_ridge, "MODELE RIDGE (avec catégories)")