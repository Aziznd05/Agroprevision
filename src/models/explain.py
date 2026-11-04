import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# 1. Charger les données
df = pd.read_csv("data/processed/dataset_modelisation.csv")
df = df.dropna(subset=["rendement_t_ha"])

train_df = df[df["annee"] <= 2024]
test_df = df[df["annee"] == 2025]

features_num = [
    "pluie_cumul_mm", "pause_pluvio_max_j", "temp_max_moy_c", 
    "ndvi_pic", "date_pic", "pente_croissance", "ndvi_auc", "engrais_kg_ha"
]
features_cat = ["culture", "type_sol", "semence_certifiee", "departement"]

# Préparation des données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
    ])

X_train = train_df[features_num + features_cat]
y_train = train_df["rendement_t_ha"]
X_test = test_df[features_num + features_cat]

# On entraîne un modèle LightGBM classique (médiane)
modele = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1))
])
modele.fit(X_train, y_train)

# --- CALCUL DE SHAP ---
# On doit extraire le préprocesseur et le modèle pour SHAP
preproc_entraine = modele.named_steps['preprocessor']
modele_entraine = modele.named_steps['regressor']

# On transforme nos données de test avec le préprocesseur
X_test_transforme = preproc_entraine.transform(X_test)

# On récupère les noms des colonnes (ex: "num__pluie_cumul_mm", "cat__culture_Maïs", etc.)
noms_colonnes = preproc_entraine.get_feature_names_out()

# On crée l'explicateur SHAP
explainer = shap.TreeExplainer(modele_entraine)
shap_values = explainer.shap_values(X_test_transforme)

# --- AFFICHAGE POUR UNE PARCELLE ---
# On prend la première parcelle du jeu de test comme exemple
index_parcelle = 0
parcelle_id = test_df.iloc[index_parcelle]["parcelle_id"]
prediction = modele.predict(X_test.iloc[[index_parcelle]])[0]

print(f"--- EXPLICATION DE LA PREVISION POUR LA PARCELLE {parcelle_id} ---")
print(f"Rendement prévu : {prediction:.2f} t/ha\n")

# On récupère les valeurs SHAP pour cette parcelle
valeurs_parcelle = shap_values[index_parcelle]

# On crée un DataFrame pour trier les facteurs
df_shap = pd.DataFrame({
    'Facteur': noms_colonnes,
    'Impact': valeurs_parcelle
})

# On nettoie les noms (on enlève "num__" et "cat__" pour faire plus propre)
df_shap['Facteur'] = df_shap['Facteur'].str.replace('num__', '').str.replace('cat__', '')

# On sépare les facteurs qui font MONTER la prévision (positifs) et ceux qui la font BAISSER (négatifs)
df_shap['Impact_absolu'] = df_shap['Impact'].abs()
df_shap = df_shap.sort_values(by='Impact_absolu', ascending=False)

print("Top 3 des facteurs les plus influents :")
for i in range(3):
    nom = df_shap.iloc[i]['Facteur']
    impact = df_shap.iloc[i]['Impact']
    sens = "HAUSSE" if impact > 0 else "BAISSE"
    print(f"{i+1}. {nom} (Fait {sens} la prévision de {abs(impact):.2f} t/ha)")