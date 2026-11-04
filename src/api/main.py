from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

# 1. Initialisation de l'API
app = FastAPI(title="AgroPrévision API", version="1.0")

# 2. Chargement des modèles et des données au démarrage
print("Chargement des modèles...")
modele_bas = joblib.load("models/modele_bas.pkl")
modele_med = joblib.load("models/modele_med.pkl")
modele_haut = joblib.load("models/modele_haut.pkl")
marge = joblib.load("models/marge_conformite.pkl")

# On charge le jeu de données pour pouvoir chercher les infos d'une parcelle
df = pd.read_csv("data/processed/dataset_modelisation.csv")
features = [
    "pluie_cumul_mm", "pause_pluvio_max_j", "temp_max_moy_c", 
    "ndvi_pic", "date_pic", "pente_croissance", "ndvi_auc", "engrais_kg_ha",
    "culture", "type_sol", "semence_certifiee", "departement"
]

# On prépare l'explicateur SHAP
preproc = modele_med.named_steps['preprocessor']
reg = modele_med.named_steps['regressor']
explainer = shap.TreeExplainer(reg)

# 3. Définition du format d'entrée (ce qu'on envoie à l'API)
class RequetePrevision(BaseModel):
    parcelle_id: str
    annee: int

# 4. Création du point d'accès (Endpoint)
@app.post("/predict")
def predire(requete: RequetePrevision):
    # On cherche la parcelle dans les données
    ligne = df[(df["parcelle_id"] == requete.parcelle_id) & (df["annee"] == requete.annee)]
    
    if ligne.empty:
        raise HTTPException(status_code=404, detail="Parcelle ou année introuvable dans la base.")
    
    # On extrait les données de la parcelle
    X_nouvelle = ligne[features]
    
    # Prédictions
    pred_bas = modele_bas.predict(X_nouvelle)[0]
    pred_med = modele_med.predict(X_nouvelle)[0]
    pred_haut = modele_haut.predict(X_nouvelle)[0]
    
    # Application de la marge de conformité
    borne_basse = max(0, pred_bas - marge) # Le rendement ne peut pas être négatif
    borne_haute = pred_haut + marge
    
    # Calcul des SHAP
    X_transforme = preproc.transform(X_nouvelle)
    shap_values = explainer.shap_values(X_transforme)[0]
    noms_colonnes = preproc.get_feature_names_out()
    
    # On prépare les 3 facteurs principaux
    df_shap = pd.DataFrame({'Facteur': noms_colonnes, 'Impact': shap_values})
    df_shap['Facteur'] = df_shap['Facteur'].str.replace('num__', '').str.replace('cat__', '')
    df_shap['Impact_abs'] = df_shap['Impact'].abs()
    df_shap = df_shap.sort_values(by='Impact_abs', ascending=False)
    
    top_3 = []
    for i in range(3):
        top_3.append({
            "facteur": df_shap.iloc[i]['Facteur'],
            "impact_t_ha": round(float(df_shap.iloc[i]['Impact']), 3),
            "sens": "Hausse" if df_shap.iloc[i]['Impact'] > 0 else "Baisse"
        })
        
    # 5. On renvoie la réponse au format JSON
    return {
        "parcelle_id": requete.parcelle_id,
        "annee": requete.annee,
        "rendement_median_t_ha": round(float(pred_med), 2),
        "intervalle_80": [round(float(borne_basse), 2), round(float(borne_haute), 2)],
        "facteurs_shap": top_3
    }