import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter

chemin_brut = "data/raw/series_ndvi_agrosahel_1.csv"
ndvi = pd.read_csv(chemin_brut)

print("--- AVANT NETTOYAGE NDVI ---")
print(ndvi[["parcelle_id", "annee", "decade", "ndvi"]].head(10))

# ==========================================
# REGLES DE NETTOYAGE NDVI (EF-04)
# ==========================================

# DQ-011 : Valeurs hors domaine (ex: -9999, -0.5, 2.0)
# Le NDVI normal d'une culture est entre -0.1 et 1.0
ndvi["ndvi"] = pd.to_numeric(ndvi["ndvi"], errors="coerce")
ndvi.loc[(ndvi["ndvi"] < -0.1) | (ndvi["ndvi"] > 1.0), "ndvi"] = np.nan

ndvi["evi"] = pd.to_numeric(ndvi["evi"], errors="coerce")
ndvi.loc[(ndvi["evi"] < -0.1) | (ndvi["evi"] > 1.0), "evi"] = np.nan

# DQ-012 : Interpolation et Lissage
# On va traiter chaque parcelle, année par année
resultats = []

for (parcelle, annee), groupe in ndvi.groupby(["parcelle_id", "annee"]):
    # 1. On s'assure d'avoir les 12 décades (1 à 12), même si vides
    toutes_les_decades = pd.DataFrame({"decade": range(1, 13)})
    groupe_complet = toutes_les_decades.merge(groupe, on="decade", how="left")
    groupe_complet["parcelle_id"] = parcelle
    groupe_complet["annee"] = annee
    
    # 2. Interpolation linéaire : on remplit les trous entre les nuages
    groupe_complet["ndvi"] = groupe_complet["ndvi"].interpolate(method="linear", limit_direction="both")
    
    # 3. Lissage Savitzky-Golay (fenêtre de 5, degré 2) pour adoucir la courbe
    # On le fait seulement si on a assez de données non nulles
    if groupe_complet["ndvi"].notna().sum() >= 5:
        # On remplit les derniers trous aux extrémités pour éviter les erreurs
        valeurs = groupe_complet["ndvi"].ffill().bfill().values
        groupe_complet["ndvi_lisse"] = savgol_filter(valeurs, window_length=5, polyorder=2)
    else:
        groupe_complet["ndvi_lisse"] = groupe_complet["ndvi"]
        
    resultats.append(groupe_complet)

ndvi_propre = pd.concat(resultats, ignore_index=True)

# On réorganise les colonnes pour faire propre
colonnes = ["parcelle_id", "annee", "decade", "ndvi", "ndvi_lisse", "evi"]
ndvi_propre = ndvi_propre[colonnes]

# ==========================================

print("\n--- APRES NETTOYAGE NDVI ---")
print(ndvi_propre[["parcelle_id", "annee", "decade", "ndvi", "ndvi_lisse"]].head(10))

# Sauvegarde
os.makedirs("data/processed", exist_ok=True)
chemin_propre = "data/processed/series_ndvi_clean.csv"
ndvi_propre.to_csv(chemin_propre, index=False)
print(f"\nFichier sauvegardé dans : {chemin_propre}")