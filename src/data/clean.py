import pandas as pd
import numpy as np
import os

chemin_brut = "data/raw/parcelles_rendements_agrosahel_brut.csv"
donnees = pd.read_csv(chemin_brut)

MAP_CULTURE = {
    "arachide": "Arachide", "arachides": "Arachide",
    "mil": "Mil", "mil souna": "Mil",
    "mais": "Maïs", "maïs": "Maïs",
    "sorgho": "Sorgho", "sorgo": "Sorgho",
    "niebe": "Niébé", "niébé": "Niébé", "cowpea": "Niébé"
}
MAP_REGION = {
    "kaolack": "Kaolack", "fatick": "Fatick", "fatik": "Fatick",
    "kaffrine": "Kaffrine", "kafrine": "Kaffrine", "diourbel": "Diourbel"
}
MAP_OUI_NON = {
    "oui": "Oui", "o": "Oui", "1": "Oui",
    "non": "Non", "n": "Non", "0": "Non"
}

def normaliser_texte(valeur, dictionnaire):
    if pd.isna(valeur):
        return np.nan
    valeur_propre = str(valeur).strip().lower()
    return dictionnaire.get(valeur_propre, valeur_propre.capitalize())

print("--- AVANT NETTOYAGE (manquants) ---")
print(donnees[["parcelle_id", "type_sol", "pH_sol"]].head(10))

# ==========================================
# REGLES DE NETTOYAGE (Numérique & Texte & Dates & Doublons)
# ==========================================
rendement = pd.to_numeric(donnees["rendement_t_ha"], errors="coerce")
donnees["rendement_t_ha"] = rendement.mask(rendement > 100, rendement / 1000).mask(rendement < 0, np.nan)

surface = pd.to_numeric(donnees["surface_ha"], errors="coerce")
donnees["surface_ha"] = surface.mask(surface > 100, surface / 10000)

ph = pd.to_numeric(donnees["pH_sol"], errors="coerce")
donnees["pH_sol"] = ph.mask((ph < 0) | (ph > 14), np.nan)

pluie = pd.to_numeric(donnees["pluie_cumul_mm"], errors="coerce")
donnees["pluie_cumul_mm"] = pluie.mask((pluie < 0) | (pluie > 2000), np.nan)

temp = pd.to_numeric(donnees["temp_max_moy_c"], errors="coerce")
donnees["temp_max_moy_c"] = temp.mask(temp > 60, (temp - 32) * 5 / 9)

annee = pd.to_numeric(donnees["annee"], errors="coerce")
donnees["annee"] = annee.mask((annee < 2020) | (annee > 2026), np.nan)

donnees["culture"] = donnees["culture"].apply(lambda x: normaliser_texte(x, MAP_CULTURE))
donnees["region"] = donnees["region"].apply(lambda x: normaliser_texte(x, MAP_REGION))
donnees["semence_certifiee"] = donnees["semence_certifiee"].apply(lambda x: normaliser_texte(x, MAP_OUI_NON))
donnees["irrigation"] = donnees["irrigation"].apply(lambda x: normaliser_texte(x, MAP_OUI_NON))
donnees["appui_technique"] = donnees["appui_technique"].apply(lambda x: normaliser_texte(x, MAP_OUI_NON))

dates = pd.to_datetime(donnees["date_semis"], errors="coerce", dayfirst=True)
donnees["date_semis"] = dates.dt.strftime("%Y-%m-%d")

lat = pd.to_numeric(donnees["latitude"], errors="coerce")
lon = pd.to_numeric(donnees["longitude"], errors="coerce")
inverse = (lat < 0) | (lon > 0)
lat_corrigee = lat.where(~inverse, lon)
lon_corrigee = lon.where(~inverse, lat)
limites_valides = lat_corrigee.between(12, 17) & lon_corrigee.between(-18, -11)
donnees["latitude"] = lat_corrigee.where(limites_valides, np.nan)
donnees["longitude"] = lon_corrigee.where(limites_valides, np.nan)

donnees = donnees.drop_duplicates().reset_index(drop=True)

# ==========================================
# NOUVELLES REGLES : IMPUTATION (EF-03)
# ==========================================

# 1. Attributs constants par parcelle (le sol et le pH ne changent pas d'une année à l'autre)
# On remplit les trous en utilisant les autres années de la même parcelle
colonnes_constantes = ["type_sol", "pH_sol", "matiere_organique_pct"]
for col in colonnes_constantes:
    donnees[col] = donnees.groupby("parcelle_id")[col].transform(lambda x: x.ffill().bfill())

# 2. Pour les valeurs qui manquent toujours (parcelles où l'info n'a jamais été saisie)
# On utilise la médiane du département pour cette culture
for col in ["pH_sol", "matiere_organique_pct"]:
    mediane = donnees.groupby(["culture", "departement"])[col].transform("median")
    donnees[col] = donnees[col].fillna(mediane)

# ==========================================

print("\n--- APRES NETTOYAGE ET IMPUTATION (manquants) ---")
print(donnees[["parcelle_id", "type_sol", "pH_sol"]].head(10))

# Sauvegarde
os.makedirs("data/processed", exist_ok=True)
chemin_propre = "data/processed/parcelles_clean_final.csv"
donnees.to_csv(chemin_propre, index=False)
print(f"\nFichier sauvegardé dans : {chemin_propre}")