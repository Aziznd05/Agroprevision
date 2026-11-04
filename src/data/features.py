import pandas as pd
import numpy as np

# 1. Charger les données NDVI propres
ndvi = pd.read_csv("data/processed/series_ndvi_clean.csv")

# 2. Créer une liste pour stocker les résultats
resultats = []

# 3. Pour chaque parcelle et chaque année, on résume la courbe
for (parcelle, annee), groupe in ndvi.groupby(["parcelle_id", "annee"]):
    # On s'assure que la série est bien triée par décade
    groupe = groupe.sort_values("decade")
    
    # Le pic de NDVI (la valeur maximale atteinte)
    pic = groupe["ndvi"].max()
    
    # La date du pic (à quelle décade le maximum a été atteint)
    # idxmax() renvoie l'index de la ligne où se trouve le max
    index_pic = groupe["ndvi"].idxmax()
    date_pic = groupe.loc[index_pic, "decade"]
    
    # La pente de croissance (différence entre la décade 5 et 1 par exemple)
    # On utilise les valeurs lissées pour éviter le bruit
    val_debut = groupe[groupe["decade"] == 3]["ndvi_lisse"].values[0]
    val_milieu = groupe[groupe["decade"] == 7]["ndvi_lisse"].values[0]
    pente = val_milieu - val_debut
    
    # L'aire sous la courbe (la somme des valeurs, approxime la santé globale)
    auc = groupe["ndvi"].sum()
    
    resultats.append({
        "parcelle_id": parcelle,
        "annee": annee,
        "ndvi_pic": pic,
        "date_pic": date_pic,
        "pente_croissance": pente,
        "ndvi_auc": auc
    })

# 4. Créer un nouveau tableau avec ces variables
features_ndvi = pd.DataFrame(resultats)

print("--- VARIABLES EXTRAITES DU NDVI ---")
print(features_ndvi.head())

# 5. Sauvegarder
features_ndvi.to_csv("data/processed/ndvi_features.csv", index=False)
print("\nFichier sauvegardé dans : data/processed/ndvi_features.csv")