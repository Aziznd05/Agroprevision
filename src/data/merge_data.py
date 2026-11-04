import pandas as pd

# 1. Charger les deux fichiers propres
parcelles = pd.read_csv("data/processed/parcelles_clean_final.csv")
features = pd.read_csv("data/processed/ndvi_features.csv")

# 2. Fusionner (comme un VLOOKUP sur Excel) en utilisant parcelle_id et annee
dataset = pd.merge(parcelles, features, on=["parcelle_id", "annee"], how="left")

# 3. Créer le jeu de données "à date partielle" (Décades 1 à 7)
# Le cahier des charges demande de prévoir 6 à 8 semaines avant la récolte.
# Pour l'instant, on garde tout le monde mais on vérifiera plus tard.
# (On ajoutera la logique de découpe à l'étape modélisation).

print("--- JEU DE DONNEES FUSIONNE ---")
print(f"Nombre de lignes : {len(dataset)}")
print(dataset[["parcelle_id", "annee", "rendement_t_ha", "ndvi_pic", "pluie_cumul_mm"]].head())

# 4. Sauvegarder le jeu de données final pour le Machine Learning
dataset.to_csv("data/processed/dataset_modelisation.csv", index=False)
print("\nFichier sauvegardé dans : data/processed/dataset_modelisation.csv")