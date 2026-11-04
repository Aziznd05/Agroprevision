import pandas as pd

# 1. On définit le chemin vers notre fichier de données
chemin_fichier = "data/raw/parcelles_rendements_agrosahel_brut.csv"

# 2. On demande à Pandas de lire le fichier CSV
donnees = pd.read_csv(chemin_fichier)

# 3. On affiche les 5 premières lignes pour voir à quoi ça ressemble
print(donnees.head())