import pandas as pd
import numpy as np

# Chemins
chemin_brut = "data/raw/parcelles_rendements_agrosahel_brut.csv"
chemin_rapport = "reports/rapport_qualite.html"

# On charge les données brutes pour compter les anomalies
donnees = pd.read_csv(chemin_brut)
total_lignes = len(donnees)

# Dictionnaire pour stocker les résultats
resultats = {}

# 1. Rendement (kg/ha -> t/ha)
rendement = pd.to_numeric(donnees["rendement_t_ha"], errors="coerce")
anomalies_rendement = (rendement > 100).sum() + (rendement < 0).sum()
resultats["Rendement (kg/ha)"] = anomalies_rendement

# 2. Surface (m2 -> ha)
surface = pd.to_numeric(donnees["surface_ha"], errors="coerce")
anomalies_surface = (surface > 100).sum()
resultats["Surface (m2)"] = anomalies_surface

# 3. pH hors domaine
ph = pd.to_numeric(donnees["pH_sol"], errors="coerce")
anomalies_ph = ((ph < 0) | (ph > 14)).sum()
resultats["pH hors [0-14]"] = anomalies_ph

# 4. Pluie aberrante
pluie = pd.to_numeric(donnees["pluie_cumul_mm"], errors="coerce")
anomalies_pluie = ((pluie < 0) | (pluie > 2000)).sum()
resultats["Pluie aberrante"] = anomalies_pluie

# 5. Température Fahrenheit
temp = pd.to_numeric(donnees["temp_max_moy_c"], errors="coerce")
anomalies_temp = (temp > 60).sum()
resultats["Température (°F)"] = anomalies_temp

# 6. Années impossibles
annee = pd.to_numeric(donnees["annee"], errors="coerce")
anomalies_annee = ((annee < 2020) | (annee > 2026)).sum()
resultats["Années impossibles"] = anomalies_annee

# 7. Coordonnées inversées
lat = pd.to_numeric(donnees["latitude"], errors="coerce")
lon = pd.to_numeric(donnees["longitude"], errors="coerce")
anomalies_gps = ((lat < 0) | (lon > 0)).sum()
resultats["GPS inversés"] = anomalies_gps

# 8. Doublons
doublons = donnees.duplicated().sum()
resultats["Doublons"] = doublons

# --- GENERATION DU HTML ---
html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Qualité - AgroPrévision</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; background-color: #f9f9f9; }}
        h1 {{ color: #176b45; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 800px; margin-top: 20px; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #176b45; color: white; }}
        .total {{ font-weight: bold; background-color: #eaf6ee; }}
    </style>
</head>
<body>
    <h1>Rapport de Qualité des Données (L2)</h1>
    <p><strong>Projet :</strong> AgroPrévision (AO-ASA-2026-02)</p>
    <p><strong>Fichier analysé :</strong> parcelles_rendements_agrosahel_brut.csv</p>
    <p><strong>Nombre total de lignes brutes :</strong> {total_lignes}</p>
    
    <h2>Anomalies détectées et corrigées</h2>
    <table>
        <tr>
            <th>Type d'anomalie (Règle)</th>
            <th>Nombre de lignes concernées</th>
        </tr>
"""

total_anomalies = 0
for regle, nombre in resultats.items():
    html += f"<tr><td>{regle}</td><td>{nombre}</td></tr>\n"
    total_anomalies += nombre

html += f"""
        <tr class="total">
            <td>Total des anomalies traitées</td>
            <td>{total_anomalies}</td>
        </tr>
    </table>
    
    <p style="margin-top:30px; font-size: 0.9em; color: #666;">
        Généré automatiquement par le pipeline de nettoyage Python.
    </p>
</body>
</html>
"""

# On écrit le fichier
import os
os.makedirs("reports", exist_ok=True)
with open(chemin_rapport, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Rapport généré avec succès : {chemin_rapport}")