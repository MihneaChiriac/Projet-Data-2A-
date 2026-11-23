#objectif: ajouter les variables suivantes: indicatrice de présence dans un bois,
# distance à l'espace vert le plus proche
# distance au composteur le plus proche
# données météo journalières

import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree



data = pd.read_csv("clean_rats.csv") #le fichier qui arrive en sortie de data_preparation.py

espaces_verts = gpd.read_file("espaces_verts.geojson")
composteurs = gpd.read_file("dechets-menagers-points-dapport-volontaire-composteurs.geojson")
meteo = pd.read_csv("clean_meteo.csv", index_col=0)

# Reprojection en Lambert 93 (mètres)
ev = espaces_verts.to_crs(2154)

# Union de tous les espaces verts, bois compris
ev_union = ev.geometry.union_all()

# Union des deux bois de Paris
bois = ev[ev["type_ev"] == "Bois"].copy()
bois_union = bois.geometry.union_all()

composteurs = composteurs[composteurs.geometry.notna()& ~composteurs.geometry.is_empty].copy()

# rEprojection en Lambert 93
composteurs = composteurs.to_crs(2154)

# Coordonnées des composteurs
xy_comp = composteurs.geometry.get_coordinates().to_numpy()

# Arbre de recherche du plus proche voisin
arbre = cKDTree(xy_comp)


points = gpd.GeoDataFrame(
    data.copy(),
    geometry=gpd.points_from_xy(
        data["X"],
        data["Y"]
    ),
    crs="EPSG:4326"
)

# Reprojection en Lambert 93
pts = points.to_crs(2154)

valides = (pts.geometry.notna()& ~pts.geometry.is_empty)
indices_valides = np.flatnonzero(valides.to_numpy())


dans_bois = pd.Series(pd.NA, index=data.index, dtype="Int64")

# indicatrice de présence dans un bois
dans_bois.iloc[indices_valides] = (
    pts.loc[valides].geometry.intersects(bois_union).astype(int).to_numpy())


distance_ev = np.full(len(data), np.nan)

# fistance a l'union des espaces verts
distance_ev[indices_valides] = (
    pts.loc[valides].geometry.distance(ev_union).to_numpy())

distance_comp = np.full(len(data), np.nan)
xy_obs = pts.loc[valides].geometry.get_coordinates().to_numpy()

#recherche composteur le plus proche
d, _ = arbre.query(xy_obs)

distance_comp[indices_valides] =d

data["dans_bois"] = dans_bois.to_numpy()
data["distance_espace_vert"] = distance_ev
data["distance_composteur"] = distance_comp

#donnees météo
meteo = meteo.rename(columns={"JOUR": "JOURDECL", "MOIS": "MOISDECL","ANNEE": "ANNEEDECL"})

# variables meteo à ajouter
colonnes_meteo = [col for col in meteo.columns if col not in ["DATE", "JOURDECL", "MOISDECL", "ANNEEDECL"]]

# jointure
data = data.merge(
    meteo[["JOURDECL", "MOISDECL", "ANNEEDECL"] + colonnes_meteo],
    on=["JOURDECL", "MOISDECL", "ANNEEDECL"],
    how="left",validate="many_to_one")
data.to_csv("clean_rats_enrichi.csv", index=False)
print(f"nb dobservations : {len(data)}")
