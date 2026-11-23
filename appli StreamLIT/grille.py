from math import cos, radians
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import shape

URL_ARRONDISSEMENTS = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "arrondissements/exports/geojson"
)
DOSSIER_DONNEES = Path(__file__).parent / "data"

TAILLE_CARREAU_M = 200
# coin sud-ouest de la grille, un peu en dehors de paris
LAT_ORIGINE = 48.81
LON_ORIGINE = 2.22
# 1 degré de latitude ≈ 111,32 km partout
PAS_LAT = TAILLE_CARREAU_M / 111_320
# 1 degré de longitude plus court à la latitude de paris
PAS_LON = TAILLE_CARREAU_M / (111_320 * cos(radians(48.86)))


# contour de paris : union des 20 arrondissements
def charger_contour_paris():
    reponse = requests.get(URL_ARRONDISSEMENTS, timeout=60)
    reponse.raise_for_status()
    polygones = [shape(arrondissement["geometry"]) for arrondissement in reponse.json()["features"]]
    return shapely.union_all(polygones)


# identifiant texte "ligne_colonne", le même dans tous les fichiers
def identifiant(lignes: pd.Series, colonnes: pd.Series) -> pd.Series:
    return lignes.astype(str) + "_" + colonnes.astype(str)


# grille de carreaux de 200 m, seulement ceux qui touchent paris
def creer_carreaux(contour) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = contour.bounds
    if lat_min < LAT_ORIGINE or lon_min < LON_ORIGINE:
        raise ValueError("l'origine de la grille doit être au sud-ouest de paris")

    nb_lignes = int(np.ceil((lat_max - LAT_ORIGINE) / PAS_LAT))
    nb_colonnes = int(np.ceil((lon_max - LON_ORIGINE) / PAS_LON))
    # toutes les combinaisons ligne / colonne du rectangle autour de paris
    lignes, colonnes = np.meshgrid(np.arange(nb_lignes), np.arange(nb_colonnes), indexing="ij")
    carreaux = pd.DataFrame({"ligne": lignes.ravel(), "colonne": colonnes.ravel()})

    carreaux["lat_min"] = LAT_ORIGINE + carreaux["ligne"] * PAS_LAT
    carreaux["lat_max"] = carreaux["lat_min"] + PAS_LAT
    carreaux["lon_min"] = LON_ORIGINE + carreaux["colonne"] * PAS_LON
    carreaux["lon_max"] = carreaux["lon_min"] + PAS_LON

    # shapely attend (lon, lat), pas (lat, lon)
    rectangles = shapely.box(
        carreaux["lon_min"].to_numpy(),
        carreaux["lat_min"].to_numpy(),
        carreaux["lon_max"].to_numpy(),
        carreaux["lat_max"].to_numpy(),
    )
    # on garde aussi les carreaux coupés par la limite de paris
    carreaux = carreaux[shapely.intersects(rectangles, contour)].copy()
    carreaux.insert(0, "carreau", identifiant(carreaux["ligne"], carreaux["colonne"]))
    return carreaux.reset_index(drop=True)


# carreau de chaque point, par simple calcul grâce à la grille régulière
def rattacher_aux_carreaux(points: pd.DataFrame, carreaux: pd.DataFrame) -> pd.Series:
    lignes = np.floor((points["lat"] - LAT_ORIGINE) / PAS_LAT).astype(int)
    colonnes = np.floor((points["lon"] - LON_ORIGINE) / PAS_LON).astype(int)
    ids = identifiant(lignes, colonnes)
    # vide si le point tombe hors des carreaux gardés
    return ids.where(ids.isin(carreaux["carreau"]))


