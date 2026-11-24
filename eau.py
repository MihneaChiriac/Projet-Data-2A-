import json
from math import cos, radians
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import shape

URL_IGN = "https://data.geopf.fr/wfs/ows"
# surfaces d'eau de la bd topo de l'ign, dans un rectangle un peu plus grand que paris
PARAMETRES_IGN = {
    "SERVICE": "WFS",
    "VERSION": "2.0.0",
    "REQUEST": "GetFeature",
    "TYPENAMES": "BDTOPO_V3:surface_hydrographique",
    # lon min, lat min, lon max, lat max
    "BBOX": "2.20,48.80,2.50,48.92,CRS:84",
    "OUTPUTFORMAT": "application/json",
    "COUNT": 5000,
}
# seine, marne et canaux ; ajouter "Retenue" pour les lacs des bois
NATURES_GARDEES = ["Ecoulement naturel", "Canal"]
DOSSIER_DONNEES = Path(__file__).parent / "données"
FICHIER_EAU = DOSSIER_DONNEES / "cours_d_eau.geojson"

METRES_PAR_DEGRE = 111_320
COS_PARIS = cos(radians(48.86))


# lat/lon en mètres sur un plan local, assez précis à l'échelle de paris
def en_metres(lat, lon):
    x = np.asarray(lon) * METRES_PAR_DEGRE * COS_PARIS
    y = np.asarray(lat) * METRES_PAR_DEGRE
    return x, y


# télécharge les cours d'eau et garde une copie dans données/
# appelée par charger_cours_d_eau si la copie n'existe pas : supprimer données/cours_d_eau.geojson
# pour la retélécharger (par exemple si on change NATURES_GARDEES)
def telecharger_cours_d_eau() -> dict:
    reponse = requests.get(URL_IGN, params=PARAMETRES_IGN, timeout=150)
    reponse.raise_for_status()
    donnees = reponse.json()
    # l'ign coupe la réponse au-delà de COUNT objets
    if donnees["numberMatched"] > len(donnees["features"]):
        raise ValueError("réponse de l'ign incomplète, augmenter COUNT")
    donnees["features"] = [
        surface for surface in donnees["features"] if surface["properties"]["nature"] in NATURES_GARDEES
    ]
    FICHIER_EAU.parent.mkdir(exist_ok=True)
    FICHIER_EAU.write_text(json.dumps(donnees), encoding="utf-8")
    return donnees


# toute l'eau en une seule forme, en mètres ; copie locale si elle existe
def charger_cours_d_eau():
    if FICHIER_EAU.exists():
        donnees = json.loads(FICHIER_EAU.read_text(encoding="utf-8"))
    else:
        donnees = telecharger_cours_d_eau()
    # l'ign donne aussi l'altitude, inutile ici
    eau = shapely.force_2d(shapely.union_all([shape(surface["geometry"]) for surface in donnees["features"]]))
    # coordonnées shapely dans l'ordre (lon, lat)
    return shapely.transform(eau, lambda coords: np.column_stack(en_metres(coords[:, 1], coords[:, 0])))


# distance en mètres au cours d'eau le plus proche, 0 si le point est dans l'eau
# marche pour un seul point ou pour des colonnes entières
def distance_a_l_eau(lat, lon, eau):
    x, y = en_metres(lat, lon)
    return shapely.distance(shapely.points(x, y), eau)


# copie du tableau avec la colonne distance_eau_m, arrondie au mètre
def ajouter_distance_eau(signalements: pd.DataFrame, eau, col_lat="Y", col_lon="X") -> pd.DataFrame:
    signalements = signalements.copy()
    distances = distance_a_l_eau(signalements[col_lat], signalements[col_lon], eau)
    signalements["distance_eau_m"] = np.round(distances).astype(int)
    return signalements
