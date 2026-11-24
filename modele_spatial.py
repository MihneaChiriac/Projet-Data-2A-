import numpy as np
import pandas as pd
import requests
import shapely
import statsmodels.api as sm
from shapely.geometry import shape

from eau import charger_cours_d_eau, distance_a_l_eau, en_metres
from grille import charger_contour_paris, creer_carreaux, rattacher_aux_carreaux
from prediction import DATE_DEBUT, DOSSIER_DONNEES, tableau_coefficients

# le modèle n'a besoin que de la date et de la position de chaque signalement
FICHIER_RATS = DOSSIER_DONNEES / "clean_rats.csv"
COLONNES_RATS = ["DATEDECL", "X", "Y"]
# même source que aggrementation_data.py (espaces_verts.geojson), bois compris
URL_ESPACES_VERTS = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "espaces_verts/exports/geojson"
)
TYPE_BOIS = "Bois"
CIBLE = "nb_signalements"
VARIABLES = [
    "distance_eau_km",
    "distance_espace_vert_100m",
    "part_dans_paris",
    "dans_espace_vert",
    # les bois sont aussi des espaces verts : dans_bois mesure l'écart des bois par rapport aux autres
    "dans_bois",
]


def charger_positions(fichier) -> pd.DataFrame:
    return pd.read_csv(fichier, usecols=COLONNES_RATS, parse_dates=["DATEDECL"])


def telecharger_espaces_verts() -> list:
    reponse = requests.get(URL_ESPACES_VERTS, timeout=180)
    reponse.raise_for_status()
    return [espace for espace in reponse.json()["features"] if espace["geometry"] is not None]


def unir_en_metres(espaces: list):
    # make_valid répare les polygones mal formés
    union = shapely.force_2d(shapely.union_all([shapely.make_valid(shape(espace["geometry"])) for espace in espaces]))
    # coordonnées shapely dans l'ordre (lon, lat)
    return shapely.transform(union, lambda coords: np.column_stack(en_metres(coords[:, 1], coords[:, 0])))


def distance_a_l_espace_vert(lat, lon, espaces_verts):
    x, y = en_metres(lat, lon)
    return shapely.distance(shapely.points(x, y), espaces_verts)


def decrire_carreaux(carreaux: pd.DataFrame, contour, eau, espaces_verts, bois) -> pd.DataFrame:
    carreaux = carreaux.copy()
    rectangles = shapely.box(
        carreaux["lon_min"].to_numpy(),
        carreaux["lat_min"].to_numpy(),
        carreaux["lon_max"].to_numpy(),
        carreaux["lat_max"].to_numpy(),
    )
    # rapport de deux surfaces du même carreau : l'unité (degrés carrés) se simplifie
    carreaux["part_dans_paris"] = shapely.area(shapely.intersection(rectangles, contour)) / shapely.area(rectangles)
    lat_centre = (carreaux["lat_min"] + carreaux["lat_max"]) / 2
    lon_centre = (carreaux["lon_min"] + carreaux["lon_max"]) / 2
    # unités choisies pour des coefficients lisibles
    carreaux["distance_eau_km"] = distance_a_l_eau(lat_centre, lon_centre, eau) / 1000
    carreaux["distance_espace_vert_100m"] = distance_a_l_espace_vert(lat_centre, lon_centre, espaces_verts) / 100
    # distance nulle = centre dans un espace vert
    carreaux["dans_espace_vert"] = (carreaux["distance_espace_vert_100m"] == 0).astype(int)
    x_centre, y_centre = en_metres(lat_centre, lon_centre)
    carreaux["dans_bois"] = shapely.intersects(shapely.points(x_centre, y_centre), bois).astype(int)
    return carreaux


def compter_par_carreau(rats: pd.DataFrame, carreaux: pd.DataFrame, debut: str) -> pd.DataFrame:
    rats = rats[rats["DATEDECL"] >= debut]
    # dans les données, X = longitude et Y = latitude
    ids = rattacher_aux_carreaux(rats.rename(columns={"Y": "lat", "X": "lon"}), carreaux)
    if ids.isna().any():
        raise ValueError(f"{ids.isna().sum()} signalements hors des carreaux")
    comptes = ids.value_counts()
    return carreaux.assign(**{CIBLE: carreaux["carreau"].map(comptes).fillna(0).astype(int)})


def estimer_modele_spatial(carreaux: pd.DataFrame):
    return sm.OLS(carreaux[CIBLE], sm.add_constant(carreaux[VARIABLES])).fit(cov_type="HC1")


def preparer_carreaux(rats: pd.DataFrame) -> pd.DataFrame:
    espaces = telecharger_espaces_verts()
    espaces_verts = unir_en_metres(espaces)
    bois = unir_en_metres([espace for espace in espaces if espace["properties"].get("type_ev") == TYPE_BOIS])

    contour = charger_contour_paris()
    carreaux = decrire_carreaux(creer_carreaux(contour), contour, charger_cours_d_eau(), espaces_verts, bois)
    return compter_par_carreau(rats, carreaux, DATE_DEBUT)


if __name__ == "__main__":
    carreaux = preparer_carreaux(charger_positions(FICHIER_RATS))
    resultat = estimer_modele_spatial(carreaux)

    print("=== MODÈLE SPATIAL : total de signalements par carreau de 200 m ===")
    print(f"Données      : {FICHIER_RATS.name} (colonnes {', '.join(COLONNES_RATS)})")
    print(f"Carreaux     : {int(resultat.nobs)} (tous, y compris sans signalement), dont "
          f"{carreaux['dans_espace_vert'].sum()} avec le centre dans un espace vert "
          f"({carreaux['dans_bois'].sum()} dans un bois)")
    print(f"Période      : à partir du {DATE_DEBUT}, {carreaux[CIBLE].sum()} signalements")
    print(f"{CIBLE} : moyenne {carreaux[CIBLE].mean():.2f}, médiane {carreaux[CIBLE].median():.0f}, "
          f"max {carreaux[CIBLE].max()}, carreaux à 0 : {(carreaux[CIBLE] == 0).mean():.1%}")
    print(f"R²           : {resultat.rsquared:.4f}")
    print("Écarts-types robustes à l'hétéroscédasticité (HC1)")

    print("\n--- Coefficients ---")
    print(tableau_coefficients(resultat).to_string(float_format=lambda x: f"{x:.4g}"))

    print("\n--- Test joint (Wald, H0 : tous les coefficients sauf la constante sont nuls) ---")
    test = resultat.f_test(", ".join(f"{variable} = 0" for variable in VARIABLES))
    # selon les cas, statsmodels renvoie un nombre ou un tableau à un élément
    print(f"F = {np.asarray(test.fvalue).item():.4g}, p-valeur = {np.asarray(test.pvalue).item():.4g}, "
          f"{len(VARIABLES)} contraintes")
