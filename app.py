from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --- ajout dbscan ---
from dbscan import regrouper_points
from modele_spatial import CIBLE, estimer_modele_spatial, preparer_carreaux
from prediction import (
    DATE_DEBUT,
    FICHIER_METEO,
    FICHIER_RATS_COMPLET,
    SAISON_PAR_MOIS,
    SEUIL_PLUIE_MM,
    charger_meteo,
    charger_signalements,
    construire_panel,
    estimer_modele,
    filtrer_periode,
    predire_risque_par_arrondissement,
)

URL_ARRONDISSEMENTS = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "arrondissements/exports/geojson"
)
CENTRE_PARIS = {"lat": 48.8566, "lon": 2.3522}
FICHIER_LOGO = Path(__file__).parent / "rat.png"
TITRE = "Rat-dar"
SOUS_TITRE = "Ou comment éviter les rats dans Paris"
# prévision météo gratuite et sans clé ; jour 0 = hier, jour 1 = aujourd'hui
URL_PREVISION_METEO = "https://api.open-meteo.com/v1/forecast"
PARAMETRES_PREVISION = {
    "latitude": CENTRE_PARIS["lat"],
    "longitude": CENTRE_PARIS["lon"],
    "daily": "temperature_2m_max,precipitation_sum",
    "timezone": "Europe/Paris",
    "past_days": 1,
    "forecast_days": 1,
}
# carte des zones à risque : indice = attendu du carreau / moyenne de Paris, 1 = carreau moyen
# échelle divergente : bleu sous la moyenne, gris à la moyenne, rouge au-dessus ;
# chaque bras va du clair au foncé avec des luminosités symétriques (vérifiées avec la skill dataviz)
ECHELLE_INDICE = ["#1c5cab", "#6da7ec", "#f0efec", "#fb6a4a", "#a50f15"]
# bornes symétriques autour de 1 pour que le gris tombe exactement sur la moyenne
BORNES_INDICE = (0, 2)
# années proposées dans l'onglet des foyers
ANNEES_FOYERS = list(range(2015, 2026))
# paramètres dbscan, fixes pour comparer les années entre elles (choix justifié dans remarques_donnees.md) :
# en 2025, la distance médiane d'un signalement à son 4e voisin est de 117 m ; à 100 m, le plus gros
# foyer d'une année reste sous 6 % des signalements (17 % à 200 m, où les foyers fusionnent)
EPS_FOYERS_M = 100
MIN_POINTS_FOYER = 5


# contours des arrondissements
def charger_arrondissements() -> dict:
    reponse = requests.get(URL_ARRONDISSEMENTS, timeout=30)
    # stop si erreur http
    reponse.raise_for_status()
    return reponse.json()


def charger_modele(rats: pd.DataFrame):
    panel = construire_panel(rats, charger_meteo(FICHIER_METEO))
    return estimer_modele(filtrer_periode(panel, DATE_DEBUT))


def charger_meteo_du_jour() -> dict:
    reponse = requests.get(URL_PREVISION_METEO, params=PARAMETRES_PREVISION, timeout=30)
    reponse.raise_for_status()
    jours = reponse.json()["daily"]
    if len(jours["time"]) != 2:
        raise ValueError("la météo doit contenir hier et aujourd'hui")
    aujourdhui = pd.Timestamp(jours["time"][1])
    return {
        "date": aujourdhui,
        "temp_max_c": jours["temperature_2m_max"][1],
        # hier est terminé : cumul complet de la journée
        "pluie_hier_mm": jours["precipitation_sum"][0],
        "pluie_veille": jours["precipitation_sum"][0] >= SEUIL_PLUIE_MM,
        "saison": SAISON_PAR_MOIS[aujourdhui.month],
    }


def calculer_risque(modele, temp_max_c: float, pluie_veille: bool, saison: str) -> pd.DataFrame:
    predictions = predire_risque_par_arrondissement(modele, temp_max_c, pluie_veille, saison)
    return pd.DataFrame(
        {
            # numéros en texte, sinon carte sans couleurs
            "arrondissement": predictions["ARRONDISSEMENT"].astype(str),
            "risque": predictions["nb_signalements_predits"],
        }
    )


# carte de paris, couleur selon le risque
def tracer_carte(geojson: dict, risques: pd.DataFrame):
    figure = px.choropleth_map( #choropleth = carte avec zones ccolorées
        risques,
        geojson=geojson,#les contouts de l'arrondissment 
        locations="arrondissement",
        # lien tableau / polygones via le numéro d'arrondissement
        featureidkey="properties.c_ar",
        color="risque",
        color_continuous_scale=["white", "red"],
        # échelle fixe : les couleurs restent comparables d'un jour à l'autre
        # (les prédictions vont de 0 à 0,41 pour des températures de -2 à 43 °C)
        range_color=(0, 0.45),
        center=CENTRE_PARIS,
        zoom=10.8,
        opacity=0.75,
        map_style="carto-positron",
        labels={"risque": "Signalements attendus", "arrondissement": "Arrondissement"},
    )
    # carte sans marges
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}) #left, right, top, bottom
    return figure


def calculer_carreaux(rats: pd.DataFrame) -> pd.DataFrame:
    carreaux, _ = preparer_carreaux(rats)
    # une valeur ajustée négative est ramenée à 0, comme pour les prédictions par arrondissement
    carreaux["attendu"] = np.maximum(0, estimer_modele_spatial(carreaux).fittedvalues).round(2)
    # attendu / moyenne = (probabilité qu'un signalement de Paris tombe dans ce carreau)
    #                      / (même probabilité pour un carreau moyen, 1 / nombre de carreaux)
    carreaux["indice"] = (carreaux["attendu"] / carreaux["attendu"].mean()).round(2)
    return carreaux


def carreaux_en_geojson(carreaux: pd.DataFrame) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": carreau.carreau,
                # (lon, lat), polygone fermé : le dernier point répète le premier
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [carreau.lon_min, carreau.lat_min],
                        [carreau.lon_max, carreau.lat_min],
                        [carreau.lon_max, carreau.lat_max],
                        [carreau.lon_min, carreau.lat_max],
                        [carreau.lon_min, carreau.lat_min],
                    ]],
                },
            }
            for carreau in carreaux.itertuples()
        ],
    }


def tracer_carreaux(carreaux: pd.DataFrame):
    figure = px.choropleth_map(
        carreaux,
        geojson=carreaux_en_geojson(carreaux),
        locations="carreau",
        color="indice",
        color_continuous_scale=ECHELLE_INDICE,
        range_color=BORNES_INDICE,
        hover_data={"indice": ":.2f", "attendu": True, CIBLE: True},
        center=CENTRE_PARIS,
        zoom=10.8,
        opacity=0.7,
        map_style="carto-positron",
        labels={
            "indice": "Risque / moyenne de Paris",
            "attendu": "Signalements attendus (2016-2025)",
            CIBLE: "Signalements observés (2016-2025)",
            "carreau": "Carreau",
        },
    )
    # pas de bordure : 2 799 contours rendraient la carte illisible
    figure.update_traces(marker_line_width=0)
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={
            "title": "Risque par rapport<br>à la moyenne de Paris",
            "tickvals": [0, 0.5, 1, 1.5, 2],
            "ticktext": ["0", "× 0,5", "moyenne", "× 1,5", "× 2"],
        },
    )
    return figure


# --- ajout dbscan ---
def signalements_de_l_annee(rats: pd.DataFrame, annee: int) -> pd.DataFrame:
    selection = rats[rats["DATEDECL"].dt.year == annee]
    # dans les données, X = longitude et Y = latitude
    signalements = selection.rename(columns={"Y": "lat", "X": "lon", "DATEDECL": "date"})
    return signalements[["date", "lat", "lon"]].reset_index(drop=True)


# points sur la carte, foyers en rouge et bruit en gris
def tracer_foyers(signalements: pd.DataFrame):
    figure = px.scatter_map(
        signalements,
        lat="lat",
        lon="lon",
        color="type",
        color_discrete_map={"foyer": "red", "isolé": "grey"},
        # numéro du foyer et date du signalement visibles au survol
        hover_data={"groupe": True, "date": "|%d/%m/%Y", "type": False},
        center=CENTRE_PARIS,
        zoom=10.8,
        map_style="carto-positron",
        labels={"type": "Signalement", "groupe": "Foyer", "date": "Date"},
    )
    figure.update_traces(marker={"size": 8})
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return figure
# --- fin ajout dbscan ---


# doit être la première commande streamlit : titre et icône de l'onglet du navigateur
st.set_page_config(page_title=TITRE, page_icon=str(FICHIER_LOGO))

# logo à gauche du titre
colonne_logo, colonne_titre = st.columns([1, 6], vertical_alignment="center")
colonne_logo.image(str(FICHIER_LOGO), width=90)
colonne_titre.title(TITRE)
colonne_titre.subheader(SOUS_TITRE)

# lus une seule fois par affichage, puis passés aux onglets
rats = charger_signalements(FICHIER_RATS_COMPLET)

onglet_risque, onglet_carreaux, onglet_dbscan = st.tabs(
    ["Prévision du jour", "Les zones à risque", "Foyers (DBSCAN)"]
)

with onglet_risque:
    meteo = charger_meteo_du_jour()
    st.subheader(f"Prévision pour aujourd'hui, {meteo['date']:%d/%m/%Y}")
    st.caption(
        f"Météo Open-Meteo : {meteo['temp_max_c']} °C maximum aujourd'hui, "
        f"{meteo['pluie_hier_mm']} mm de pluie hier, saison : {meteo['saison']}. "
        "Valeurs = nombre de signalements attendus par arrondissement (régression linéaire, données 2016-2025)."
    )
    st.info(
        "**Sur quelles données ?** Régression linéaire estimée sur le nombre de signalements "
        "**par arrondissement et par jour**, du 01/01/2016 au 31/12/2025 (3 653 jours × 20 arrondissements). "
        "Variables : température maximale du jour, pluie de la veille, saison et arrondissement.  \n"
        "**Différence avec l'onglet « Les zones à risque » :** cette carte change chaque jour avec la météo, "
        "mais elle ne connaît pas la géographie fine (eau, espaces verts) : chaque arrondissement "
        "est résumé par son niveau moyen."
    )

    risques = calculer_risque(charger_modele(rats), meteo["temp_max_c"], meteo["pluie_veille"], meteo["saison"])
    st.plotly_chart(tracer_carte(charger_arrondissements(), risques), width="stretch")

with onglet_carreaux:
    carreaux = calculer_carreaux(rats)
    moyenne_paris = f"{carreaux['attendu'].mean():.1f}".replace(".", ",")
    st.caption(
        "Risque de chaque carreau de 200 m rapporté à la moyenne de Paris : signalements attendus dans le "
        f"carreau sur 2016-2025, divisés par la moyenne des 2 799 carreaux ({moyenne_paris} signalements). "
        "C'est aussi la probabilité qu'un signalement de Paris tombe dans ce carreau, comparée à celle d'un "
        "carreau moyen. 1 = comme la moyenne (gris), 2 = deux fois plus (rouge), 0,5 = deux fois moins (bleu). "
        "Valeurs attendues : régression de modele_spatial.py, négatifs ramenés à 0. Ce n'est pas une "
        "prévision pour aujourd'hui."
    )
    st.info(
        "**Sur quelles données ?** Régression linéaire estimée sur le nombre **total** de signalements "
        "de 2016 à 2025 dans chacun des 2 799 carreaux de 200 m qui touchent Paris. "
        "Variables : distance à l'eau, distance à l'espace vert le plus proche, centre dans un espace vert, "
        "centre dans un bois, part du carreau dans Paris.  \n"
        "**Différence avec l'onglet « Prévision du jour » :** cette carte est beaucoup plus fine et tient "
        "compte de la géographie, mais elle ne change pas d'un jour à l'autre : le modèle n'utilise "
        "ni la date ni la météo."
    )
    st.plotly_chart(tracer_carreaux(carreaux), width="stretch")

# --- ajout dbscan ---
with onglet_dbscan:
    # une année à la fois : sur plusieurs années, les foyers fusionnent en quelques foyers géants
    annee = st.select_slider("Année", options=ANNEES_FOYERS, value=ANNEES_FOYERS[-1])
    st.caption(
        f"Foyer = au moins {MIN_POINTS_FOYER} signalements de l'année à moins de {EPS_FOYERS_M} m les uns des autres "
        "(DBSCAN), mêmes paramètres pour toutes les années."
    )

    signalements = signalements_de_l_annee(rats, annee)
    signalements["groupe"] = regrouper_points(signalements[["lat", "lon"]], EPS_FOYERS_M, MIN_POINTS_FOYER)
    # -1 = bruit d'après dbscan
    signalements["type"] = np.where(signalements["groupe"] == -1, "isolé", "foyer")

    nb_foyers = signalements.loc[signalements["groupe"] != -1, "groupe"].nunique()
    nb_isoles = (signalements["groupe"] == -1).sum()
    st.write(f"{annee} : {len(signalements)} signalements, {nb_foyers} foyers trouvés, {nb_isoles} signalements isolés")

    st.plotly_chart(tracer_foyers(signalements), width="stretch")
# --- fin ajout dbscan ---
