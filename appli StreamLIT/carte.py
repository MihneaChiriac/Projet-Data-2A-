import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --- ajout dbscan ---
from dbscan import regrouper_points
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


# contours des arrondissements
def charger_arrondissements() -> dict:
    reponse = requests.get(URL_ARRONDISSEMENTS, timeout=30)
    # stop si erreur http
    reponse.raise_for_status()
    return reponse.json()


def charger_modele():
    """Estime la régression de prediction.py sur les données depuis DATE_DEBUT."""
    panel = construire_panel(charger_signalements(FICHIER_RATS_COMPLET), charger_meteo(FICHIER_METEO))
    return estimer_modele(filtrer_periode(panel, DATE_DEBUT))


def charger_meteo_du_jour() -> dict:
    """Variables du modèle pour aujourd'hui : date, température max prévue, pluie d'hier (la veille) et saison."""
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
    """Signalements attendus par arrondissement, au format attendu par tracer_carte."""
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


# --- ajout dbscan ---
# signalements bidon : quelques foyers serrés + points éparpillés dans paris
def creer_signalements_factices() -> pd.DataFrame:
    generateur = np.random.default_rng(seed=42)
    centres_foyers = [(48.8590, 2.3470), (48.8675, 2.3637), (48.8840, 2.3430)]
    # 30 points autour de chaque centre, écart d'environ 60 à 90 m
    points_foyers = [
        generateur.normal(loc=centre, scale=0.0008, size=(30, 2))
        for centre in centres_foyers
    ]
    # 60 points au hasard dans un rectangle autour de paris
    points_eparpilles = np.column_stack(
        [
            generateur.uniform(48.83, 48.89, size=60),
            generateur.uniform(2.28, 2.41, size=60),
        ]
    )
    points = np.vstack([*points_foyers, points_eparpilles])
    return pd.DataFrame(points, columns=["lat", "lon"])


# points sur la carte, foyers en rouge et bruit en gris
def tracer_foyers(signalements: pd.DataFrame):
    figure = px.scatter_map(
        signalements,
        lat="lat",
        lon="lon",
        color="type",
        color_discrete_map={"foyer": "red", "isolé": "grey"},
        # numéro du foyer visible au survol
        hover_data={"groupe": True, "type": False},
        center=CENTRE_PARIS,
        zoom=10.8,
        map_style="carto-positron",
        labels={"type": "Signalement", "groupe": "Foyer"},
    )
    figure.update_traces(marker={"size": 8})
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return figure
# --- fin ajout dbscan ---


st.title("Risque par arrondissement pour les rats à Paris")

onglet_risque, onglet_dbscan = st.tabs(["Risque par arrondissement", "Foyers (DBSCAN)"])

with onglet_risque:
    meteo = charger_meteo_du_jour()
    st.subheader(f"Prévision pour aujourd'hui, {meteo['date']:%d/%m/%Y}")
    st.caption(
        f"Météo Open-Meteo : {meteo['temp_max_c']} °C maximum aujourd'hui, "
        f"{meteo['pluie_hier_mm']} mm de pluie hier, saison : {meteo['saison']}. "
        "Valeurs = nombre de signalements attendus par arrondissement (régression linéaire, données 2016-2025)."
    )

    risques = calculer_risque(charger_modele(), meteo["temp_max_c"], meteo["pluie_veille"], meteo["saison"])
    st.plotly_chart(tracer_carte(charger_arrondissements(), risques), width="stretch")

# --- ajout dbscan ---
with onglet_dbscan:
    st.warning("Signalements factices")

    eps_metres = st.slider("Rayon eps (mètres)", min_value=20, max_value=500, value=100, step=10)
    min_points = st.slider("Nombre minimum de points par foyer", min_value=2, max_value=20, value=5)

    signalements = creer_signalements_factices()
    signalements["groupe"] = regrouper_points(signalements[["lat", "lon"]], eps_metres, min_points)
    # -1 = bruit d'après dbscan
    signalements["type"] = np.where(signalements["groupe"] == -1, "isolé", "foyer")

    nb_foyers = signalements.loc[signalements["groupe"] != -1, "groupe"].nunique()
    nb_isoles = (signalements["groupe"] == -1).sum()
    st.write(f"{nb_foyers} foyers trouvés, {nb_isoles} signalements isolés")

    st.plotly_chart(tracer_foyers(signalements), width="stretch")
# --- fin ajout dbscan ---
