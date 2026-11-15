import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ajout dbscan
from DBSCAN import regrouper_points

URL_ARRONDISSEMENTS = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "arrondissements/exports/geojson"
)
CENTRE_PARIS = {"lat": 48.8566, "lon": 2.3522}


# contours des arrondissements, gardés en cache pour pas retélécharger
@st.cache_data
def charger_arrondissements() -> dict:
    reponse = requests.get(URL_ARRONDISSEMENTS, timeout=30)
    # stop si erreur http
    reponse.raise_for_status()
    return reponse.json()


# risque bidon en attendant les vraies données
def creer_risque_factice():
    # seed fixe, mêmes valeurs à chaque lancement
    generateur = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            # numéros en texte, sinon carte sans couleurs
            "arrondissement": [str(numero) for numero in range(1, 21)],
            "risque": generateur.uniform(0, 1, size=20).round(2),
        }
    )


# signalements bidon : quelques foyers serrés + points éparpillés dans paris
def creer_signalements_factices() -> pd.DataFrame:
    generateur = np.random.default_rng(seed=42)
    centres_foyers = [(48.8590, 2.3470), (48.8675, 2.3637), (48.8840, 2.3430)]
    # 30 points autour de chaque centre
    points_foyers = [
        generateur.normal(loc=centre, scale=0.0008, size=(30, 2))
        for centre in centres_foyers
    ]
    # 60 points au hasard dans un rectangle
    points_eparpilles = np.column_stack(
        [
            generateur.uniform(48.83, 48.89, size=60),
            generateur.uniform(2.28, 2.41, size=60),
        ]
    )
    points = np.vstack([*points_foyers, points_eparpilles])
    return pd.DataFrame(points, columns=["lat", "lon"])



# carte de paris, couleur selon le risque
def tracer_carte(geojson: dict, risques: pd.DataFrame):
    figure = px.choropleth_map( #choropleth = carte avec zones ccolorées
        risques,
        geojson=geojson, #les contouts de l'arrondissment 
        locations="arrondissement",
        # lien tableau / polygones via le numéro d'arrondissement
        featureidkey="properties.c_ar",
        color="risque",
        color_continuous_scale=["white", "red"],
        # échelle fixe, 0 = blanc et 1 = rouge quelles que soient les valeurs
        range_color=(0, 1),
        center=CENTRE_PARIS,
        zoom=10.8,
        opacity=0.75,
        map_style="carto-positron",
        labels={"risque": "Risque", "arrondissement": "Arrondissement"},
    )
    # carte sans marges
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}) #left, right, top, bottom
    return figure


# ajout dbscan

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


st.title("Risque par arrondissement pour les rats à Paris")
st.warning("Données factices")


onglet_risque, onglet_dbscan = st.tabs(["Risque par arrondissement", "Foyers (DBSCAN)"])



with onglet_risque:
    st.warning("Données factices")

    carte = tracer_carte(charger_arrondissements(), creer_risque_factice())
    st.plotly_chart(carte, width="stretch")


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

